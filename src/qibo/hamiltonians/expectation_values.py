from functools import reduce

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, identity, kron

from qibo import Circuit, gates
from qibo.backends import _check_backend, matrices
from qibo.hamiltonians import Hamiltonian
from qibo.config import raise_error
from qibo.backends.abstract import Backend

_PAULI_MATRICES_SPARSE = {
    "I": identity(2, format="csr"),
    "X": csr_matrix(matrices.X, dtype=complex),
    "Y": csr_matrix(matrices.Y, dtype=complex),
    "Z": csr_matrix(matrices.Z, dtype=complex),
}


def get_expval_hamiltonian(
    circuit: Circuit,
    hamiltonian: (
        Hamiltonian
        | csr_matrix
        | list[tuple[float, str]]
        | dict[str, list[tuple[float, str]]]
    ),
    force_complete_z: bool = True,
    nshots: int | None = None,
    backend: Backend | None = None,
) -> (
    tuple[float, dict[str, tuple[float, float]]]
    | tuple[tuple[float, float], dict[str, tuple[float, float, float]]]
):
    """
    Estimate the expectation value of a Hamiltonian with respect to a quantum circuit.

    The Hamiltonian can be provided in multiple formats (Qibo `Hamiltonian`,
    sparse matrix, list of terms with coefficients and Pauli words, or a dictionary
    grouping commuting terms). Each Pauli word is measured in an appropriate basis,
    and either statevector simulation or shot counts are used to estimate expectation values,
    (and uncertainties in the case of samples).

    Args:
        circuit (Circuit): Quantum circuit preparing the state on which the Hamiltonian is evaluated.
        hamiltonian (Hamiltonian | csr_matrix | list[tuple[float, str]] | dict[str, list[tuple[float, str]]]):
            Hamiltonian to evaluate. Can be provided as:
            - Qibo `Hamiltonian` object. Only possible if ``nshots = None``.
            - Scipy sparse CSR matrix. Only possible if ``nshots = None``.
            - List of terms in the linear combination of Pauli words,
                in the format: ``[(coef, pauli_word)]``.
            - Dictionary with grouped mutually-commuting terms
                in the format: ``{pauli_word_measure: [(coef, pauli_word)]}``.
        force_complete_z (bool): If True, force that one of the paulis to be measured is "ZZ...ZZ",
            then group the rest - only relevant is grouping is performed.
        nshots (int | None, optional): Number of measurement shots.
            If ``None``, returns infinite precision expectation values (no sampling noise).
            Defaults to ``None``.
        backend (Backend, optional): backend to be used in the execution.
            If ``None``, it uses the current backend. Defaults to ``None``.

    Returns:
        tuple:
            If `nshots` is ``None``:
                - expval (float): Exact expectation value of the Hamiltonian.
                - results (dict[str, tuple[float, float]]): Mapping from Pauli string to a tuple
                    `(coef, pauli_expval)`, where `pauli_expval` is the expected value of the
                    respective individual Pauli word.

            If `nshots` is an integer:
                - (expval, expval_SE) (tuple[float, float]): Estimated expectation value and
                    standard error of the Hamiltonian.
                - results (dict[str, tuple[float, float, float]]): Mapping from Pauli string to a
                    tuple `(coef, pauli_expval, pauli_expval_SE)`, where `pauli_expval_SE` is the
                    standard error for the expected value of the respective individual Pauli word.

    Example:

        .. testcode::

                from qibo.hamiltonians.expectation_values import get_expval_hamiltonian
                from qibo.quantum_info.random_ensembles import random_clifford
                from scipy.sparse import csr_matrix

                nqubits = 4
                circuit = random_clifford(nqubits)

                # different input otions
                # IMPORTANT: with finite sampling, it's only possible to pass list of terms or grouped terms dict!
                hamiltonian = qibo.hamiltonians.XXZ(nqubits=nqubits)
                hamiltonian = csr_matrix(qibo.hamiltonians.XXZ(nqubits=nqubits).matrix)
                hamiltonian = [
                    (1.0, "XXII"),
                    (1.0, "IXXI"),
                    (1.0, "IIXX"),
                    (1.0, "XIIX"),
                    (1.0, "YYII"),
                    (1.0, "IYYI"),
                    (1.0, "IIYY"),
                    (1.0, "YIIY"),
                    (0.5, "ZZII"),
                    (0.5, "IZZI"),
                    (0.5, "IIZZ"),
                    (0.5, "ZIIZ"),
                ]
                hamiltonian = {
                    "XXXX": [(1.0, "IIXX"), (1.0, "IXXI"), (1.0, "XIIX"), (1.0, "XXII")],
                    "YYYY": [(1.0, "IIYY"), (1.0, "IYYI"), (1.0, "YIIY"), (1.0, "YYII")],
                    "ZZZZ": [(0.5, "IIZZ"), (0.5, "IZZI"), (0.5, "ZIIZ"), (0.5, "ZZII")],
                }

                nhots = None
                expval, expval_terms = get_expval_hamiltonian(
                    circuit,
                    hamiltonian,
                    nshots,
                )

                nhots = int(1e6)
                (expval, expval_SE), expval_terms = get_expval_hamiltonian(
                    circuit,
                    hamiltonian,
                    nshots,
                )

    """
    backend = _check_backend(backend)

    if isinstance(hamiltonian, Hamiltonian):
        if nshots:
            raise_error(
                ValueError,
                f"Hamiltonian type {type(hamiltonian)} does not support `nshots != None`!",
            )
        state = backend.execute_circuit(circuit).state()
        return hamiltonian.expectation(state), None
    elif isinstance(hamiltonian, csr_matrix):
        if nshots:
            raise_error(
                ValueError,
                f"Hamiltonian type {type(hamiltonian)} does not support `nshots != None`!",
            )
        state = backend.execute_circuit(circuit).state()
        return (backend.np.conj(state) @ hamiltonian @ state).real, None
    elif isinstance(hamiltonian, list):
        return _get_expval_hamilt_list_of_terms(
            circuit,
            hamiltonian,
            force_complete_z,
            nshots,
            backend,
        )
    elif isinstance(hamiltonian, dict):
        return _get_expval_hamilt_dict_grouped_terms(
            circuit,
            hamiltonian,
            nshots,
            backend,
        )
    else:
        raise raise_error(
            TypeError, f"Hamiltonian type {type(hamiltonian)} not suported!"
        )


def _pauli_word_to_sparse(pauli_word: str) -> csr_matrix:
    """Converts a Pauli word represented as a string into its sparse matrix representation.

    Args:
        pauli_word (str): A string like "XIYZI" representing the Pauli word.

    Returns:
        csr_matrix: A sparse matrix representing the Pauli word.
    """
    if not all(p in _PAULI_MATRICES_SPARSE for p in pauli_word):
        raise ValueError(
            "Invalid Pauli word. Allowed characters are 'I', 'X', 'Y', 'Z'."
        )

    pauli_word_sparse = reduce(
        lambda a, b: kron(a, b, format="csr"),
        (_PAULI_MATRICES_SPARSE[ch] for ch in pauli_word),
    )

    return pauli_word_sparse


def _get_expval_hamilt_list_of_terms_inf_prec(
    circuit: Circuit,
    hamilt_terms_list: list[tuple[float, str]],
    backend: Backend | None = None,
) -> tuple[float, dict[str, tuple[float, float]]]:
    """Computes the the expected value of the observable represented as a list [(coef, pauli_word)]
    of the terms in the linear combination of Pauli words with respect to the state prepared by
    the circuit, using statevector simulation (i.e., with infinite precision).
    Returns the numeric expected value as well as a dictionary with the expected values per term.

    Args:
        circuit (Circuit): Quantum circuit preparing the state with which the expected value will
            be computed.
        hamilt_terms_list (list[tuple[float, str]]): Observable whose expected value
            will be calculated, explicitly represented as a list of terms in the linear combination
            of Pauli words [(coef, pauli_word)].
        backend (Backend, optional): backend to be used in the execution.
            If ``None``, it uses the current backend. Defaults to ``None``.

    Returns:
        tuple[float, dict[str, tuple[float, float]]]: numeric expected value and a dictionary whose
        keys are the Pauli words, with a tuple of coef and respective Pauli word expval.
    """

    backend = _check_backend(backend)

    state = backend.execute_circuit(circuit).state()

    expval_terms = {}
    for coef, pauli_word in hamilt_terms_list:
        expval_pauli = (
            backend.np.conj(state) @ _pauli_word_to_sparse(pauli_word) @ state
        ).real
        expval_terms[pauli_word] = (coef, expval_pauli)

    expval = sum(coef * expval_pauli for coef, expval_pauli in expval_terms.values())

    return expval, expval_terms


def _pauli_word_std_str_to_symbolic_str(pauli_word: str) -> str:
    """Converts a Pauli word string representation from standard to symbolic expression format.
    Exaple: "XIXYZI" -> "X0 X2 Y3 Z4"

    Args:
        pauli_word (str): Pauli word in standard format, example: "XIXYZI".

    Returns:
        str: Pauli word in symbolic format, example: "X0 X2 Y3 Z4".
    """
    out_str = ""
    for i, p in enumerate(pauli_word):
        if p != "I":
            out_str += f"{p}{i} "
    return out_str.strip()


def _pauli_word_symbolic_str_to_std_str(pauli_word: str, nqubits: int) -> str:
    """Converts a Pauli word string representation from symbolic expression to standard format.
    Example: "X0 X2 Y3 Z4" -> "XIXYZI"

    Args:
        pauli_word (str): Pauli word in symbolic expression format, example: "X0 X2 Y3 Z4".
        nqubits (int): Number of qunits (length of pauli word), necessary to insert identities.

    Returns:
        str: Pauli word in symbolic format, example: "XIXYZI".
    """
    out_str = ["I"] * nqubits
    for term in pauli_word.split():
        pauli = term[:1]
        idx = int(term[1:])
        out_str[idx] = pauli
    return "".join(out_str)


def _check_terms_commutativity(term1: str, term2: str, qubitwise: bool) -> bool:
    """
    Check if terms 1 and 2 are mutually commuting. The 'qubitwise' flag determines if the check is
    for general commutativity (False), or the stricter qubitwise commutativity.

    Args:
        term1/term2: Strings representing a single Pauli term. E.g. "X0 Z1 Y3".
        Obtained from a Qibo SymbolicTerm as ``" ".join(factor.name for factor in term.factors)``.
        qubitwise (bool): Determines if the check is for general commutativity,
        or the stricter qubitwise commutativity

    Returns:
        bool: Do terms 1 and 2 commute?
    """
    # Get a list of common qubits for each term
    common_qubits = {_term[1:] for _term in term1.split() if _term[0] != "I"} & {
        _term[1:] for _term in term2.split() if _term[0] != "I"
    }
    if not common_qubits:
        return True
    # Get the single Pauli operators for the common qubits for both Pauli terms
    term1_ops = [_op for _op in term1.split() if _op[1:] in common_qubits]
    term2_ops = [_op for _op in term2.split() if _op[1:] in common_qubits]
    if qubitwise:
        # Qubitwise: Compare the Pauli terms at the common qubits. Any difference => False
        return all(_op1 == _op2 for _op1, _op2 in zip(term1_ops, term2_ops))
    # General commutativity:
    # Get the number of single Pauli operators that do NOT commute
    n_noncommuting_ops = sum(_op1 != _op2 for _op1, _op2 in zip(term1_ops, term2_ops))
    # term1 and term2 have general commutativity iff n_noncommuting_ops is even
    return n_noncommuting_ops % 2 == 0


def _group_commuting_paulis(
    hamilt_terms_list: list[tuple[float, str]],
    force_complete_z: bool = True,
) -> list[list[tuple[float, str]]]:
    """Receives as input an observable represented as a list with terms in the linear combination
    of Pauli words, and returns a list with mutually-commuting terms.

    Args:
        hamilt_terms_list (list[tuple[float, str]]): Observable whose terms will be grouped,
            explicitly represented as a list of terms in the linear combination
            of Pauli words [(coef, pauli_word)].
        force_complete_z (bool): If True, force that one of the paulis to be measured is "ZZ...ZZ",
            then group the rest.

    Returns:
        list[list[tuple[float, str]]]: list of lists with each group of mutually-commuting terms
        from the input observable, contatning coeficient and Pauli word [[(coef, pauli_word)]]
    """

    if force_complete_z:
        terms_with_only_z = [
            term for term in hamilt_terms_list if set(term[-1]) in [{"I", "Z"}, {"I"}, {"Z"}]
        ]
        terms_groups_z = [terms_with_only_z]
        # update terms to be grouped, removing the ones with only z
        hamilt_terms_list = [
            term for term in hamilt_terms_list if term not in terms_with_only_z
        ]

    else:
        terms_groups_z = []

    # sorting by pauli wprds seems to help getting less groupings
    hamilt_terms_list = sorted(hamilt_terms_list, key=lambda x: x[-1])

    # get pauli words in symbolic notation "XIXYZ" -> "X0 X2 Y3 Z4"
    pauli_terms = [
        _pauli_word_std_str_to_symbolic_str(pauli_word)
        for pauli_word in [term[-1] for term in hamilt_terms_list]
    ]

    G = nx.Graph()
    G.add_nodes_from(pauli_terms)

    # Solving for the minimum clique cover is equivalent to the
    # graph colouring problem for the complement graph
    G.add_edges_from(
        (term1, term2)
        for _i1, term1 in enumerate(pauli_terms)
        for _i2, term2 in enumerate(pauli_terms)
        if _i2 > _i1 and not _check_terms_commutativity(term1, term2, qubitwise=True)
    )

    sorted_groups = nx.coloring.greedy_color(G)
    group_ids = set(sorted_groups.values())
    term_groups = [
        [group for group, group_id in sorted_groups.items() if group_id == _id]
        for _id in group_ids
    ]

    # go back to standard strings
    nqubits = len(hamilt_terms_list[0][-1])
    term_groups = [
        [
            _pauli_word_symbolic_str_to_std_str(pauli_word, nqubits)
            for pauli_word in group
        ]
        for group in term_groups
    ]

    # bring the coefs
    map_pauli_word_coef = {pauli_word: coef for coef, pauli_word in hamilt_terms_list}
    term_groups = terms_groups_z + [
        [(map_pauli_word_coef.get(pauli_word), pauli_word) for pauli_word in group]
        for group in term_groups
    ]

    return term_groups


def _get_measure_pauli_from_commuting_terms(group: list[tuple[float, str]]) -> str:
    """Extract the measurement basis word from a group of mutually commuting Pauli words.

    For a set of Pauli words that commute qubitwise, this function determines the
    single measurement basis needed to evaluate all terms. Each qubit position will
    use the first non-identity Pauli operator encountered across all input words.

    Args:
        pauli_words (list): List of grouped terms [(coef, pauli_word)] containing
            mutually commuting Pauli words.
            Example: [(1.0, 'XXII'), (1.0, 'IXXI'), (1.0, 'IIXX')]

    Returns:
        str: Pauli word string representing the measurement basis for each qubit.
        Only contains the essential non-identity operators needed for measurement.
        Example: "XXXX"
    """

    measurement_basis = {}

    n_qubits = len(group[0][-1])

    for _, pauli_word in group:
        if pauli_word == "I" * n_qubits:
            continue
        for qubit_position, pauli_operator in enumerate(pauli_word):
            if pauli_operator == "I" or qubit_position in measurement_basis:
                continue
            measurement_basis[qubit_position] = pauli_operator

    # force "I" in qubits without measurement
    measurement_basis = {i: measurement_basis.get(i, "I") for i in range(n_qubits)}

    return "".join(list(measurement_basis.values()))


def _get_commuting_grouped_terms_from_hamilt_list_terms(
    hamilt_terms_list: list[tuple[float, str]],
    force_complete_z: bool = True,
) -> dict[str, list[tuple[float, str]]]:
    """Builds a dictionary with the groups of measurements and respective goruped terms
    for the input operator. Output structure: {pauli_word_measure: [(coef, pauli_word)]}.

    Args:
        hamilt_terms_list (list[tuple[float, str]]): Observable whose terms will be grouped,
            explicitly represented as a list of terms in the linear combination
            of Pauli words [(coef, pauli_word)].
        force_complete_z (bool): If True, force that one of the paulis to be measured is "ZZ...ZZ",
            then group the rest.

    Returns:
        dict[str, list[tuple[float, str]]]: Dictionary with the groups of measurements for the
        input operator. In the keys, we have the Pauli word to be measured,
        and in the values there are lists of the commuting terms in the format [(coef, pauli_word)]
    """

    groups_commuting_paulis = _group_commuting_paulis(
        hamilt_terms_list, force_complete_z
    )

    groupings_measurement = {
        _get_measure_pauli_from_commuting_terms(group): group
        for group in groups_commuting_paulis
    }

    return groupings_measurement


def _measure_circuit_pauli_word_operator(
    qc: Circuit, pauli_word_operator: str
) -> Circuit:
    """Returns a circuit with measurements in appropriate basis for the specified Pauli word.
    Will measure each qubit on X, Y, or Z basis according to the respective Pauli.

    Args:
        qc (Circuit): Circuit to append meeasurements.
        pauli_word_operator (str): Pauli word represented as a string,
        indicating the measurement to be made.

    Returns:
        Circuit: Circuit with measurements according to specified Pauli word.
    """

    qc = qc.copy()

    for qubit, pauli in zip(range(qc.nqubits), pauli_word_operator):

        if pauli == "X":
            qc.add(gates.H(qubit))
        elif pauli == "Y":
            qc.add(gates.S(qubit).dagger())
            qc.add(gates.H(qubit))

        qc.add(gates.M(qubit))

    return qc


def _get_expval_hamilt_dict_grouped_terms_samples(
    circuit: Circuit,
    grouped_hamilt_dict: dict[str, list[tuple[float, str]]],
    nshots: int,
    backend: Backend | None = None,
) -> tuple[tuple[float, float], dict[str, tuple[float, float, float]]]:
    """Computes the expected value of a Hamiltonian represented as a dictionary of
    mutually-commuting grouped terms, with respect to the samples produced by the input circuit,
    controlled by the number of shots specified by `nshots`.
    This function assumes that the Hamiltonian was grouped into mutually-commuting terms.

    Args:
        circuit (Circuit): Quantum circuit preparing the state with which the expected value will
            be computed.
        grouped_hamilt_dict (dict[str, list[tuple[float, str]]]): Observable whose expected value
            will be calculated, explicitly represented as a Dictionary with
            grouped mutually-commuting terms {pauli_word_measure: [(coef, pauli_word)]}.
        nshots (int): Number of shots (samples) for each measurement.
        backend (Backend, optional): backend to be used in the execution.
            If ``None``, it uses the current backend. Defaults to ``None``.

    Returns:
        tuple[tuple[float, float], dict[str, tuple[float, float, float]]]: Returns first a tuple
        with the computed expected value and the respective standard error, then a
        dictionary whose keys are the Pauli words, and tuple of coef, expval and std error as value.
    """

    backend = _check_backend(backend)

    expval = 0.0
    var_total = 0.0

    expval_terms = {}

    for measured_pauli, group in grouped_hamilt_dict.items():

        circuit_measd_pauli = _measure_circuit_pauli_word_operator(
            circuit, measured_pauli
        )
        result_measd_pauli = circuit_measd_pauli(nshots=nshots)
        counts_measd_pauli = result_measd_pauli.frequencies(binary=True)

        # array with each bitstring measured (used below)
        # each column is one qubit, each row is one unique measured bitstring
        # i transform to integer to sum it easily below
        measured_bitstrings_array = backend.cast(
            [[int(b) for b in bitstr] for bitstr in counts_measd_pauli.keys()],
            dtype=backend.np.int32,
        )
        # counts of each measured bitstring - will be used below as weights
        counts_weights = backend.cast(
            list(counts_measd_pauli.values()), dtype=backend.np.int32
        )
        nshots_group = counts_weights.sum()

        # create a matrix of eigenvalues for each Pauli word given the measured bitstrings
        eigenvals_given_bitstring = []
        coefs = []

        for coef, pauli_word in group:

            # indices of qubits that are not identity - i used this as mask
            # to get the 1s in qubits pisitions which are not I below
            mask = backend.cast([p != "I" for p in pauli_word], dtype=backend.np.bool_)

            # eigenvalue: (-1)^(sum of bits at non-identity positions)
            vals = (-1) ** backend.np.sum(measured_bitstrings_array[:, mask], axis=1)
            eigenvals_given_bitstring.append(vals)
            coefs.append(coef)

        # in rows: each pauli in the group
        # in columns: the eigenvalue of the respective measured bitstring
        eigenvals_given_bitstring = backend.cast(
            eigenvals_given_bitstring, dtype=backend.np.float64
        )
        coefs = backend.cast(coefs, dtype=backend.np.float64)

        # let's get the average using the counts as weights!
        mu = backend.np.average(
            eigenvals_given_bitstring, axis=1, weights=counts_weights
        )
        cov_pop = (
            eigenvals_given_bitstring * counts_weights
        ) @ eigenvals_given_bitstring.T / nshots_group - backend.np.outer(mu, mu)

        # variance of the mean: divide by total shots in this group
        cov_mean = cov_pop / nshots_group

        # acumulate expectation value and variance
        expval += coefs @ mu
        var_total += coefs @ cov_mean @ coefs

        # store results per term
        for i, (coef, pauli_word) in enumerate(group):
            exp = mu[i]
            stderr = np.sqrt(((cov_mean @ coefs)[i]).sum())
            # stderr = np.sqrt(cov_mean[i, i])
            expval_terms[pauli_word] = (coef, exp, stderr)

    # final standard error
    expval_SE = backend.np.sqrt(var_total)

    return (expval, expval_SE), expval_terms


def _get_expval_hamilt_list_of_terms(
    circuit: Circuit,
    hamilt_terms_list: list[tuple[float, str]],
    force_complete_z: bool = True,
    nshots: int | None = None,
    backend: Backend | None = None,
) -> (
    tuple[float, dict[str, tuple[float, float]]]
    | tuple[tuple[float, float], dict[str, tuple[float, float, float]]]
):
    """Computes the the expected value of the observable represented as a list [(coef, pauli_word)]
    of the terms in the linear combination of Pauli words, either:
    with respect to the samples produced by the input circuit,
    controlled by the number of shots specified by `nshots`;
    or with respect to the state prepared by the circuit, using statevector simulation
    (infinite precision).

    Args:
        circuit (Circuit): Quantum circuit preparing the state with which the expected value will
            be computed.
        force_complete_z (bool): If True, force that one of the paulis to be measured is "ZZ...ZZ",
            then group the rest.
        hamilt_terms_list (list[tuple[float, str]]): Observable whose expected value
            will be calculated, explicitly represented as a list of terms in the linear combination
            of Pauli words [(coef, pauli_word)].
        nshots (int, optional): Number of shots (samples) for each measurement.
            if ``None``, performs infinite precision simulation. Defaults to ``None``.
        backend (Backend, optional): backend to be used in the execution.
            If ``None``, it uses the current backend. Defaults to ``None``.

    Returns:
        tuple[float, dict[str, tuple[float, float]]]: numeric expected value and a dictionary whose
        keys are the Pauli words, with a tuple of coef and respective Pauli word expval.
        -- or --
        tuple[tuple[float, float], dict[str, tuple[float, float, float]]]: returns first a tuple
        with the computed expected value and the respective standard error, then a
        dictionary whose keys are the Pauli words, and tuple of coef, expval and std error as value.
    """

    if nshots is None:
        return _get_expval_hamilt_list_of_terms_inf_prec(
            circuit,
            hamilt_terms_list,
            backend,
        )
    else:
        grouped_hamilt_dict = _get_commuting_grouped_terms_from_hamilt_list_terms(
            hamilt_terms_list, force_complete_z
        )
        return _get_expval_hamilt_dict_grouped_terms_samples(
            circuit,
            grouped_hamilt_dict,
            nshots,
            backend,
        )


def _flatten_list(list_of_lists):
    return [x for sublist in list_of_lists for x in sublist]


def _get_expval_hamilt_dict_grouped_terms(
    circuit: Circuit,
    grouped_hamilt_dict: dict[str, list[tuple[float, str]]],
    nshots: int | None = None,
    backend: Backend | None = None,
) -> (
    tuple[float, dict[str, tuple[float, float]]]
    | tuple[tuple[float, float], dict[str, tuple[float, float, float]]]
):
    """Computes the expected value of a Hamiltonian represented as a dictionary of
    mutually-commuting grouped terms, either:
    with respect to the samples produced by the input circuit,
    controlled by the number of shots specified by `nshots`;
    or with respect to the state prepared by the circuit, using statevector simulation
    (infinite precision).

    Args:
        circuit (Circuit): Quantum circuit preparing the state with which the expected value will
            be computed.
        grouped_hamilt_dict (dict[str, list[tuple[float, str]]]): Observable whose expected value
            will be calculated, explicitly represented as a Dictionary with
            grouped mutually-commuting terms {pauli_word_measure: [(coef, pauli_word)]}.
        nshots (int, optional): Number of shots (samples) for each measurement.
            if ``None``, performs infinite precision simulation. Defaults to ``None``.
        backend (Backend, optional): backend to be used in the execution.
            If ``None``, it uses the current backend. Defaults to ``None``.

    Returns:
        tuple[float, dict[str, tuple[float, float]]]: numeric expected value and a dictionary whose
        keys are the Pauli words, with a tuple of coef and respective Pauli word expval.
        -- or --
        tuple[tuple[float, float], dict[str, tuple[float, float, float]]]: returns first a tuple
        with the computed expected value and the respective standard error, then a
        dictionary whose keys are the Pauli words, and tuple of coef, expval and std error as value.
    """

    if nshots is None:
        hamilt_terms_list = _flatten_list(grouped_hamilt_dict.values())

        return _get_expval_hamilt_list_of_terms_inf_prec(
            circuit,
            hamilt_terms_list,
            backend,
        )
    else:
        return _get_expval_hamilt_dict_grouped_terms_samples(
            circuit,
            grouped_hamilt_dict,
            nshots,
            backend,
        )
