from functools import reduce

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, identity, kron

from qibo import Circuit, gates
from qibo.backends import _check_backend, matrices

PAULI_MATRICES_SPARSE = {
    "I": identity(2, format="csr"),
    "X": csr_matrix(matrices.X, dtype=complex),
    "Y": csr_matrix(matrices.Y, dtype=complex),
    "Z": csr_matrix(matrices.Z, dtype=complex),
}


def pauli_word_to_sparse(pauli_word: str) -> csr_matrix:
    """Converts a Pauli word represented as a string into its sparse matrix representation.

    Args:
        pauli_word (str): A string like "XIYZI" representing the Pauli word.

    Returns:
        csr_matrix: A sparse matrix representing the Pauli word.
    """
    if not all(p in PAULI_MATRICES_SPARSE for p in pauli_word):
        raise ValueError(
            "Invalid Pauli word. Allowed characters are 'I', 'X', 'Y', 'Z'."
        )

    pauli_word_sparse = reduce(
        lambda a, b: kron(a, b, format="csr"),
        (PAULI_MATRICES_SPARSE[ch] for ch in pauli_word),
    )

    return pauli_word_sparse


def get_expval_from_linear_comb_of_paulis_from_statevector(
    circuit: Circuit, lin_comb_pauli: list[tuple[float, str]], backend=None
) -> float:
    """Computes the the expected value of the observable represented as a linear combination
    of Pauli words with respect to the state prepared by the circuit, using statevector simulation
    (i.e., with infinite precision)

    Args:
        circuit (Circuit): Quantum circuit preparing the state with which the expected value will be computed.
        lin_comb_pauli (list[tuple[float, str]]): Observable whose expected value
            will be calculated, explicitly represented as a linear combination of Pauli words
            via a list of tuples [(coef, pauli_word)].
            As an example, the Heisenberg XXZ model Hamiltonian on `n=4` qubits with `delta=0.5` is, in this representation,
            the following list:
            [(1.0, 'XXII'),
            (1.0, 'IXXI'),
            (1.0, 'IIXX'),
            (1.0, 'XIIX'),
            (1.0, 'YYII'),
            (1.0, 'IYYI'),
            (1.0, 'IIYY'),
            (1.0, 'YIIY'),
            (0.5, 'ZZII'),
            (0.5, 'IZZI'),
            (0.5, 'IIZZ'),
            (0.5, 'ZIIZ')]
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.

    Returns:
        float: The computed expected value.
    """

    backend = _check_backend(backend)

    state = circuit().state()

    expval = 0
    for coef, pauli_word in lin_comb_pauli:
        expval += (
            coef
            * (backend.np.conj(state) @ pauli_word_to_sparse(pauli_word) @ state).real
        )

    return expval


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
    !!!!! TAKEN FROM `qibochem.measurement.optimization` !!!!!

    Check if terms 1 and 2 are mutually commuting. The 'qubitwise' flag determines if the check is for general
    commutativity (False), or the stricter qubitwise commutativity.

    Args:
        term1/term2: Strings representing a single Pauli term. E.g. "X0 Z1 Y3". Obtained from a Qibo SymbolicTerm as
            ``" ".join(factor.name for factor in term.factors)``.
        qubitwise (bool): Determines if the check is for general commutativity, or the stricter qubitwise commutativity

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
    lin_comb_pauli: list[tuple[float, str]],
) -> list[list[tuple[float, str]]]:
    """Receives as input an observable represented as a linear combination
    of Pauli words, and returns a list with mutually-commuting terms, useful for

    Args:
        lin_comb_pauli (list[tuple[float, str]]): Observable whose expected value
            will be calculated, explicitly represented as a linear combination of Pauli words
            via a list of tuples [(coef, pauli_word)].
            As an example, the Heisenberg XXZ model Hamiltonian on `n=4` qubits with `delta=0.5` is, in this representation,
            the following list:
            [(1.0, 'XXII'),
            (1.0, 'IXXI'),
            (1.0, 'IIXX'),
            (1.0, 'XIIX'),
            (1.0, 'YYII'),
            (1.0, 'IYYI'),
            (1.0, 'IIYY'),
            (1.0, 'YIIY'),
            (0.5, 'ZZII'),
            (0.5, 'IZZI'),
            (0.5, 'IIZZ'),
            (0.5, 'ZIIZ')]

    Returns:
        list[list[tuple[float, str]]]: list of lists with each group of mutually-commuting terms
            from the input observable, contatning coeficient and Pauli word [[(coef, pauli_word)]]
    """

    # get pauli words in symbolic notation "XIXYZ" -> "X0 X2 Y3 Z4"
    pauli_terms = [
        _pauli_word_std_str_to_symbolic_str(pauli_word)
        for pauli_word in [term[-1] for term in lin_comb_pauli]
    ]

    G = nx.Graph()
    G.add_nodes_from(pauli_terms)

    # Solving for the minimum clique cover is equivalent to the graph colouring problem for the complement graph
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
    nqubits = len(lin_comb_pauli[0][-1])
    term_groups = [
        [
            _pauli_word_symbolic_str_to_std_str(pauli_word, nqubits)
            for pauli_word in group
        ]
        for group in term_groups
    ]

    # bring the coefs
    map_pauli_word_coef = {pauli_word: coef for coef, pauli_word in lin_comb_pauli}
    term_groups = [
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

    for _, pauli_word in group:
        for qubit_position, pauli_operator in enumerate(pauli_word):
            if pauli_operator == "I" or qubit_position in measurement_basis:
                continue
            measurement_basis[qubit_position] = pauli_operator

    return "".join(list(measurement_basis.values()))


def _get_groupings_commuting_terms_from_linear_comb_of_paulis(
    lin_comb_pauli: list[tuple[float, str]],
) -> dict[str, list[tuple[float, str]]]:
    """Builds a dictionary with the groups of measurements and respective goruped terms
    for theinput operator. Output structure: {pauli_word_measure: [(coef, pauli_word)]}.

    Args:
        lin_comb_pauli (list[tuple[float, str]]): Observable whose expected value
            will be calculated, explicitly represented as a linear combination of Pauli words
            via a list of tuples [(coef, pauli_word)].
            As an example, the Heisenberg XXZ model Hamiltonian on `n=4` qubits with `delta=0.5` is, in this representation,
            the following list:
            [(1.0, 'XXII'),
            (1.0, 'IXXI'),
            (1.0, 'IIXX'),
            (1.0, 'XIIX'),
            (1.0, 'YYII'),
            (1.0, 'IYYI'),
            (1.0, 'IIYY'),
            (1.0, 'YIIY'),
            (0.5, 'ZZII'),
            (0.5, 'IZZI'),
            (0.5, 'IIZZ'),
            (0.5, 'ZIIZ')]

    Returns:
        dict[str, list[tuple[float, str]]]: Dictionary with the groups of measurements for the
        input operator. In the keys, we have
        the Pauli word to be measured, whose values are lists of the commuting terms in the format [(coef, pauli_word)]
    """

    groups_commuting_paulis = _group_commuting_paulis(lin_comb_pauli)

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


def _single_shot_pauli_outcome(pauli_word: str, bitstring: str) -> int:
    """Returns the outcome (eigenvalue) of the respective Pauli word given the
    measured bitstring

    Args:
        pauli_word (str): Measured Pauli word
        bitstring (str): Measured bitstring

    Returns:
        int: Eigenvalue (either -1 or +1)
    """
    indices_not_id = [i for i, pauli in enumerate(pauli_word) if pauli != "I"]
    eigenvalue = (-1) ** sum(int(bitstring[i]) for i in indices_not_id)
    return eigenvalue


def get_expval_from_linear_comb_of_paulis_from_samples(
    circuit: Circuit,
    lin_comb_pauli: list[tuple[float, str]] | dict[str, list[tuple[float, str]]],
    nshots: int,
    backend=None,
) -> float:
    """Computes the the expected value of an observable represented as a linear combination
    of Pauli words with respect to the state prepared by the circuit, from the counts resulting
    from finite measurements specified by `nshots`

    Args:
        circuit (Circuit): Quantum circuit preparing the state with which the expected value will be computed.
        lin_comb_pauli (list[tuple[float, str]] | dict[str, list[tuple[float, str]]]): Observable whose expected value
            will be calculated, explicitly represented as a linear combination of Pauli words, which may be
            already grouped in mutually-commuting terms or not, depending on the type of input:

             The input can be provided in two formats:

            1. **Dictionary with grouped mutually-commuting terms {pauli_word_measure: [(coef, pauli_word)]}**.
                Example - Heisenberg XXZ model Hamiltonian on `n=4` qubits with `delta=0.5`:

                    {
                        'XXXX': [(1.0, 'XXII'), (1.0, 'IXXI'), (1.0, 'IIXX'), (1.0, 'XIIX')],
                        'YYYY': [(1.0, 'YYII'), (1.0, 'IYYI'), (1.0, 'IIYY'), (1.0, 'YIIY')],
                        'ZZZZ': [(0.5, 'ZZII'), (0.5, 'IZZI'), (0.5, 'IIZZ'), (0.5, 'ZIIZ')]
                    }

            1. **List of terms [(coef, pauli_word)]**.
               Example - Heisenberg XXZ model Hamiltonian on `n=4` qubits with `delta=0.5`:

                    [(1.0, 'XXII'),
                    (1.0, 'IXXI'),
                    (1.0, 'IIXX'),
                    (1.0, 'XIIX'),
                    (1.0, 'YYII'),
                    (1.0, 'IYYI'),
                    (1.0, 'IIYY'),
                    (1.0, 'YIIY'),
                    (0.5, 'ZZII'),
                    (0.5, 'IZZI'),
                    (0.5, 'IIZZ'),
                    (0.5, 'ZIIZ')]

        nshots (int): Number of shots (samples) for measurement
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.

    Returns:
        float: The computed expected value.
    """

    backend = _check_backend(backend)

    if isinstance(lin_comb_pauli, list):
        # group paulis if not already grouped
        groupings_measurement = (
            _get_groupings_commuting_terms_from_linear_comb_of_paulis(lin_comb_pauli)
        )
    else:
        groupings_measurement = lin_comb_pauli

    expval = 0.0
    var_total = 0.0

    for measured_pauli, group in groupings_measurement.items():

        circuit_measd_pauli = _measure_circuit_pauli_word_operator(
            circuit, measured_pauli
        )
        result_measd_pauli = circuit_measd_pauli(nshots=nshots)
        counts_measd_pauli = result_measd_pauli.frequencies(binary=True)

        # array with each bitstring measured (used below)
        # each column is one qubit, each row is one unique measured bitstring
        # i transform to integer to sum it easily below
        measured_bitstrings_array = np.array(
            [[int(b) for b in bitstr] for bitstr in counts_measd_pauli.keys()]
        )
        # counts of each measured bitstring - will be used below as weights
        counts_weights = np.array(list(counts_measd_pauli.values()), dtype=int)
        nshots_group = counts_weights.sum()

        # create a matrix of eigenvalues for each Pauli word given the measured bitstrings
        eigenvals_given_bitstring = []
        coefs = []

        for coef, pauli_word in group:

            # indices of qubits that are not identity - i used this as mask
            # to get the 1s in qubits pisitions which are not I below
            mask = np.array([p != "I" for p in pauli_word])

            # eigenvalue: (-1)^(sum of bits at non-identity positions)
            vals = (-1) ** np.sum(measured_bitstrings_array[:, mask], axis=1)
            eigenvals_given_bitstring.append(vals)
            coefs.append(coef)

        # in rows: each pauli in the group
        # in columns: the eigenvalue of the respective measured bitstring
        eigenvals_given_bitstring = np.array(eigenvals_given_bitstring, dtype=float)
        coefs = np.array(coefs, dtype=float)

        # let's get the average using the counts as weights!
        mu = np.average(eigenvals_given_bitstring, axis=1, weights=counts_weights)
        cov_pop = (
            eigenvals_given_bitstring * counts_weights
        ) @ eigenvals_given_bitstring.T / nshots_group - np.outer(mu, mu)

        # variance of the mean: divide by total shots in this group
        cov_mean = cov_pop / nshots_group

        # acumulate expectation value and variance
        expval += coefs @ mu
        var_total += coefs @ cov_mean @ coefs

    # final standard error
    expval_SE = np.sqrt(var_total)

    # 95% CI
    expval_95_CI = (expval - 1.96 * expval_SE, expval + 1.96 * expval_SE)

    return expval, expval_SE, expval_95_CI
