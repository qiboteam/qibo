"""Module defining the Hamming-weight-preserving backend."""

from typing import List, Union

import numpy as np
from scipy.special import binom

from qibo import gates
from qibo.config import raise_error


def apply_gate(self, gate, state, nqubits: int, weight: int):
    """Apply ``gate`` to ``state``.

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): gate to apply to ``state``.
        state (ndarray): state to apply ``gate`` to.
        nqubits (int): total number of qubits in ``state``.
        weight (int): fixed Hamming weight of ``state``.

    Returns:
        ndarray: ``state`` after the action of ``gate``.
    """
    if isinstance(gate, gates.M):
        return gate.apply_hamming_weight(self, state, nqubits, weight)

    if isinstance(gate, gates.CCZ):
        # CCZ has a custom apply method because currently it is the only
        # 3-qubit gate that is also Hamming-weight-preserving
        # and this custom method is faster than the n-qubit method
        return self._apply_gate_CCZ(gate, state, nqubits, weight)

    if len(gate.target_qubits) == 1:
        return self._apply_gate_single_qubit(gate, state, nqubits, weight)

    if len(gate.target_qubits) == 2:
        return self._apply_gate_two_qubit(gate, state, nqubits, weight)

    return self._apply_gate_n_qubit(gate, state, nqubits, weight)


def execute_circuit(self, circuit, weight: int, initial_state=None, nshots: int = 1000):
    """Execute ``circuit`` by applying the queue of gates to the ``initial_state``.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): Hamming-weight-preserving circuit
            to be executed.
        weight (int): fixed Hamming weight of the ``initial_state``.
        initial_state (ndarray, optional): initial state that ``circuit`` acts on.
            If ``None``, defaults to :math:`\\ket{0^{n-k} \\, 1^{k}}`,
            with :math:`n` being the total number of qubits in the circuit,
            and :math:`k` being the Hamming ``weight``. Defaults to ``None``.
        nshots (int, optional): total number of shots to simulate when ``circuit``
            contains measurement gates (:class:`qibo.gates.M`). Defaults to :math:`1000`.

    Returns:
        :class:`qibo.quantum_info.hamming_weight.HammingWeightResult`: Object
        containing the results of circuit execution of a Hamming-weight-preserving
        circuit.
    """
    from qibo.quantum_info.hamming_weight import (  # pylint: disable=import-outside-toplevel
        HammingWeightResult,
    )

    if circuit.density_matrix:
        raise_error(RuntimeError, "Density matrix simulation is not supported.")

    for gate in circuit.queue:
        if isinstance(gate, gates.Channel):
            raise_error(RuntimeError, "Circuit must not contain channels.")
        if not gate.hamming_weight and not isinstance(gate, gates.M):
            raise_error(
                RuntimeError,
                "Circuit contains non-Hammming weight preserving  gates.",
            )

    nqubits = circuit.nqubits

    self._dict_indexes = self._get_lexicographical_order(nqubits, weight)

    try:
        if initial_state is None:
            n_choose_k = int(binom(nqubits, weight))
            initial_state = self.np.zeros(n_choose_k)
            initial_state[0] = 1
            initial_state = self.cast(initial_state, dtype=self.dtype)

        state = initial_state
        for gate in circuit.queue:
            state = self.apply_gate(gate, state, nqubits, weight)

        result = HammingWeightResult(
            state,
            weight,
            nqubits,
            measurements=circuit.measurements,
            nshots=nshots,
            engine=self.platform,
        )

        return result

    except self.oom_error:  # pragma: no cover
        raise_error(
            RuntimeError,
            f"State does not fit in {self.device} memory."
            "Please switch the execution device to a "
            "different one using ``qibo.set_device``.",
        )


def _gray_code(self, initial_string):
    """Return all bitstrings of a fixed Hamming weight.

    Uses the ``ehrlich_algorithm`` with an ``initial_string``.

    Args:
        initial_string (ndarray): Array of bits representing the input
            of the Ehrlich algorithm.

    Returns:
        ndarray: All bitstrings with the same Hamming weight as ``initial_string``.
    """
    from qibo.models.encodings import _ehrlich_algorithm  # pylint: disable=C0415

    strings = _ehrlich_algorithm(initial_string, return_indices=False)
    strings = [list(string) for string in strings]
    strings = self.np.asarray(strings, dtype=int)

    return strings


def _get_cached_strings(
    self, nqubits: int, weight: int, ncontrols: int = 0, two_qubit_gate: bool = True
):
    """Generate list of strings necessary for the custom ``apply_gate`` method.

    Given the total number of qubits ``nqubits``, the Hamming ``weight`` to be
    preserved, and the number of controls ``ncontrols`` in the gate, returns a
    sequence of bitstrings for the custom ``apply_gate`` method.

    The sequence is generated for the first gate with a unique combination of ``nqubits``,
    ``weight`` and ``ncontrols``, and then cached to be resued for similar gates.

    Args:
        nqubits (int): total number of qubits in the quantum system.
        weight (int): Hamming weight of the state that gates are acting on.
        ncontrols (int, optional): number of controls in the gate that is being
            applied to the state. Defaults to :math:`0`.
        two_qubit_gate (bool, optional): if ``True``, generate strings assuming the
            gate is a two-qubit gate. If ``False``, it assumes the gate is a single-qubit
            gate. Defaults to ``True``.

    Returns:
        ndarray or list: ndarray of bitstrings for two-qubit gates or a list of two ndarrays
        of bitstrings for single-qubit gates.
    """
    if two_qubit_gate:
        initial_string = self.np.array(
            [1] * (weight - 1 - ncontrols)
            + [0] * ((nqubits - 2 - ncontrols) - (weight - 1 - ncontrols))
        )
        strings = self._gray_code(initial_string)
    else:
        initial_string = self.np.array(
            [1] * (weight - ncontrols)
            + [0] * ((nqubits - 1 - ncontrols) - (weight - ncontrols))
        )
        strings_0 = self._gray_code(initial_string)

        initial_string = self.np.array(
            [1] * (weight - 1 - ncontrols)
            + [0] * ((nqubits - 1 - ncontrols) - max(0, (weight - 1 - ncontrols)))
        )
        strings_1 = self._gray_code(initial_string)

        strings = [strings_0, strings_1]

    return strings


def _get_lexicographical_order(self, nqubits, weight):
    """Sort bistrings generated from ``self._get_cached_strings`` in lexicographical order.

    Bitstrings are sorted in lexicographical (ascending) order.
    Moreover, they are also converted to integers and both representations are cached,
    creating a map between the full statevector representation and the condensed
    Hamming-weight-preserving representation.

    Args:
        nqubits (int): total number of qubits.
        weight (int): Hamming weight of the state.

    Returns:
        dict: Dictionary with the cached bitstrings and the aforementioned map.
    """
    n_choose_k = int(binom(nqubits, weight))
    indexes = list(range(n_choose_k))

    lexicographical_order = self._get_cached_strings(nqubits + 2, weight + 1)
    lexicographical_order = [
        "".join(item.astype(str)) for item in lexicographical_order
    ]
    lexicographical_order.sort()
    lexicographical_order_int = [int(item, base=2) for item in lexicographical_order]
    _dict_indexes = dict(
        zip(lexicographical_order, zip(indexes, lexicographical_order_int))
    )

    return _dict_indexes


def _get_single_qubit_matrix(self, gate):
    """Return non-zero elements of the matrix representation of
    Hamming-weight-preserving single-qubit gates.

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): Hamming-weight-preserving
            single-qubit gate.

    Returns:
        tuple(complex, complex): Non-zero elements of a Hamming-weight-preserving ``gate``.
    """
    matrix = gate.matrix(backend=self)

    if gate.name in ["cz", "crz", "cu1"]:
        matrix = matrix[2:, 2:]

    return self.np.diag(matrix)


def _apply_gate_single_qubit(self, gate, state, nqubits, weight):
    """Custom ``apply_gate`` method for Hamming-weight-preserving single-qubit gates.

    Instead of relying on matrix multiplication, this method applies
    Hamming-weight-preserving single-qubit gates by directly multiplying
    the amplitudes of interest elementwise by the necessary phase(s).

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): Hamming-weight-preserving
            single-qubit gate to be applied to ``state``
        state (ndarray): state to suffer the action of ``gate``.
        nqubits (int): total number of qubits in the circuit.
        weight (int): Hamming weight of ``state``.

    Returns:
        ndarray: ``state`` after the action of ``gate``.
    """
    qubits = list(gate.target_qubits)
    controls = list(gate.control_qubits)
    ncontrols = len(controls)
    other_qubits = list(set(range(nqubits)) ^ set(qubits + controls))

    key_0, key_1 = f"{ncontrols}_0", f"{ncontrols}_1"
    if (
        key_0 not in self._dict_cached_strings_one
        or key_1 not in self._dict_cached_strings_one
    ):
        strings_0, strings_1 = self._get_cached_strings(
            nqubits, weight, ncontrols, False
        )
        self._dict_cached_strings_one[key_0] = strings_0
        self._dict_cached_strings_one[key_1] = strings_1
    else:
        strings_0 = self._dict_cached_strings_one[key_0]
        strings_1 = self._dict_cached_strings_one[key_1]

    matrix_00, matrix_11 = self._get_single_qubit_matrix(gate)

    indexes_one = self.np.zeros((len(strings_1), nqubits), dtype=str)

    if matrix_00 != 1.0 and nqubits - weight > 0:
        indexes_zero = self.np.zeros((len(strings_0), nqubits), dtype=str)
        indexes_zero[:, other_qubits] = strings_0
        indexes_zero[:, qubits] = ["0"]
        if len(controls) > 0:
            indexes_zero[:, controls] = "1"
        indexes_zero = self.np.array(
            [self._dict_indexes["".join(elem)][0] for elem in indexes_zero]
        )

        state[indexes_zero] *= matrix_00

    if matrix_11 != 1.0 and weight - ncontrols > 0:
        indexes_one[:, other_qubits] = strings_1
        if len(controls) > 0:
            indexes_one[:, controls] = "1"

        indexes_one[:, qubits] = ["1"]
        indexes_one = self.np.array(
            [self._dict_indexes["".join(elem)][0] for elem in indexes_one]
        )

        state[indexes_one] *= matrix_11

    return state


def _update_amplitudes(
    self,
    state,
    qubits: List[int],
    controls: List[int],
    other_qubits: List[int],
    weight: int,
    matrix_element: Union[complex, float],
    bitlist: List[str],
    shift: int,
):
    """Update in-place the amplitudes changed by a two-qubit Hamming-weight-preserving gate.

    Args:
        state (ndarray): state that the two-qubit gate acts on.
        qubits (list): target qubits of the gate.
        controls (list): control qubits of the gate.
        other_qubits (list): remaining qubits in the circuit.
        weight (int): Hamming weight of ``state``.
        matrix_element (complex or float): non-zero element of the gate that
            multiplies the amplitudes of ``state``.
        bitlist (list): If the amplitude being updated is associated to the computational basis
            state :math:`\\ket{\\dots00\\dots}`, then ``bitlist=["0", "0"]``.
            If the amplitude is associated to :math:`\\ket{\\dots11\\dots}`,
            then ``bitlist=["1", "1"]``.
        shift (int): shift in the Hamming weight necessary to generate bitstrings.
            If the amplitude being updated is associated to the computational basis
            state :math:`\\ket{\\dots00\\dots}`, then ``shift`` is :math:`1`.
            If the amplitude is associated to :math:`\\ket{\\dots11\\dots}`,
            then ``shift`` is :math:`-1`.

    Returns:
        ndarray: ``state`` after the action of two-qubit Hamming-weight-preserving gate.
    """
    ncontrols = len(controls)
    nqubits = len(qubits) + ncontrols + len(other_qubits)

    strings = self._get_cached_strings(nqubits, weight + shift, ncontrols)
    indexes_in = self.np.zeros((len(strings), nqubits), dtype=str)
    indexes_in[:, other_qubits] = strings
    if ncontrols > 0:
        indexes_in[:, controls] = "1"
    indexes_in[:, qubits] = bitlist
    indexes_in = self.np.array(
        [self._dict_indexes["".join(elem)][0] for elem in indexes_in]
    )
    state[indexes_in] *= matrix_element

    return state


def _apply_gate_two_qubit(self, gate, state, nqubits, weight):
    """Custom ``apply_gate`` method for Hamming-weight-preserving two-qubit gates.

    Instead of relying on matrix multiplication, this method applies
    Hamming-weight-preserving two-qubit gates by directly multiplying
    the amplitudes of interest elementwise by the necessary phase(s).

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): Hamming-weight-preserving
            two-qubit gate to be applied to ``state``
        state (ndarray): state to suffer the action of ``gate``.
        nqubits (int): total number of qubits in the circuit.
        weight (int): Hamming weight of ``state``.

    Returns:
        ndarray: ``state`` after the action of ``gate``.
    """
    qubits = list(gate.target_qubits)
    controls = list(gate.control_qubits)
    ncontrols = len(controls)
    other_qubits = list(set(list(range(nqubits))) ^ set(qubits + controls))

    key = f"{ncontrols}"
    if key not in self._dict_cached_strings_two:
        self._dict_cached_strings_two[key] = self._get_cached_strings(
            nqubits, weight, ncontrols
        )

    strings = self._dict_cached_strings_two[key]

    matrix = gate.matrix(backend=self)
    matrix_0000, matrix_1111 = matrix[0, 0], matrix[3, 3]
    matrix_0101, matrix_0110 = matrix[1, 1], matrix[1, 2]
    matrix_1001, matrix_1010 = matrix[2, 1], matrix[2, 2]

    if weight - ncontrols > 0 and weight not in [0, nqubits]:
        indexes_in = self.np.zeros((len(strings), nqubits), dtype=str)
        indexes_in[:, other_qubits] = strings
        if len(controls) > 0:
            indexes_in[:, controls] = "1"
        indexes_in[:, qubits] = ["1", "0"]
        indexes_out = self.np.copy(indexes_in)
        indexes_out[:, qubits] = ["0", "1"]
        indexes_in = self.np.array(
            [self._dict_indexes["".join(elem)][0] for elem in indexes_in]
        )
        indexes_out = self.np.array(
            [self._dict_indexes["".join(elem)][0] for elem in indexes_out]
        )

        old_in, old_out = state[indexes_in], state[indexes_out]
        new_amplitudes_in = matrix_1010 * old_in + matrix_1001 * old_out
        new_amplitudes_out = matrix_0101 * old_out + matrix_0110 * old_in

        state[indexes_in] = new_amplitudes_in
        state[indexes_out] = new_amplitudes_out

    if (
        weight - ncontrols >= 0
        and nqubits - weight > 1
        and (matrix_0000.real != 1 or abs(matrix_0000.imag) > 0)
    ):
        # update the |...00...> amplitudes if necessary
        state = self._update_amplitudes(
            state,
            qubits,
            controls,
            other_qubits,
            weight,
            matrix_0000,
            ["0", "0"],
            shift=1,
        )

    if weight - ncontrols > 1 and (matrix_1111.real != 1 or abs(matrix_1111.imag) > 0):
        # update the |...11...> amplitudes if necessary
        state = self._update_amplitudes(
            state,
            qubits,
            controls,
            other_qubits,
            weight,
            matrix_1111,
            ["1", "1"],
            shift=-1,
        )

    return state


def _apply_gate_CCZ(self, gate, state, nqubits: int, weight: int):
    """Custom ``apply_gate`` method for the :class:`qibo.gates.CCZ` gate.

    Args:
        gate (:class:`qibo.gates.CCZ`): :math:`2`-controlled :math:`Z` gate
            to be applied to ``state``.
        state (ndarray): state to suffer the action of ``gate``.
        nqubits (int): total number of qubits in the circuit.
        weight (int): Hamming-weight of ``state``.

    Returns:
        ndarray: ``state`` after the action of the :class:`qibo.gates.CCZ` gate.
    """
    qubits = list(gate.qubits)
    gate_qubits = len(qubits)

    self._dict_indexes = self._get_lexicographical_order(nqubits, weight)

    strings = list(self._dict_indexes.keys())

    for j in range(len(strings)):
        gate_string = [strings[j][q] for q in qubits]
        if gate_string.count("1") == gate_qubits:
            state[j] *= -1

    return state


def _apply_gate_n_qubit(self, gate, state, nqubits, weight):
    """Custom ``apply_gate`` method for Hamming-weight-preserving n-qubit gates.

    This method performs matrix multiplication directly in the subspace with Hamming-weight
    ``weight`` without the need to calculate the full matrix representation of the gate.

    .. note::
        The attribute ``gate.hamming_weight`` must be manually set to ``True`` for this method to work.

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): Hamming-weight-preserving
            :math:`n`-qubit gate to be applied to ``state``
        state (ndarray): state to suffer the action of ``gate``.
        nqubits (int): total number of qubits in the circuit.
        weight (int): Hamming weight of ``state``.

    Returns:
        ndarray: ``state`` after the action of ``gate``.
    """
    gate_matrix = gate.matrix(backend=self)
    qubits = list(gate.target_qubits)
    gate_qubits = len(qubits)
    if 2 ** (gate_qubits) != gate_matrix.shape[0]:
        qubits = list(gate.qubits)
        gate_qubits = len(qubits)
        controls = []
        ncontrols = 0
    else:
        controls = list(gate.control_qubits)
        ncontrols = len(controls)

    other_qubits = list(set(qubits + controls) ^ set(range(nqubits)))
    map_ = qubits + controls + other_qubits
    gate_matrix = gate.matrix(backend=self)

    self._dict_indexes = self._get_lexicographical_order(nqubits, weight)

    strings = self.np.array(list(self._dict_indexes.keys()))
    indexes = self.np.array([index[1] for index in self._dict_indexes.values()])
    dim = len(indexes)

    matrix = self.np.zeros((dim, dim))
    matrix = self.cast(matrix, dtype=self.dtype)
    mod_condition = indexes[:, None] % 2 ** (
        nqubits - gate_qubits - ncontrols
    ) == indexes[None, :] % 2 ** (nqubits - gate_qubits - ncontrols)

    control_substrings = self.np.array(
        [s[gate_qubits : gate_qubits + ncontrols].count("1") for s in strings]
    )
    control_condition = control_substrings[:, None] == ncontrols

    row_indices = indexes[:, None] // 2 ** (nqubits - gate_qubits)
    col_indices = indexes[None, :] // 2 ** (nqubits - gate_qubits)

    matrix[mod_condition & control_condition] = gate_matrix[row_indices, col_indices][
        mod_condition & control_condition
    ]

    diagonal_indices = self.np.eye(dim, dtype=bool)
    matrix[mod_condition & ~control_condition & diagonal_indices] = 1

    new_matrix = self.np.zeros((dim, dim))
    new_matrix = self.cast(new_matrix, dtype=self.dtype)

    strings_array = self.np.array([list(s) for s in strings])
    reordered_strings_array = strings_array[:, map_]

    reordered_strings = self.np.array(["".join(s) for s in reordered_strings_array])
    reordered_indexes = [
        self.np.where(strings == new_string_i)[0][0]
        for new_string_i in reordered_strings
    ]
    new_matrix = matrix[reordered_indexes][:, reordered_indexes]

    new_matrix = self.cast(new_matrix)
    state = new_matrix @ state

    return state


def calculate_symbolic(
    self, state, nqubits, weight, decimals=5, cutoff=1e-10, max_terms=20
):
    """Dirac notation representation of the state in the computational basis.

    Args:
        state (ndarray): state to suffer the action of ``gate``.
        nqubits (int): total number of qubits in the circuit.
        weight (int): Hamming-weight of ``state``.
        decimals (int, optional): Number of decimals for the amplitudes.
            Defaults to :math:`5`.
        cutoff (float, optional): Amplitudes with absolute value smaller than the
            cutoff are ignored from the representation. Defaults to  :math:`1e-10`.
        max_terms (int, optional): Maximum number of terms to print. If the state
            contains more terms they will be ignored. Defaults to :math:`20`.

    Returns:
        str: String representing the state in the computational basis.
    """
    state = self.to_numpy(state)
    terms = []

    self._dict_indexes = self._get_lexicographical_order(nqubits, weight)

    strings = list(self._dict_indexes.keys())
    for i in self.np.nonzero(state)[0]:
        i = int(i)
        b = strings[i]
        if self.np.abs(state[i]) >= cutoff:
            x = self.np.round(state[i], decimals)
            terms.append(f"{x}|{b}>")
        if len(terms) >= max_terms:
            terms.append("...")
            return terms
    return terms


def calculate_probabilities(self, state, qubits, weight, nqubits):
    """Calculate the probabilities of the measured qubits from the statevector.

    Args:
        state (ndarray): state to suffer the action of ``gate``.
        qubits (list or set, optional): Set of qubits that are measured.
        weight (int): Hamming-weight of ``state``.
        nqubits (int): total number of qubits in the circuit.

    Returns:
        ndarray: Probabilities over the input qubits.
    """
    rtype = self.np.real(state).dtype

    self._dict_indexes = self._get_lexicographical_order(nqubits, weight)

    strings = list(self._dict_indexes.keys())
    indexes = [index[0] for index in self._dict_indexes.values()]

    measured_strings = {}
    for string, index in zip(strings, indexes):
        measured_string = "".join(string[q] for q in qubits)
        measured_strings[measured_string] = measured_strings.get(
            measured_string, 0
        ) + self.np.real(self.np.abs(state[index]) ** 2)

    strings = list(measured_strings.keys())
    indexes = [int(string, 2) for string in strings]
    probs = self.np.zeros(2 ** len(qubits), dtype=rtype)
    for index, string in zip(indexes, strings):
        probs[index] = measured_strings[string]

    probs = self.cast(probs, dtype=rtype)
    return probs


def collapse_state(self, state, qubits, shot, weight, nqubits, normalize=True):
    """Collapse state vector according to measurement shot.

    Args:
        state (ndarray): state to suffer the action of ``gate``.
        qubits (list or set, optional): Set of qubits that are measured.
        shot (list): Decimal value of the bitstring measured.
        weight (int): Hamming-weight of ``state``.
        nqubits (int): total number of qubits in the circuit.

    Returns:
        ndarray: collapsed ``state``.
    """
    state = self.cast(state, dtype=state.dtype)

    self._dict_indexes = self._get_lexicographical_order(nqubits, weight)

    strings = list(self._dict_indexes.keys())
    indexes = [index[0] for index in self._dict_indexes.values()]
    binshot = self.samples_to_binary(shot, len(qubits))[0]
    binshot = "".join(str(s) for s in binshot)
    for string, index in zip(strings, indexes):
        if "".join(string[q] for q in qubits) != binshot:
            state[index] = 0

    if normalize:
        norm = self.np.sqrt(self.np.sum(self.np.abs(state) ** 2))
        state = state / norm

    return state
