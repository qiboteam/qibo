"""Module defining the Hamming-weight-preserving backend."""

import numpy as np
from scipy.special import binom

from qibo.backends.numpy import NumpyBackend


class HammingWeightBackend(NumpyBackend):
    def __init__(self, engine=None):
        super().__init__()

        from qibo.backends import construct_backend  # pylint: disable=C415

        if engine is None:
            from qibo.backends import (  # pylint: disable=C0415
                _check_backend,
                _get_engine_name,
            )

            engine = _get_engine_name(_check_backend(engine))

        self.platform = engine
        self.engine = construct_backend(self.platform)
        self.np = self.engine.np

        # cached order of operations for single- and two-qubit gates
        self._dict_cached_strings_one = {}
        self._dict_cached_strings_two = {}

        # map to translate the order of bitstrings from the Gray code
        # to the lexicographical (i.e. ascending) order
        self._lexicographical_order = None

        self.name = "hamming_weight"

    def apply_gate(self, gate, state, nqubits, weight):
        if len(gate.target_qubits) == 1:
            return self._apply_gate_single_qubit(gate, state, nqubits, weight)

        return self._apply_gate_two_qubit(gate, state, nqubits, weight)

    def execute_circuit(self, circuit, weight: int, initial_state=None, nshots=1000):
        nqubits = circuit.nqubits
        n_choose_k = int(binom(nqubits, weight))
        indexes = list(range(n_choose_k))

        lexicographical_order = self._get_cached_strings(nqubits + 2, weight + 1)
        lexicographical_order = [
            "".join(item.astype(str)) for item in lexicographical_order
        ]
        lexicographical_order.sort()
        self._dict_indexes = dict(zip(lexicographical_order, indexes))
        del lexicographical_order, indexes

        if initial_state is None:
            initial_state = self.engine.cast(self.np.zeros(n_choose_k))
            initial_state[0] = 1

        state = initial_state
        for gate in circuit.queue:
            state = self.apply_gate(gate, state, nqubits, weight)

        return state

    def _gray_code(self, initial_string):
        from qibo.models.encodings import _ehrlich_algorithm  # pylint: disable=C0415

        strings = _ehrlich_algorithm(initial_string, False)
        strings = [list(string) for string in strings]
        strings = np.asarray(strings, dtype=int)

        return strings

    def _get_cached_strings(
        self, nqubits: int, weight: int, ncontrols: int = 0, two_qubit_gate: bool = True
    ):
        if two_qubit_gate:
            initial_string = np.array(
                [1] * (weight - 1 - ncontrols)
                + [0] * ((nqubits - 2 - ncontrols) - (weight - 1 - ncontrols))
            )
            strings = self._gray_code(initial_string)
        else:
            initial_string = np.array(
                [1] * (weight - ncontrols)
                + [0] * ((nqubits - 1 - ncontrols) - (weight - ncontrols))
            )
            strings_0 = self._gray_code(initial_string)

            initial_string = np.array(
                [1] * (weight - 1 - ncontrols)
                + [0] * ((nqubits - 1 - ncontrols) - max(0, (weight - 1 - ncontrols)))
            )
            strings_1 = self._gray_code(initial_string)

            strings = [strings_0, strings_1]

        return strings

    def _get_single_qubit_matrix(self, gate):
        matrix = gate.matrix(backend=self.engine)

        if gate.name in ["cz", "crz", "cu1"]:
            matrix = matrix[2:, 2:]

        return matrix

    def _apply_gate_single_qubit(self, gate, state, nqubits, weight):
        qubits = list(gate.target_qubits)
        controls = list(gate.control_qubits)
        ncontrols = len(controls)
        other_qubits = list(set(list(range(nqubits))) ^ set(qubits + controls))

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

        matrix = self._get_single_qubit_matrix(gate)
        matrix_00, matrix_11 = self.np.diag(matrix)

        indexes_one = np.zeros((len(strings_1), nqubits), dtype=str)

        if matrix_00 != 1.0:
            indexes_zero = np.zeros((len(strings_0), nqubits), dtype=str)
            indexes_zero[:, other_qubits] = strings_0
            indexes_zero[:, qubits] = ["0"]
            if len(controls) > 0:
                indexes_zero[:, controls] = "1"
            indexes_zero = np.array(
                [self._dict_indexes["".join(elem)] for elem in indexes_zero]
            )

            state[indexes_zero] *= matrix_00

        if matrix_11 != 1.0 and weight - ncontrols > 0:
            indexes_one[:, other_qubits] = strings_1
            if len(controls) > 0:
                indexes_one[:, controls] = "1"

            indexes_one[:, qubits] = ["1"]
            indexes_one = np.array(
                [self._dict_indexes["".join(elem)] for elem in indexes_one]
            )

            state[indexes_one] *= matrix_11

        return state

    def _apply_gate_two_qubit(self, gate, state, nqubits, weight):
        # Right now, it works only with two-qubit Givens rotations,
        # e.g. gates.RBS, gates.GIVENS, gates.SWAP, gates.iSWAP,
        # gates.SiSWAP, and gates.RZZ (up to global phase).
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

        matrix = gate.matrix(backend=self.engine)
        matrix_0101 = matrix[1, 1]
        matrix_0110 = matrix[1, 2]
        matrix_1001 = matrix[2, 1]
        matrix_1010 = matrix[2, 2]

        indexes_in = np.zeros((len(strings), nqubits), dtype=str)
        indexes_in[:, other_qubits] = strings
        if len(controls) > 0:
            indexes_in[:, controls] = "1"
        indexes_in[:, qubits] = ["1", "0"]
        indexes_out = np.copy(indexes_in)
        indexes_out[:, qubits] = ["0", "1"]

        indexes_in = np.array(
            [self._dict_indexes["".join(elem)] for elem in indexes_in]
        )
        indexes_out = np.array(
            [self._dict_indexes["".join(elem)] for elem in indexes_out]
        )

        old_in, old_out = state[indexes_in], state[indexes_out]

        new_amplitudes_in = matrix_1010 * old_in + matrix_1001 * old_out
        new_amplitudes_out = matrix_0101 * old_out + matrix_0110 * old_in

        state[indexes_in] = new_amplitudes_in
        state[indexes_out] = new_amplitudes_out

        return state
