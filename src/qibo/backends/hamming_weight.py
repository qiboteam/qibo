import numpy as np
from scipy.special import binom

from qibo.backends.numpy import NumpyBackend


class HammingWeightBackend(NumpyBackend):
    def __init__(self):
        super().__init__()
        self._dict_cached_strings = {}

    def _get_cached_strings(self, nqubits: int, weight: int, ncontrols: int = 0):
        initial_string = np.array(
            [1] * (weight - 1 - ncontrols)
            + [0] * ((nqubits - 2 - ncontrols) - (weight - 1 - ncontrols))
        )
        strings =_ ehrlich_algorithm(initial_string, False)
        strings = [list(string) for string in strings]
        strings = np.asarray(strings, dtype=int)

        return strings

    def apply_gate(self, gate, state, nqubits, weight):
        # Right now, it works only with two-qubit Givens rotations,
        # e.g. gates.RBS, gates.GIVENS, gates.SWAP, gates.iSWAP,
        # gates.SiSWAP, and gates.RZZ (up to global phase).
        qubits = list(gate.target_qubits)
        controls = list(gate.control_qubits)
        ncontrols = len(controls)
        other_qubits = list(set(list(range(nqubits))) ^ set(qubits + controls))

        key = f"{ncontrols}"
        if key not in self._dict_cached_strings:
            self._dict_cached_strings[key] = self._get_cached_strings(
                nqubits, weight, ncontrols
            )

        strings = self._dict_cached_strings[key]

        matrix = gate.matrix().real
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

    def execute_circuit(self, circuit, weight: int, initial_state=None, nshots=1000):
        # Right now, it works only for ``weight'>=3`` and for gates with
        # number of controls < weight.
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
            initial_state = self.np.zeros(n_choose_k, dtype=float)
            initial_state[0] = 1

        state = initial_state
        for gate in circuit.queue:
            state = self.apply_gate(gate, state, nqubits, weight)

        return state
