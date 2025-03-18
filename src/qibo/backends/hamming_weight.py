"""Module defining the Hamming-weight-preserving backend."""

import numpy as np
from scipy.special import binom

from qibo import gates
from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error


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
        if engine == "numpy":
            self.engine = construct_backend(self.platform)
        elif engine == "numba":
            self.engine = construct_backend("qibojit", platform=self.platform)
        elif engine == "cupy":  # pragma: no cover
            self.engine = construct_backend("qibojit", platform=self.platform)
        elif engine == "cuquantum":  # pragma: no cover
            self.engine = construct_backend("qibojit", platform=self.platform)
        elif engine == "tensorflow":  # pragma: no cover
            self.engine = construct_backend("qiboml", platform=self.platform)
        elif engine == "torch":  # pragma: no cover
            self.engine = construct_backend("qiboml", platform=self.platform)
        else:  # pragma: no cover
            raise_error(
                NotImplementedError,
                f"Backend `{engine}` is not supported for Hamming weight preserving circuit simulation.",
            )
        self.np = self.engine.np

        # cached order of operations for single- and two-qubit gates
        self._dict_cached_strings_one = {}
        self._dict_cached_strings_two = {}

        # map to translate the order of bitstrings from the Gray code
        # to the lexicographical (i.e. ascending) order
        self._dict_indexes = None

        self.name = "hamming_weight"

    def cast(self, x, dtype=None, copy: bool = False):
        """Cast an object as the array type of the current backend.

        Args:
            x: Object to cast to array.
            copy (bool, optional): If ``True`` a copy of the object is created in memory.
                Defaults to ``False``.
        """
        return self.engine.cast(x, dtype=dtype, copy=copy)

    def apply_gate(self, gate, state, nqubits, weight):
        if isinstance(gate, gates.M):
            return gate.apply_hamming_weight(self, state, weight, nqubits)

        if isinstance(gate, gates.CCZ):
            return self._apply_gate_CCZ(gate, state, nqubits, weight)

        if len(gate.target_qubits) == 1:
            return self._apply_gate_single_qubit(gate, state, nqubits, weight)

        if len(gate.target_qubits) == 2:
            return self._apply_gate_two_qubit(gate, state, nqubits, weight)

        return self._apply_gate_n_qubit(gate, state, nqubits, weight)

    def execute_circuit(self, circuit, weight: int, initial_state=None, nshots=None):

        from qibo.quantum_info.hamming_weight import (  # pylint: disable=C0415
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
                initial_state = self.engine.cast(self.np.zeros(n_choose_k))
                initial_state[0] = 1

            state = initial_state
            for gate in circuit.queue:
                state = self.apply_gate(gate, state, nqubits, weight)

            result = HammingWeightResult(
                state,
                weight,
                nqubits,
                measurements=circuit.measurements,
                nshots=nshots,
                backend=self,
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
        from qibo.models.encodings import _ehrlich_algorithm  # pylint: disable=C0415

        strings = _ehrlich_algorithm(initial_string, return_indices=False)
        strings = [list(string) for string in strings]
        strings = self.np.asarray(strings, dtype=int)

        return strings

    def _get_cached_strings(
        self, nqubits: int, weight: int, ncontrols: int = 0, two_qubit_gate: bool = True
    ):
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
        n_choose_k = int(binom(nqubits, weight))
        indexes = list(range(n_choose_k))

        lexicographical_order = self._get_cached_strings(nqubits + 2, weight + 1)
        lexicographical_order = [
            "".join(item.astype(str)) for item in lexicographical_order
        ]
        lexicographical_order.sort()
        lexicographical_order_int = [
            int(item, base=2) for item in lexicographical_order
        ]
        _dict_indexes = dict(
            zip(lexicographical_order, zip(indexes, lexicographical_order_int))
        )

        return _dict_indexes

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

    def _apply_gate_two_qubit(self, gate, state, nqubits, weight):
        # Right now, it works only with two-qubit Givens rotations,
        # e.g. gates.RBS, gates.GIVENS, gates.SWAP, gates.iSWAP,
        # gates.SiSWAP, and gates.RZZ (up to global phase).
        # state = self.cast(state_1,copy=True)
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
        matrix_0000 = matrix[0, 0]
        matrix_1111 = matrix[3, 3]
        matrix_0101 = matrix[1, 1]
        matrix_0110 = matrix[1, 2]
        matrix_1001 = matrix[2, 1]
        matrix_1010 = matrix[2, 2]

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

        if weight - ncontrols >= 0:
            # update the |...00...> amplitudes if necessary
            if matrix_0000.real != 1 or abs(matrix_0000.imag) > 0:
                indexes_in = self.np.zeros((len(strings), nqubits), dtype=str)
                indexes_in[:, other_qubits] = strings
                if len(controls) > 0:
                    indexes_in[:, controls] = "1"
                indexes_in[:, qubits] = ["0", "0"]
                indexes_in = self.np.array(
                    [self._dict_indexes["".join(elem)][0] for elem in indexes_in]
                )
                state[indexes_in] *= matrix_0000

        if weight - ncontrols > 1:
            strings = self._get_cached_strings(nqubits, weight - 1, ncontrols)
            # update the |...11...> amplitudes if necessary
            if matrix_1111.real != 1 or abs(matrix_1111.imag) > 0:
                indexes_in = self.np.zeros((len(strings), nqubits), dtype=str)
                indexes_in[:, other_qubits] = strings
                if len(controls) > 0:
                    indexes_in[:, controls] = "1"
                indexes_in[:, qubits] = ["1", "1"]
                indexes_in = self.np.array(
                    [self._dict_indexes["".join(elem)][0] for elem in indexes_in]
                )
                state[indexes_in] *= matrix_1111

        return state

    def _apply_gate_CCZ(self, gate, state, nqubits, weight):
        qubits = list(gate.qubits)
        gate_qubits = len(qubits)

        if (
            self._dict_indexes is None
            or list(self._dict_indexes.keys())[0].count("1") != weight
        ):
            self._dict_indexes = self._get_lexicographical_order(nqubits, weight)

        strings = list(self._dict_indexes.keys())

        for i in range(len(strings)):
            gate_string = [strings[i][q] for q in qubits]
            if gate_string.count("1") == gate_qubits:
                state[i] *= -1

        return state

    def _apply_gate_n_qubit(self, gate, state, nqubits, weight):
        gate_matrix = gate.matrix(backend=self.engine)
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

        other_qubits = list(set(qubits + controls) ^ set(list(range(nqubits))))
        map = qubits + controls + other_qubits
        gate_matrix = gate.matrix(backend=self.engine)

        if (
            self._dict_indexes is None
            or list(self._dict_indexes.keys())[0].count("1") != weight
        ):
            self._dict_indexes = self._get_lexicographical_order(nqubits, weight)

        strings = list(self._dict_indexes.keys())
        indexes = [index[1] for index in self._dict_indexes.values()]
        dim = len(indexes)
        matrix = self.np.zeros((dim, dim), dtype=complex)
        for i, index_i in enumerate(indexes):
            for j, index_j in enumerate(indexes):
                if index_i % 2 ** (
                    nqubits - gate_qubits - ncontrols
                ) == index_j % 2 ** (nqubits - gate_qubits - ncontrols):
                    if (
                        strings[j][gate_qubits : gate_qubits + ncontrols].count("1")
                        == ncontrols
                    ):
                        matrix[i, j] = gate_matrix[
                            index_i // 2 ** (nqubits - gate_qubits),
                            index_j // 2 ** (nqubits - gate_qubits),
                        ]
                    else:
                        matrix[i, j] = 1 if i == j else 0

        new_matrix = self.np.zeros((dim, dim), dtype=complex)
        for i, string_i in enumerate(strings):
            new_string_i = [""] * len(string_i)
            for k in range(nqubits):
                new_string_i[map[k]] = string_i[k]
            new_string_i = "".join(new_string_i)
            new_index_i = strings.index(new_string_i)

            for j, string_j in enumerate(strings):
                new_string_j = [""] * len(string_j)
                for k in range(nqubits):
                    new_string_j[map[k]] = string_j[k]
                new_string_j = "".join(new_string_j)
                new_index_j = strings.index(new_string_j)

                new_matrix[new_index_i, new_index_j] = matrix[i, j]

        new_matrix = self.cast(new_matrix)
        state = self.np.matmul(new_matrix, state)
        return state

    def calculate_symbolic(
        self, state, nqubits, weight, decimals=5, cutoff=1e-10, max_terms=20
    ):
        state = self.to_numpy(state)
        terms = []

        if (
            self._dict_indexes is None
            or list(self._dict_indexes.keys())[0].count("1") != weight
        ):
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
        rtype = self.np.real(state).dtype

        if (
            self._dict_indexes is None
            or list(self._dict_indexes.keys())[0].count("1") != weight
        ):
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
        state = self.cast(state)
        if (
            self._dict_indexes is None
            or list(self._dict_indexes.keys())[0].count("1") != weight
        ):
            self._dict_indexes = self._get_lexicographical_order(nqubits, weight)

        strings = list(self._dict_indexes.keys())
        indexes = [index[0] for index in self._dict_indexes.values()]
        binshot = self.samples_to_binary(shot, len(qubits))[0]
        for string, index in zip(strings, indexes):
            if "".join(str(s) for s in binshot) != "".join(string[q] for q in qubits):
                state[index] = 0

        if normalize:
            norm = self.np.sqrt(self.np.sum(self.np.abs(state) ** 2))
            state = state / norm
        state = self.cast(state)
        return state
