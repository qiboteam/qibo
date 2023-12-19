"""Module definig the Clifford object, which allows phase-space representation of Clifford circuits and stabilizer states."""

from dataclasses import dataclass
from functools import reduce
from itertools import product
from typing import Optional, Union

import numpy as np

from qibo import Circuit, gates
from qibo.backends import Backend, CliffordBackend
from qibo.config import raise_error
from qibo.gates import M
from qibo.measurements import frequencies_to_binary


@dataclass
class Clifford:
    """Object storing the results of a circuit execution with the :class:`qibo.backends.clifford.CliffordBackend`.

    Args:
        data (ndarray or :class:`qibo.models.circuit.Circuit`): If ``ndarray``, it is the
            symplectic matrix of the state in phase-space representation.
            If :class:`qibo.models.circuit.Circuit`, it is a circuit composed only of Clifford
            gates and measurements.
        nqubits (int, optional): number of qubits of the state.
        measurements (list, optional): list of measurements gates :class:`qibo.gates.M`.
            Defaults to ``None``.
        nshots (int, optional): number of shots used for sampling the measurements.
            Defaults to :math:`1000`.
        engine (:class:`qibo.backends.abstract.Backend`, optional): engine to use in the execution
            of the :class:`qibo.backends.CliffordBackend`.
            It accepts all ``qibo`` backends besides the :class:`qibo.backends.TensorflowBackend`,
            which is not supported. If ``None``, defaults to :class:`qibo.backends.NumpyBackend`
            Defaults to ``None``.
    """

    data: Union[np.ndarray, Circuit]
    nqubits: Optional[int] = None
    measurements: Optional[list] = None
    nshots: int = 1000
    engine: Optional[Backend] = None

    _backend: Optional[CliffordBackend] = None
    _measurement_gate = None
    _samples: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.data, Circuit):
            clifford = self.from_circuit(self.data, engine=self.engine)
            self.symplectic_matrix = clifford.symplectic_matrix
            self.nqubits = clifford.nqubits
            self.measurements = clifford.measurements
            self.engine = clifford.engine
            self._samples = clifford._samples
            self._measurement_gate = clifford._measurement_gate
            self._backend = clifford._backend
        else:
            # adding the scratch row if not provided
            self.symplectic_matrix = self.data
            if self.symplectic_matrix.shape[0] % 2 == 0:
                self.symplectic_matrix = np.vstack(
                    (self.symplectic_matrix, np.zeros(self.symplectic_matrix.shape[1]))
                )
            self.nqubits = int((self.symplectic_matrix.shape[1] - 1) / 2)
            self._backend = CliffordBackend(self.engine)

    @classmethod
    def from_circuit(
        cls,
        circuit: Circuit,
        initial_state: Optional[np.ndarray] = None,
        nshots: int = 1000,
        engine: Optional[Backend] = None,
    ):
        """Allows to create a :class:`qibo.quantum_info.clifford.Clifford` object by executing the input circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Clifford circuit to run.
            initial_state (ndarray, optional): symplectic matrix of the initial state.
                If ``None``, defaults to the symplectic matrix of the zero state.
                Defaults to ``None``.
            nshots (int, optional): number of measurement shots to perform
                if ``circuit`` has measurement gates. Defaults to :math:`10^{3}`.
            engine (:class:`qibo.backends.abstract.Backend`, optional): engine to use in the
                execution of the :class:`qibo.backends.CliffordBackend`.
                It accepts all ``qibo`` backends besides the
                :class:`qibo.backends.TensorflowBackend`, which is not supported.
                If ``None``, defaults to :class:`qibo.backends.NumpyBackend`.
                Defaults to ``None``.

        Returns:
            (:class:`qibo.quantum_info.clifford.Clifford`): Object storing the result of the circuit execution.
        """
        cls._backend = CliffordBackend(engine)

        return cls._backend.execute_circuit(circuit, initial_state, nshots)

    def to_circuit(self, algorithm: Optional[str] = None):
        if not isinstance(algorithm, (str, type(None))):
            raise_error(
                TypeError,
                f"``algorithm`` must be type int, but it is type {type(algorithm)}",
            )

        if algorithm is not None and algorithm not in ["AG04", "BM20"]:
            raise_error(ValueError, f"``algorithm`` {algorithm} not found.")

        if self.nqubits == 1:
            return _single_qubit_clifford_decomposition(self.symplectic_matrix)

        if algorithm is None:
            algorithm = "AG04"

        if algorithm == "BM20":
            return _decomposition_BM20(self)

        return _decomposition_AG04(self)

    def generators(self, return_array: bool = False):
        """Extracts the generators of stabilizers and destabilizers.

        Args:
            return_array (bool, optional): If ``True`` returns the generators as ``ndarray``.
                If ``False``, their representation as strings is returned. Defaults to ``False``.

        Returns:
            (list, list): Generators and their corresponding phases, respectively.
        """
        return self._backend.symplectic_matrix_to_generators(
            self.symplectic_matrix, return_array
        )

    def stabilizers(self, symplectic: bool = False, return_array: bool = False):
        """Extracts the stabilizers of the state.

        Args:
            return_array (bool, optional): If ``True`` returns the stabilizers as ``ndarray``.
                If ``False``, returns stabilizers as strings. Defaults to ``False``.

        Returns:
            (list): Stabilizers of the state.
        """
        if not symplectic:
            generators, phases = self.generators(return_array)

            return self._construct_operators(
                generators[self.nqubits :],
                phases[self.nqubits :],
            )

        return self.symplectic_matrix[self.nqubits : -1, :]

    def destabilizers(self, symplectic: bool = False, return_array: bool = False):
        """Extracts the destabilizers of the state.

        Args:
            return_array (bool, optional): If ``True`` returns the destabilizers as ``ndarray``.
                If ``False``, their representation as strings is returned. Defaults to ``False``.

        Returns:
            (list): Destabilizers of the state.
        """
        if not symplectic:
            generators, phases = self.generators(return_array)

            return self._construct_operators(
                generators[: self.nqubits], phases[: self.nqubits]
            )

        return self.symplectic_matrix[: self.nqubits, :]

    def state(self):
        """Builds the density matrix representation of the state.

        .. note::
            This method is inefficient in runtime and memory for a large number of qubits.

        Returns:
            (ndarray): Density matrix of the state.
        """
        stabilizers = self.stabilizers(return_array=True)

        return self.engine.np.sum(stabilizers, axis=0) / len(stabilizers)

    @property
    def measurement_gate(self):
        """Single measurement gate containing all measured qubits.

        Useful for sampling all measured qubits at once when simulating.
        """
        if self._measurement_gate is None:
            for gate in self.measurements:
                if self._measurement_gate is None:
                    self._measurement_gate = M(*gate.init_args, **gate.init_kwargs)
                else:
                    self._measurement_gate.add(gate)

        return self._measurement_gate

    def samples(self, binary: bool = True, registers: bool = False):
        """Returns raw measurement samples.

        Args:
            binary (bool, optional): If ``False``, return samples in binary form.
                If ``True``, returns samples in decimal form. Defalts to ``True``.
            registers (bool, optional): If ``True``, groups samples according to registers.
                Defaults to ``False``.

        Returns:
            If ``binary`` is ``True``
                samples are returned in binary form as a tensor
                of shape ``(nshots, n_measured_qubits)``.
            If ``binary`` is ``False``
                samples are returned in decimal form as a tensor
                of shape ``(nshots,)``.
            If ``registers`` is ``True``
                samples are returned in a ``dict`` where the keys are the register
                names and the values are the samples tensors for each register.
            If ``registers`` is ``False``
                a single tensor is returned which contains samples from all the
                measured qubits, independently of their registers.
        """
        if not self.measurements:
            raise_error(RuntimeError, "No measurement provided.")

        measured_qubits = self.measurement_gate.qubits

        if self._samples is None:
            if self.measurements[0].result.has_samples():
                samples = np.concatenate(
                    [gate.result.samples() for gate in self.measurements], axis=1
                )
            else:
                samples = self._backend.sample_shots(
                    self.symplectic_matrix, measured_qubits, self.nqubits, self.nshots
                )
            if self.measurement_gate.has_bitflip_noise():
                p0, p1 = self.measurement_gate.bitflip_map
                bitflip_probabilities = self._backend.cast(
                    [
                        [p0.get(q) for q in measured_qubits],
                        [p1.get(q) for q in measured_qubits],
                    ]
                )
                samples = self._backend.cast(samples, dtype="int32")
                samples = self._backend.apply_bitflips(samples, bitflip_probabilities)
            # register samples to individual gate ``MeasurementResult``
            qubit_map = {
                q: i for i, q in enumerate(self.measurement_gate.target_qubits)
            }
            self._samples = self._backend.cast(samples, dtype="int32")
            for gate in self.measurements:
                rqubits = tuple(qubit_map.get(q) for q in gate.target_qubits)
                gate.result.register_samples(self._samples[:, rqubits], self._backend)

        if registers:
            return {
                gate.register_name: gate.result.samples(binary)
                for gate in self.measurements
            }

        if binary:
            return self._samples

        return self._backend.samples_to_decimal(self._samples, len(measured_qubits))

    def frequencies(self, binary: bool = True, registers: bool = False):
        """Returns the frequencies of measured samples.

        Args:
            binary (bool, optional): If ``True``, return frequency keys in binary form.
                If ``False``, return frequency keys in decimal form. Defaults to ``True``.
            registers (bool, optional): If ``True``, groups frequencies according to registers.
                Defaults to ``False``.

        Returns:
            A `collections.Counter` where the keys are the observed values
            and the values the corresponding frequencies, that is the number
            of times each measured value/bitstring appears.

            If ``binary`` is ``True``
                the keys of the `Counter` are in binary form, as strings of
                :math:`0`s and :math`1`s.
            If ``binary`` is ``False``
                the keys of the ``Counter`` are integers.
            If ``registers`` is ``True``
                a `dict` of `Counter` s is returned where keys are the name of
                each register.
            If ``registers`` is ``False``
                a single ``Counter`` is returned which contains samples from all
                the measured qubits, independently of their registers.
        """
        measured_qubits = self.measurement_gate.target_qubits
        freq = self._backend.calculate_frequencies(self.samples(False))
        if registers:
            if binary:
                return {
                    gate.register_name: frequencies_to_binary(
                        self._backend.calculate_frequencies(gate.result.samples(False)),
                        len(gate.target_qubits),
                    )
                    for gate in self.measurements
                }

            return {
                gate.register_name: self._backend.calculate_frequencies(
                    gate.result.samples(False)
                )
                for gate in self.measurements
            }

        if binary:
            return frequencies_to_binary(freq, len(measured_qubits))

        return freq

    def probabilities(self, qubits: Optional[Union[tuple, list]] = None):
        """Computes the probabilities of the selected qubits from the measured samples.

        Args:
            qubits (tuple or list, optional): Qubits for which to compute the probabilities.

        Returns:
            (ndarray): Measured probabilities.
        """
        if isinstance(qubits, list):
            qubits = tuple(qubits)

        measured_qubits = self.measurement_gate.qubits
        if qubits is not None:
            if not set(qubits).issubset(set(measured_qubits)):
                raise_error(
                    RuntimeError,
                    f"Asking probabilities for qubits {qubits}, but only qubits {measured_qubits} were measured.",
                )
            qubits = [measured_qubits.index(q) for q in qubits]
        else:
            qubits = range(len(measured_qubits))

        probs = [0 for _ in range(2 ** len(measured_qubits))]
        samples = self.samples(binary=False)
        for s in samples:
            probs[int(s)] += 1

        probs = self.engine.cast(probs) / len(samples)

        return self._backend.calculate_probabilities(
            self.engine.np.sqrt(probs), qubits, len(measured_qubits)
        )

    def copy(self, deep: bool = False):
        """Returns copy of :class:`qibo.quantum_info.clifford.Clifford` object.

        Args:
            deep (bool, optional): If ``True``, creates another copy in memory.
                Defaults to ``False``.

        Returns:
            :class:`qibo.quantum_info.clifford.Clifford`: copy of original ``Clifford`` object.
        """
        if not isinstance(deep, bool):
            raise_error(
                TypeError, f"``deep`` must be type bool, but it is type {type(deep)}."
            )

        symplectic_matrix = (
            self.engine.np.copy(self.symplectic_matrix)
            if deep
            else self.symplectic_matrix
        )

        return self.__class__(
            symplectic_matrix, self.nqubits, self.measurements, self.nshots, self.engine
        )

    def _construct_operators(self, generators: list, phases: list):
        """Helper function to construct all the operators from their generators.

        Args:
            generators (list or ndarray): generators.
            phases (list or ndarray): phases of the generators.

        Returns:
            (list): All operators generated by the generators of the stabilizer group.
        """

        if not isinstance(generators[0], str):
            generators = self._backend.cast(generators)
            phases = self._backend.cast(phases)

            operators = generators * phases.reshape(-1, 1, 1)
            identity = self.engine.identity_density_matrix(
                self.nqubits, normalize=False
            )
            operators = self._backend.cast([(g, identity) for g in operators])

            return self._backend.cast(
                [reduce(self.engine.np.matmul, ops) for ops in product(*operators)]
            )

        operators = list(np.copy(generators))
        for i in (phases == -1).nonzero()[0]:
            i = int(i)
            operators[i] = f"-{operators[i]}"

        identity = "".join(["I" for _ in range(self.nqubits)])

        operators = [(g, identity) for g in operators]

        return [_string_product(ops) for ops in product(*operators)]


def _one_qubit_paulis_string_product(pauli_1: str, pauli_2: str):
    """Calculate the product of two single-qubit Paulis represented as strings.

    Args:
        pauli_1 (str): First Pauli operator.
        pauli_2 (str): Second Pauli operator.

    Returns:
        (str): Product of the two Pauli operators.
    """
    products = {
        "XY": "iZ",
        "YZ": "iX",
        "ZX": "iY",
        "YX": "-iZ",
        "ZY": "-iX",
        "XZ": "iY",
        "XX": "I",
        "ZZ": "I",
        "YY": "I",
        "XI": "X",
        "IX": "X",
        "YI": "Y",
        "IY": "Y",
        "ZI": "Z",
        "IZ": "Z",
    }
    prod = products[
        "".join([p.replace("i", "").replace("-", "") for p in (pauli_1, pauli_2)])
    ]
    # calculate the phase
    sign = len([True for p in (pauli_1, pauli_2, prod) if "-" in p])
    n_i = len([True for p in (pauli_1, pauli_2, prod) if "i" in p])
    sign = "-" if sign % 2 == 1 else ""
    if n_i == 0:
        i = ""
    elif n_i == 1:
        i = "i"
    elif n_i == 2:
        i = ""
        sign = "-" if sign == "" else ""
    elif n_i == 3:
        i = "i"
        sign = "-" if sign == "" else ""
    return "".join([sign, i, prod.replace("i", "").replace("-", "")])


def _string_product(operators: list):
    """Calculates the tensor product of a list of operators represented as strings.

    Args:
        operators (list): list of operators.

    Returns:
        (str): String representing the tensor product of the operators.
    """
    # calculate global sign
    phases = len([True for op in operators if "-" in op])
    i = len([True for op in operators if "i" in op])
    # remove the - signs and the i
    operators = "|".join(operators).replace("-", "").replace("i", "").split("|")

    prod = []
    for op in zip(*operators):
        op = [o for o in op if o != "I"]
        if len(op) == 0:
            tmp = "I"
        elif len(op) > 1:
            tmp = reduce(_one_qubit_paulis_string_product, op)
        else:
            tmp = op[0]
        # append signs coming from products
        if tmp[0] == "-":
            phases += 1
        # count i coming from products
        if "i" in tmp:
            i += 1
        prod.append(tmp.replace("i", "").replace("-", ""))
    result = "".join(prod)

    # product of the i-s
    if i % 4 == 1 or i % 4 == 3:
        result = f"i{result}"
    if i % 4 == 2 or i % 4 == 3:
        phases += 1

    phases = "-" if phases % 2 == 1 else ""

    return f"{phases}{result}"


def _decomposition_AG04(clifford):
    """Returns a Clifford object decomposed into a circuit based on Aaronson-Gottesman method.

    Args:
        clifford (:class:`qibo.quantum_info.clifford.Clifford`): Clifford object.

    Return:
        :class:`qibo.models.circuit.Circuit`: Clifford circuit.

    Reference:
        1. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
           Phys. Rev. A 70, 052328 (2004).
           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_
    """
    nqubits = clifford.nqubits

    circuit = Circuit(nqubits)
    clifford_copy = clifford.copy(deep=True)

    for k in range(nqubits):
        # put a 1 one into position by permuting and using Hadamards(i,i)
        _set_qubit_x_true(clifford_copy, circuit, k)
        # make all entries in row i except ith equal to 0
        # by using phase gate and CNOTS
        _set_row_x_zero(clifford_copy, circuit, k)
        # treat Zs
        _set_row_z_zero(clifford_copy, circuit, k)

    for k in range(nqubits):
        if clifford_copy.symplectic_matrix[:nqubits, -1][k]:
            clifford_copy.symplectic_matrix = clifford._backend.clifford_operations.Z(
                clifford_copy.symplectic_matrix, k, nqubits
            )
            circuit.add(gates.Z(k))
        if clifford_copy.symplectic_matrix[nqubits:-1, -1][k]:
            clifford_copy.symplectic_matrix = clifford._backend.clifford_operations.X(
                clifford_copy.symplectic_matrix, k, nqubits
            )
            circuit.add(gates.X(k))

    return circuit.invert()


def _decomposition_BM20(clifford):
    """Optimal CNOT-cost decomposition of a Clifford operator on :math:`n \\in \\{2, 3 \\}` into a circuit based on Bravyi-Maslov method.

    Args:
        clifford (:class:`qibo.quantum_info.clifford.Clifford`): Clifford object.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Clifford circuit.

    References:
        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the structure of the Clifford group*,
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_.
    """
    nqubits = clifford.nqubits
    clifford_copy = clifford.copy(deep=True)

    if nqubits > 3:
        raise_error(
            ValueError, "This methos can only be implemented for ``nqubits <= 3``."
        )

    if nqubits == 1:
        return _single_qubit_clifford_decomposition(clifford_copy.symplectic_matrix)

    inverse_circuit = Circuit(nqubits)

    cnot_cost = _cnot_cost(clifford_copy)

    # Find composition of circuits with CX and (H.S)^a gates to reduce CNOT count
    while cnot_cost > 0:
        clifford_copy, inverse_circuit, cnot_cost = _reduce_cost(
            clifford_copy, inverse_circuit, cnot_cost
        )

    # Decompose the remaining product of 1-qubit cliffords
    last_row = clifford_copy.engine.cast([False] * 3, dtype=bool)
    circuit = Circuit(nqubits)
    for qubit in range(nqubits):
        position = [qubit, qubit + nqubits]
        single_qubit_circuit = _single_qubit_clifford_decomposition(
            clifford_copy.engine.np.append(
                clifford_copy.symplectic_matrix[position][:, position + [-1]], last_row
            ).reshape(3, 3)
        )
        if len(single_qubit_circuit.queue) > 0:
            for gate in single_qubit_circuit.queue:
                gate.init_args = [qubit]
                gate.target_qubits = (qubit,)
                circuit.queue.extend([gate])

    # Add the inverse of the 2-qubit reductions circuit
    if len(inverse_circuit.queue) > 0:
        circuit.queue.extend(inverse_circuit.invert().queue)

    return circuit


def _single_qubit_clifford_decomposition(symplectic_matrix):
    """Decompose symplectic matrix of a single-qubit Clifford into a Clifford circuit.

    Args:
        symplectic_matrix (ndarray): Symplectic matrix to be decomposed.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Clifford circuit.
    """
    circuit = Circuit(nqubits=1)

    # Add phase correction
    destabilizer_phase, stabilizer_phase = symplectic_matrix[:-1, -1]
    if destabilizer_phase and not stabilizer_phase:
        circuit.add(gates.Z(0))
    elif not destabilizer_phase and stabilizer_phase:
        circuit.add(gates.X(0))
    elif destabilizer_phase and stabilizer_phase:
        circuit.add(gates.Y(0))

    destabilizer_x, destabilizer_z = symplectic_matrix[0, 0], symplectic_matrix[0, 1]
    stabilizer_x, stabilizer_z = symplectic_matrix[1, 0], symplectic_matrix[1, 1]

    if stabilizer_z and not stabilizer_x:
        if destabilizer_z:
            circuit.add(gates.S(0))
    elif not stabilizer_z and stabilizer_x:
        if destabilizer_x:
            circuit.add(gates.SDG(0))
        circuit.add(gates.H(0))
    else:
        if not destabilizer_z:
            circuit.add(gates.S(0))
        circuit.add(gates.H(0))
        circuit.add(gates.S(0))

    return circuit


def _set_qubit_x_true(clifford, circuit, qubit):
    """Set destabilizer.X[qubit, qubit] to be True.

    This is done by permuting columns l > qubit or if necessary applying
    a Hadamard
    """
    nqubits = clifford.nqubits

    x = clifford.destabilizers(symplectic=True)
    x, z = x[:, :nqubits][qubit], x[:, nqubits:-1][qubit]

    if x[qubit]:
        return

    # Try to find non-zero element
    for k in range(qubit + 1, nqubits):
        if np.all(x[k]):
            clifford.symplectic_matrix = clifford._backend.clifford_operations.SWAP(
                clifford.symplectic_matrix, k, qubit, nqubits
            )
            circuit.add(gates.SWAP(k, qubit))
            return

    # no non-zero element found: need to apply Hadamard somewhere
    for k in range(qubit, nqubits):
        if np.all(z[k]):
            clifford.symplectic_matrix = clifford._backend.clifford_operations.H(
                clifford.symplectic_matrix, k, nqubits
            )
            circuit.add(gates.H(k))
            if k != qubit:
                clifford.symplectic_matrix = clifford._backend.clifford_operations.SWAP(
                    clifford.symplectic_matrix, k, qubit, nqubits
                )
                circuit.add(gates.SWAP(k, qubit))
            return


def _set_row_x_zero(clifford, circuit, qubit):
    """Set destabilizer.X[qubit, i] to False for all i > qubit.

    This is done by applying CNOTS assumes k<=N and A[k][k]=1
    """
    nqubits = clifford.nqubits

    x = clifford.destabilizers(symplectic=True)
    x, z = x[:, :nqubits][qubit], x[:, nqubits:-1][qubit]

    # Check X first
    for k in range(qubit + 1, nqubits):
        if x[k]:
            clifford.symplectic_matrix = clifford._backend.clifford_operations.CNOT(
                clifford.symplectic_matrix, qubit, k, nqubits
            )
            circuit.add(gates.CNOT(qubit, k))

    # Check whether Zs need to be set to zero:
    if np.any(z[qubit:]):
        if not z[qubit]:
            # to treat Zs: make sure row.Z[k] to True
            clifford.symplectic_matrix = clifford._backend.clifford_operations.S(
                clifford.symplectic_matrix, qubit, nqubits
            )
            circuit.add(gates.S(qubit))

        # reverse CNOTS
        for k in range(qubit + 1, nqubits):
            if z[k]:
                clifford.symplectic_matrix = clifford._backend.clifford_operations.CNOT(
                    clifford.symplectic_matrix, k, qubit, nqubits
                )
                circuit.add(gates.CNOT(k, qubit))
        # set row.Z[qubit] to False
        clifford.symplectic_matrix = clifford._backend.clifford_operations.S(
            clifford.symplectic_matrix, qubit, nqubits
        )
        circuit.add(gates.S(qubit))


def _set_row_z_zero(clifford, circuit, qubit):
    """Set stabilizer.Z[qubit, i] to False for all i > qubit.

    Implemented by applying (reverse) CNOTs.
    It assumes ``qubit < nqubits`` and that ``_set_row_x_zero`` has been called first.
    """
    nqubits = clifford.nqubits

    x = clifford.stabilizers(symplectic=True)
    x, z = x[:, :nqubits][qubit], x[:, nqubits:-1][qubit]

    # check whether Zs need to be set to zero:
    if np.any(z[qubit + 1 :]):
        for k in range(qubit + 1, nqubits):
            if z[k]:
                clifford.symplectic_matrix = clifford._backend.clifford_operations.CNOT(
                    clifford.symplectic_matrix, k, qubit, nqubits
                )
                circuit.add(gates.CNOT(k, qubit))

    # check whether Xs need to be set to zero:
    if np.any(x[qubit:]):
        clifford.symplectic_matrix = clifford._backend.clifford_operations.H(
            clifford.symplectic_matrix, qubit, nqubits
        )
        circuit.add(gates.H(qubit))
        for k in range(qubit + 1, nqubits):
            if x[k]:
                clifford.symplectic_matrix = clifford._backend.clifford_operations.CNOT(
                    clifford.symplectic_matrix, qubit, k, nqubits
                )
                circuit.add(gates.CNOT(qubit, k))
        if z[qubit]:
            clifford.symplectic_matrix = clifford._backend.clifford_operations.S(
                clifford.symplectic_matrix, qubit, nqubits
            )
            circuit.add(gates.S(qubit))
        clifford.symplectic_matrix = clifford._backend.clifford_operations.H(
            clifford.symplectic_matrix, qubit, nqubits
        )
        circuit.add(gates.H(qubit))


def _cnot_cost(clifford):
    """Return the number of CX gates required for Clifford decomposition."""
    if clifford.nqubits > 3:
        raise_error(ValueError, "No Clifford CNOT cost function for ``nqubits > 3``.")

    if clifford.nqubits == 3:
        return _cnot_cost3(clifford)

    return _cnot_cost2(clifford)


def _rank2(a, b, c, d):
    """Return rank of 2x2 boolean matrix."""
    if (a & d) ^ (b & c):
        return 2
    if a or b or c or d:
        return 1
    return 0


def _cnot_cost2(clifford):
    """Return CNOT cost of a 2-qubit clifford."""
    symplectic_matrix = clifford.symplectic_matrix[:-1, :-1]

    r00 = _rank2(
        symplectic_matrix[0, 0],
        symplectic_matrix[0, 2],
        symplectic_matrix[2, 0],
        symplectic_matrix[2, 2],
    )
    r01 = _rank2(
        symplectic_matrix[0, 1],
        symplectic_matrix[0, 3],
        symplectic_matrix[2, 1],
        symplectic_matrix[2, 3],
    )

    if r00 == 2:
        return r01

    return r01 + 1 - r00


def _cnot_cost3(clifford):
    """Return CNOT cost of a 3-qubit clifford."""

    # pylint: disable=too-many-return-statements,too-many-boolean-expressions

    symplectic_matrix = clifford.symplectic_matrix[:-1, :-1]

    nqubits = 3
    # create information transfer matrices R1, R2
    R1 = np.zeros((nqubits, nqubits), dtype=int)
    R2 = np.zeros((nqubits, nqubits), dtype=int)
    for q1 in range(nqubits):
        for q2 in range(nqubits):
            R2[q1, q2] = _rank2(
                symplectic_matrix[q1, q2],
                symplectic_matrix[q1, q2 + nqubits],
                symplectic_matrix[q1 + nqubits, q2],
                symplectic_matrix[q1 + nqubits, q2 + nqubits],
            )
            mask = np.zeros(2 * nqubits, dtype=int)
            mask[[q2, q2 + nqubits]] = 1
            loc_y_x = np.array_equal(
                symplectic_matrix[q1, :] & mask, symplectic_matrix[q1, :]
            )
            loc_y_z = np.array_equal(
                symplectic_matrix[q1 + nqubits, :] & mask,
                symplectic_matrix[q1 + nqubits, :],
            )
            loc_y_y = np.array_equal(
                (symplectic_matrix[q1, :] ^ symplectic_matrix[q1 + nqubits, :]) & mask,
                (symplectic_matrix[q1, :] ^ symplectic_matrix[q1 + nqubits, :]),
            )
            R1[q1, q2] = 1 * (loc_y_x or loc_y_z or loc_y_y) + 1 * (
                loc_y_x and loc_y_z and loc_y_y
            )

    diag1 = np.sort(np.diag(R1)).tolist()
    diag2 = np.sort(np.diag(R2)).tolist()

    nz1 = np.count_nonzero(R1)
    nz2 = np.count_nonzero(R2)

    if diag1 == [2, 2, 2]:
        return 0

    if diag1 == [1, 1, 2]:
        return 1

    if (
        diag1 == [0, 1, 1]
        or (diag1 == [1, 1, 1] and nz2 < 9)
        or (diag1 == [0, 0, 2] and diag2 == [1, 1, 2])
    ):
        return 2

    if (
        (diag1 == [1, 1, 1] and nz2 == 9)
        or (
            diag1 == [0, 0, 1]
            and (nz1 == 1 or diag2 == [2, 2, 2] or (diag2 == [1, 1, 2] and nz2 < 9))
        )
        or (diag1 == [0, 0, 2] and diag2 == [0, 0, 2])
        or (diag2 == [1, 2, 2] and nz1 == 0)
    ):
        return 3

    if diag2 == [0, 0, 1] or (
        diag1 == [0, 0, 0]
        and (
            (diag2 == [1, 1, 1] and nz2 == 9 and nz1 == 3)
            or (diag2 == [0, 1, 1] and nz2 == 8 and nz1 == 2)
        )
    ):
        return 5

    if nz1 == 3 and nz2 == 3:
        return 6

    return 4


def _reduce_cost(clifford, inverse_circuit, cost):
    """Two-qubit cost reduction step"""
    nqubits = clifford.nqubits

    for control in range(nqubits):
        for target in range(control + 1, nqubits):
            for n0, n1 in product(range(3), repeat=2):
                # Apply a 2-qubit block
                reduced = clifford.copy(deep=True)
                for qubit, n in [(control, n0), (target, n1)]:
                    if n == 1:
                        reduced.symplectic_matrix = (
                            reduced._backend.clifford_operations.SDG(
                                reduced.symplectic_matrix, qubit, nqubits
                            )
                        )
                        reduced.symplectic_matrix = (
                            reduced._backend.clifford_operations.H(
                                reduced.symplectic_matrix, qubit, nqubits
                            )
                        )
                    elif n == 2:
                        reduced.symplectic_matrix = (
                            reduced._backend.clifford_operations.SDG(
                                reduced.symplectic_matrix, qubit, nqubits
                            )
                        )
                        reduced.symplectic_matrix = (
                            reduced._backend.clifford_operations.H(
                                reduced.symplectic_matrix, qubit, nqubits
                            )
                        )
                        reduced.symplectic_matrix = (
                            reduced._backend.clifford_operations.SDG(
                                reduced.symplectic_matrix, qubit, nqubits
                            )
                        )
                        reduced.symplectic_matrix = (
                            reduced._backend.clifford_operations.H(
                                reduced.symplectic_matrix, qubit, nqubits
                            )
                        )
                reduced.symplectic_matrix = reduced._backend.clifford_operations.CNOT(
                    reduced.symplectic_matrix, control, target, nqubits
                )

                # Compute new cost
                new_cost = _cnot_cost(reduced)

                if new_cost == cost - 1:
                    # Add decomposition to inverse circuit
                    for qubit, n in [(control, n0), (target, n1)]:
                        if n == 1:
                            inverse_circuit.add(gates.SDG(qubit))
                            inverse_circuit.add(gates.H(qubit))
                        elif n == 2:
                            inverse_circuit.add(gates.H(qubit))
                            inverse_circuit.add(gates.S(qubit))
                    inverse_circuit.add(gates.CNOT(control, target))

                    return reduced, inverse_circuit, new_cost

    raise_error(RuntimeError, "Failed to reduce CNOT cost.")
