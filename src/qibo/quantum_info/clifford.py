"""Module definig the Clifford object, which allows phase-space representation of Clifford circuits and stabilizer states."""

from dataclasses import dataclass
from functools import reduce
from itertools import product
from typing import Optional, Union

import numpy as np

from qibo import Circuit
from qibo.backends import Backend, CliffordBackend
from qibo.config import raise_error
from qibo.gates import M
from qibo.measurements import frequencies_to_binary


@dataclass
class Clifford:
    """Object storing the results of a circuit execution with the :class:`qibo.backends.clifford.CliffordBackend`.

    Args:
        symplectic_matrix (ndarray): Symplectic matrix of the state in phase-space representation.
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

    symplectic_matrix: np.ndarray
    nqubits: Optional[int] = None
    measurements: Optional[list] = None
    nshots: int = 1000
    engine: Optional[Backend] = None

    _backend: Optional[CliffordBackend] = None
    _measurement_gate = None
    _samples: Optional[int] = None

    def __post_init__(self):
        # adding the scratch row if not provided
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
        """Allows to create a ``Clifford`` object by executing the input circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Clifford circuit to run.
            initial_state (np.ndarray): The initial tableu state.
            nshots (int): The number of shots to perform.

        Returns:
            (:class:`qibo.quantum_info.clifford.Clifford`): The object storing the result of the circuit execution.
        """
        cls._backend = CliffordBackend(engine)

        return cls._backend.execute_circuit(circuit, initial_state, nshots)

    # TODO: implement this method in separate PR based on `Bravyi & Maslov (2022) <https://arxiv.org/abs/2003.09412>`_.
    @classmethod
    def to_circuit(cls, algorithm=None):  # pragma: no cover
        raise_error(NotImplementedError, "`to_circuit` method not implemented yet.")

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

    def stabilizers(self, return_array: bool = False):
        """Extracts the stabilizers of the state.

        Args:
            return_array (bool, optional): If ``True`` returns the stabilizers as ``ndarray``.
                If ``False``, returns stabilizers as strings. Defaults to ``False``.

        Returns:
            (list): Stabilizers of the state.
        """
        generators, phases = self.generators(return_array)

        return self._construct_operators(
            generators[self.nqubits :],
            phases[self.nqubits :],
        )

    def destabilizers(self, return_array: bool = False):
        """Extracts the destabilizers of the state.

        Args:
            return_array (bool, optional): If ``True`` returns the destabilizers as ``ndarray``.
                If ``False``, their representation as strings is returned. Defaults to ``False``.

        Returns:
            (list): Destabilizers of the state.
        """
        generators, phases = self.generators(return_array)

        return self._construct_operators(
            generators[: self.nqubits], phases[: self.nqubits]
        )

    def state(self):
        """Builds the density matrix representation of the state.

        .. note::
            This method is inefficient in runtime and memory for a large number of qubits.

        Returns:
            (ndarray): Density matrix of the state.
        """
        stabilizers = self.stabilizers(True)

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
