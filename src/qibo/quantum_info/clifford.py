"""Module definig the Clifford object, which allows phase-space representation of Clifford circuits and stabilizer states."""

from dataclasses import dataclass, field
from functools import reduce
from itertools import product
from typing import Optional, Union

import numpy as np

from qibo import Circuit
from qibo.backends import CliffordBackend
from qibo.config import raise_error
from qibo.gates import M
from qibo.measurements import frequencies_to_binary

from ._clifford_utils import _decomposition_AG04, _decomposition_BM20, _string_product


@dataclass
class Clifford:
    """Object storing the results of a circuit execution with the :class:`qibo.backends.clifford.CliffordBackend`.

    Args:
        data (ndarray or :class:`qibo.models.circuit.Circuit`): If ``ndarray``, it is the
            symplectic matrix of the stabilizer state in phase-space representation.
            If :class:`qibo.models.circuit.Circuit`, it is a circuit composed only of Clifford
            gates and computational-basis measurements.
        nqubits (int, optional): number of qubits of the state.
        measurements (list, optional): list of measurements gates :class:`qibo.gates.M`.
            Defaults to ``None``.
        nshots (int, optional): number of shots used for sampling the measurements.
            Defaults to :math:`1000`.
        engine (str, optional): engine to use in the execution of the
            :class:`qibo.backends.CliffordBackend`. It accepts ``"numpy"``, ``"numba"``,
            ``"cupy"``, and ``"stim"`` (see `stim <https://github.com/quantumlib/Stim>`_).
            If ``None``, defaults to the corresponding engine
            from the current backend. Defaults to ``None``.
    """

    symplectic_matrix: np.ndarray = field(init=False)
    data: Union[np.ndarray, Circuit] = field(repr=False)
    nqubits: Optional[int] = None
    measurements: Optional[list] = None
    nshots: int = 1000
    engine: Optional[str] = None

    _backend: Optional[CliffordBackend] = None
    _measurement_gate = None
    _samples: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.data, Circuit):
            clifford = self.from_circuit(self.data, engine=self.engine)
            self.symplectic_matrix = clifford.symplectic_matrix
            self.nqubits = clifford.nqubits
            self.measurements = clifford.measurements
            self._samples = clifford._samples
            self._measurement_gate = clifford._measurement_gate
        else:
            # adding the scratch row if not provided
            self.symplectic_matrix = self.data
            if self.symplectic_matrix.shape[0] % 2 == 0:
                self.symplectic_matrix = np.vstack(
                    (
                        self.symplectic_matrix,
                        np.zeros(self.symplectic_matrix.shape[1], dtype=np.uint8),
                    )
                )
            self.nqubits = int((self.symplectic_matrix.shape[1] - 1) / 2)
        if self._backend is None:
            self._backend = CliffordBackend(self.engine)
        self.engine = self._backend.engine

    @classmethod
    def from_circuit(
        cls,
        circuit: Circuit,
        initial_state: Optional[np.ndarray] = None,
        nshots: int = 1000,
        engine: Optional[str] = None,
    ):
        """Allows to create a :class:`qibo.quantum_info.clifford.Clifford` object by executing the input circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Clifford circuit to run.
            initial_state (ndarray, optional): symplectic matrix of the initial state.
                If ``None``, defaults to the symplectic matrix of the zero state.
                Defaults to ``None``.
            nshots (int, optional): number of measurement shots to perform
                if ``circuit`` has measurement gates. Defaults to :math:`10^{3}`.
            engine (str, optional): engine to use in the execution of the
                :class:`qibo.backends.CliffordBackend`. It accepts ``"numpy"``, ``"numba"``,
                ``"cupy"``, and ``"stim"`` (see `stim <https://github.com/quantumlib/Stim>`_).
                If ``None``, defaults to the corresponding engine
                from the current backend. Defaults to ``None``.

        Returns:
            (:class:`qibo.quantum_info.clifford.Clifford`): Object storing the result of the circuit execution.
        """
        cls._backend = CliffordBackend(engine)

        return cls._backend.execute_circuit(circuit, initial_state, nshots)

    def to_circuit(self, algorithm: Optional[str] = "AG04"):
        """Converts symplectic matrix into a Clifford circuit.

        Args:
            algorithm (str, optional): If ``AG04``, uses the decomposition algorithm from
                `Aaronson & Gottesman (2004) <https://arxiv.org/abs/quant-ph/0406196>`_.
                If ``BM20`` and ``Clifford.nqubits <= 3``, uses the decomposition algorithm from
                `Bravyi & Maslov (2020) <https://arxiv.org/abs/2003.09412>`_.
                Defaults to ``AG04``.

        Returns:
            :class:`qibo.models.circuit.Circuit`: circuit composed of Clifford gates.
        """
        if not isinstance(algorithm, str):
            raise_error(
                TypeError,
                f"``algorithm`` must be type str, but it is type {type(algorithm)}",
            )

        if algorithm not in ["AG04", "BM20"]:
            raise_error(ValueError, f"``algorithm`` {algorithm} not found.")

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
            symplectic (bool, optional): If ``True``, returns the rows of the symplectic matrix
                that correspond to the :math:`n` generators of the :math:`2^{n}` total stabilizers,
                independently of ``return_array``.
            return_array (bool, optional): To be used when ``symplectic = False``.
                If ``True`` returns the stabilizers as ``ndarray``.
                If ``False``, returns stabilizers as strings. Defaults to ``False``.

        Returns:
            (ndarray or list): Stabilizers of the state.
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
            symplectic (bool, optional): If ``True``, returns the rows of the symplectic matrix
                that correspond to the :math:`n` generators of the :math:`2^{n}` total
                destabilizers, independently of ``return_array``.
            return_array (bool, optional): To be used when ``symplectic = False``.
                If ``True`` returns the destabilizers as ``ndarray``.
                If ``False``, their representation as strings is returned.
                Defaults to ``False``.

        Returns:
            (ndarray or list): Destabilizers of the state.
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
                gate.result.register_samples(self._samples[:, rqubits])

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
                the keys of the :class:`collections.Counter` are in binary form,
                as strings of :math:`0` and :math`1`.
            If ``binary`` is ``False``
                the keys of the :class:`collections.Counter` are integers.
            If ``registers`` is ``True``
                a `dict` of :class:`collections.Counter` is returned where keys are
                the name of each register.
            If ``registers`` is ``False``
                a single :class:`collections.Counter` is returned which contains samples
                from all the measured qubits, independently of their registers.
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

        probs = self.engine.cast(probs, float) / len(samples)

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
            symplectic_matrix,
            self.nqubits,
            self.measurements,
            self.nshots,
            _backend=self._backend,
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
