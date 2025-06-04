from dataclasses import dataclass
from typing import Optional, Union

from qiboml import ndarray

from qibo import Circuit, gates, transpiler
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian, Z
from qibo.hamiltonians.hamiltonians import SymbolicHamiltonian
from qibo.result import CircuitResult, MeasurementOutcomes, QuantumState
from qibo.transpiler import Passes


@dataclass
class QuantumDecoding:
    """
    Abstract decoder class.

    Args:
        nqubits (int): total number of qubits.
        qubits (tuple[int], optional): set of qubits it acts on, by default ``range(nqubits)``.
        nshots (int, optional): number of shots used for circuit execution and sampling.
        backend (Backend, optional): backend used for computation, by default the globally-set backend is used.
        transpiler (Passes, optional): transpiler to run before circuit execution, by default no transpilation
                                       is performed on the circuit (``transpiler=None``).
    """

    nqubits: int
    qubits: Optional[tuple[int]] = None
    nshots: Optional[int] = None
    backend: Optional[Backend] = None
    transpiler: Optional[Passes] = None
    _circuit: Circuit = None

    def __post_init__(self):
        """Ancillary post initialization operations."""
        self.qubits = (
            tuple(range(self.nqubits)) if self.qubits is None else tuple(self.qubits)
        )
        self._circuit = Circuit(self.nqubits)
        self.backend = _check_backend(self.backend)
        self._circuit.add(gates.M(*self.qubits))

    def __call__(
        self, x: Circuit
    ) -> Union[CircuitResult, QuantumState, MeasurementOutcomes]:
        """Combine the input circuir with the internal one and execute them with the internal backend.

        Args:
            x (Circuit): input circuit.

        Returns:
            (CircuitResult | QuantumState | MeasurementOutcomes): the execution ``qibo.result`` object.
        """
        self._circuit.density_matrix = x.density_matrix
        self._circuit.init_kwargs["density_matrix"] = x.density_matrix
        if self.transpiler is not None:
            x, _ = self.transpiler(x)
        return self.backend.execute_circuit(x + self._circuit, nshots=self.nshots)

    @property
    def circuit(
        self,
    ) -> Circuit:
        """A copy of the internal circuit.

        Returns:
            (Circuit): a copy of the internal circuit.
        """
        return self._circuit.copy()

    def set_backend(self, backend: Backend):
        """Set the internal backend.

        Args:
            backend (Backend): backend to be set.
        """
        self.backend = backend

    @property
    def output_shape(self):
        """The shape of the decoded outputs."""
        raise_error(NotImplementedError)

    @property
    def analytic(self) -> bool:
        """Whether the decoder is analytic, i.e. the gradient is ananlytically computable, or not
        (e.g. if sampling is involved).

        Returns:
            (bool): ``True`` if ``nshots`` is ``None``, ``False`` otherwise.
        """
        if self.nshots is None:
            return True
        return False

    def __hash__(self) -> int:
        return hash((self.qubits, self.nshots, self.backend))


class Probabilities(QuantumDecoding):
    """The probabilities decoder."""

    # TODO: collapse on ExpectationDecoding if not analytic

    def __call__(self, x: Circuit) -> ndarray:
        """Computes the final state probabilities.

        Args:
            x (Circuit): input circuit.

        Returns:
            (ndarray): the final probabilities.
        """
        return super().__call__(x).probabilities(self.qubits)

    @property
    def output_shape(self) -> tuple[int, int]:
        """Shape of the output probabilities.

        Returns:
            (tuple[int, int]): a ``(1, 2**nqubits)`` shape.
        """
        return (1, 2**self.nqubits)

    @property
    def analytic(self) -> bool:
        return True


@dataclass
class Expectation(QuantumDecoding):
    r"""The expectation value decoder.

    Args:
        observable (Hamiltonian | ndarray): the observable to calculate the expectation value of,
    by default :math:`Z_0\otimes Z_1\otimes ... \otimes Z_n` is used.
    """

    observable: Union[ndarray, Hamiltonian] = None

    def __post_init__(self):
        """Ancillary post initialization operations."""
        if self.observable is None:
            self.observable = Z(len(self.qubits), dense=True, backend=self.backend)
        super().__post_init__()

    def __call__(self, x: Circuit) -> ndarray:
        """Execute the input circuit and calculate the expectation value of the internal observable on
        the final state

        Args:
            x (Circuit): input Circuit.

        Returns:
            (ndarray): the calculated expectation value.
        """
        if self.analytic:
            return self.observable.expectation(
                super().__call__(x).state(),
            ).reshape(1, 1)
        else:
            if isinstance(self.observable, SymbolicHamiltonian):
                self._circuit.density_matrix = x.density_matrix
                self._circuit.init_kwargs["density_matrix"] = x.density_matrix
                x = x + self._circuit
                if self.transpiler is not None:
                    x, _ = self.transpiler(x)
                return self.observable.expectation_from_circuit(
                    x,
                    nshots=self.nshots,
                ).reshape(1, 1)
            else:
                return self.observable.expectation_from_samples(
                    super().__call__(x).frequencies(),
                    qubit_map=self.qubits,
                ).reshape(1, 1)

    @property
    def output_shape(self) -> tuple[int, int]:
        """Shape of the output expectation value.

        Returns:
            (tuple[int, int]): a ``(1, 1)`` shape.
        """
        return (1, 1)

    def set_backend(self, backend: Backend):
        """Set the internal and observable's backends.

        Args:
            backend (Backend): backend to be set.
        """
        if isinstance(self.observable, Hamiltonian):
            matrix = self.backend.to_numpy(self.observable.matrix)
            super().set_backend(backend)
            self.observable = Hamiltonian(
                nqubits=self.nqubits,
                matrix=self.backend.cast(matrix),
                backend=self.backend,
            )
        else:
            super().set_backend(backend)
            self.observable.backend = backend

    def __hash__(self) -> int:
        return hash((self.qubits, self.nshots, self.backend, self.observable))


class State(QuantumDecoding):
    """The state decoder."""

    def __call__(self, x: Circuit) -> ndarray:
        """Compute the final state of the input circuit and separates it in its real and
        imaginary parts stacked on top of each other.

        Args:
            x (Circuit): input Circuit.

        Returns:
            (ndarray): the final state.
        """
        state = super().__call__(x).state()
        return self.backend.np.vstack(  # pylint: disable=no-member
            (
                self.backend.np.real(state),  # pylint: disable=no-member
                self.backend.np.imag(state),  # pylint: disable=no-member
            )
        ).reshape(self.output_shape)

    @property
    def output_shape(self) -> tuple[int, int, int]:
        """Shape of the output state.

        Returns:
            (tuple[int, int, int]): a ``(2, 1, 2**nqubits)`` shape.
        """
        return (2, 1, 2**self.nqubits)

    @property
    def analytic(self) -> bool:
        return True


class Samples(QuantumDecoding):
    """The samples decoder."""

    def __post_init__(self):
        super().__post_init__()

    def __call__(self, x: Circuit) -> ndarray:
        """Sample the final state of the circuit.

        Args:
            x (Circuit): input Circuit.

        Returns:
            (ndarray): the generated samples.
        """
        return self.backend.cast(super().__call__(x).samples(), self.backend.precision)

    @property
    def output_shape(self) -> tuple[int, int]:
        """Shape of the output samples.

        Returns:
            (tuple[int, int]): a ``(nshots, nqubits)`` shape.
        """
        return (self.nshots, len(self.qubits))

    @property
    def analytic(self) -> bool:  # pragma: no cover
        return False
