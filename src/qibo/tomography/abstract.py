"""Abstract class with the methods shared by tomographic protocols."""

from abc import ABC, abstractmethod

from qibo import Circuit
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error


class Tomography(ABC):
    """Abstract class for tomographic protocols.

    A protocol is defined by (i) :meth:`circuits`, which builds the circuits that need
    to be executed, and (ii) :meth:`__call__`, which runs the whole pipeline, i.e.
    circuits generation, execution, and post-processing.
    """

    @abstractmethod
    def __call__(self, circuit: Circuit, *args, **kwargs):
        """Runs the whole tomographic pipeline for ``circuit``."""

    @abstractmethod
    def circuits(self, circuit: Circuit, *args, **kwargs) -> list[Circuit]:
        """Generates the circuits that need to be executed by the protocol."""

    def execute(
        self,
        circuits: list[Circuit],
        nshots: int = 1000,
        backend: Backend | None = None,
    ) -> list:
        """Executes a list of circuits.

        Args:
            circuits (list[:class:`qibo.models.circuit.Circuit`]): circuits to be executed.
            nshots (int, optional): number of shots per circuit.
                Defaults to :math:`1000`.
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend
                to be used in the execution. If ``None``, it uses the current
                backend. Defaults to ``None``.

        Returns:
            list: execution results, in the same order as ``circuits``.
        """
        backend = _check_backend(backend)

        return [backend.execute_circuit(circuit, nshots=nshots) for circuit in circuits]

    def _check_circuit(self, circuit: Circuit):
        """Checks that ``circuit`` has no measurements, since protocols add their own.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be checked.
        """
        if circuit.measurements:
            raise_error(ValueError, "``circuit`` must not contain measurement gates.")
