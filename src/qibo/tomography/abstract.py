"""Abstract class with the methods shared by tomographic protocols."""

from abc import ABC, abstractmethod

from qibo import Circuit, gates
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

    def _pauli_circuits(
        self, circuit: Circuit, observables: list[str]
    ) -> list[Circuit]:
        """Appends the basis changes and the measurements of each Pauli string to ``circuit``.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit preparing the
                state, without measurements.
            observables (list): Pauli strings to be measured, each one a ``str`` of
                ``"I"``, ``"X"``, ``"Y"`` and ``"Z"``, one per qubit, *e.g.* ``"XIZ"``.

        Returns:
            list: one circuit for each Pauli string, in the same order. Each circuit
            measures all qubits, after rotating the Pauli :math:`X` with
            :class:`qibo.gates.H`, and the Pauli :math:`Y` with :class:`qibo.gates.SDG`
            followed by :class:`qibo.gates.H`. The identity and the Pauli :math:`Z` need
            no rotation. The outcomes of the qubits of the identity are not used.
        """
        self._check_circuit(circuit)

        nqubits = circuit.nqubits

        for observable in observables:
            if (
                not isinstance(observable, str)
                or len(observable) != nqubits
                or set(observable) - set("IXYZ")
            ):
                raise_error(
                    ValueError,
                    f"``observables`` must be strings of {nqubits} characters "
                    + f"``I``, ``X``, ``Y`` or ``Z``, but one is {observable!r}.",
                )

        circuits = []
        for observable in observables:
            tomography_circuit = circuit.copy()
            for qubit, pauli in enumerate(observable):
                if pauli == "Y":
                    tomography_circuit.add(gates.SDG(qubit))
                if pauli in "XY":
                    tomography_circuit.add(gates.H(qubit))
            tomography_circuit.add(gates.M(*range(nqubits)))
            circuits.append(tomography_circuit)

        return circuits
