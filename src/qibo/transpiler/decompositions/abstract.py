from qibo import gates
from qibo.backends import Backend
from qibo.config import raise_error


class GateDecompositions:
    """Abstract data structure that holds decompositions of gates."""

    def __init__(self):
        self.decompositions = {}
        self._epsilon = None

    def add(self, gate, decomposition):
        """Register a decomposition for a gate."""
        self.decompositions[gate] = decomposition

    def _check_instance(self, gate: gates.Gate, backend: Backend | None = None) -> list:
        special_gates = (
            gates.FusedGate,
            gates.Unitary,
            gates.GeneralizedfSim,
            gates.fSim,
        )
        decomposition = self.decompositions[gate.__class__]

        if isinstance(gate, gates.FanOut):
            decomposition = decomposition(gate)
        elif gate.parameters:
            decomposition = (
                decomposition(gate, backend)
                if isinstance(gate, special_gates)
                else decomposition(gate)
            )

        return decomposition

    def count_2q(self, gate: gates.Gate, backend: Backend | None = None) -> int:
        """Count the number of two-qubit gates in the decomposition of the given gate."""
        decomposition = self._check_instance(gate, backend)
        return len(tuple(g for g in decomposition if len(g.qubits) > 1))

    def count_1q(self, gate: gates.Gate, backend: Backend | None = None) -> int:
        """Count the number of single qubit gates in the decomposition of the given gate."""
        decomposition = self._check_instance(gate, backend)
        return len(tuple(g for g in decomposition if len(g.qubits) == 1))

    def __call__(self, gate: gates.Gate, backend: Backend | None = None) -> list:
        """Decompose a gate."""
        decomposition = self._check_instance(gate, backend)

        # Decompositions given as functions create new gates in every call, which
        # already act on the right qubits if the gate acts on ``0, ..., n - 1``.
        # The gates of static decompositions are shared between calls, so they are
        # copied. The copy also remaps them onto the actual qubits of the gate.
        creates_gates = callable(self.decompositions[gate.__class__])
        if creates_gates and gate.qubits == tuple(range(len(gate.qubits))):
            return decomposition

        return [g.on_qubits(dict(enumerate(gate.qubits))) for g in decomposition]

    def _set_precision_cliff_plus_t(
        self, epsilon: float = 1e-16, mpmath_dps: int = 256
    ):
        import mpmath

        mpmath.mp.dps = mpmath_dps
        self._epsilon = mpmath.mpmathify(epsilon)

    def _rz_into_cliff_and_t(
        self,
        theta: float,
        qubit: int = 0,
    ):
        try:
            import mpmath
            from pygridsynth.gridsynth import gridsynth_gates

            theta = float(theta)
            theta = mpmath.mpmathify(theta)

            sequence = gridsynth_gates(theta=theta, epsilon=self._epsilon)
            # the "W" gates are global phases, which are dropped
            sequence = sequence.split("W")[0]

            gate_list = [
                getattr(gates, gate_name)(qubit) for gate_name in reversed(sequence)
            ]

            return gate_list
        except ModuleNotFoundError:  # pragma: no cover
            raise_error(
                NotImplementedError,
                "``pygridsynth`` and ``mpmath`` are required for this decomposition.",
            )
