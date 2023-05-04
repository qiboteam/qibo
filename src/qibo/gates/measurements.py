from typing import Dict, Optional, Tuple

from qibo.config import raise_error
from qibo.gates.abstract import Gate
from qibo.gates.gates import Z
from qibo.states import MeasurementResult


class M(Gate):
    """The measure gate.

    Args:
        *q (int): id numbers of the qubits to measure.
            It is possible to measure multiple qubits using ``gates.M(0, 1, 2, ...)``.
            If the qubits to measure are held in an iterable (eg. list) the ``*``
            operator can be used, for example ``gates.M(*[0, 1, 4])`` or
            ``gates.M(*range(5))``.
        register_name (str): Optional name of the register to distinguish it
            from other registers when used in circuits.
        collapse (bool): Collapse the state vector after the measurement is
            performed. Can be used only for single shot measurements.
            If ``True`` the collapsed state vector is returned. If ``False``
            the measurement result is returned.
        basis (:class:`qibo.gates.Gate`, list): Basis to measure.
            Can be a qibo gate or a callable that accepts a qubit,
            for example: ``lambda q: gates.RX(q, 0.2)``
            or a list of these, if a different basis will be used for each
            measurement qubit.
            Default is Z.
        p0 (dict): Optional bitflip probability map. Can be:
            A dictionary that maps each measured qubit to the probability
            that it is flipped, a list or tuple that has the same length
            as the tuple of measured qubits or a single float number.
            If a single float is given the same probability will be used
            for all qubits.
        p1 (dict): Optional bitflip probability map for asymmetric bitflips.
            Same as ``p0`` but controls the 1->0 bitflip probability.
            If ``p1`` is ``None`` then ``p0`` will be used both for 0->1 and
            1->0 bitflips.
    """

    def __init__(
        self,
        *q,
        register_name: Optional[str] = None,
        collapse: bool = False,
        basis: Gate = Z,
        p0: Optional["ProbsType"] = None,
        p1: Optional["ProbsType"] = None,
    ):
        super().__init__()
        self.name = "measure"
        self.draw_label = "M"
        self.target_qubits = tuple(q)
        self.register_name = register_name
        self.collapse = collapse
        self.result = MeasurementResult(self)
        # list of measurement pulses implementing the gate
        # relevant for experiments only
        self.pulses = None

        self.init_args = q
        self.init_kwargs = {
            "register_name": register_name,
            "collapse": collapse,
            "p0": p0,
            "p1": p1,
        }
        if collapse:
            if p0 is not None or p1 is not None:
                raise_error(
                    NotImplementedError,
                    "Bitflip measurement noise is not available when collapsing.",
                )

        if p1 is None:
            p1 = p0
        if p0 is None:
            p0 = p1
        self.bitflip_map = (self._get_bitflip_map(p0), self._get_bitflip_map(p1))

        # list of gates that will be added to the circuit before the
        # measurement, in order to rotate to the given basis
        if not isinstance(basis, list):
            basis = len(self.target_qubits) * [basis]
        elif len(basis) != len(self.target_qubits):
            raise_error(
                ValueError,
                f"Given basis list has length {len(basis)} while "
                f"we are measuring {len(self.target_qubits)} qubits.",
            )
        self.basis = []
        for qubit, basis_cls in zip(self.target_qubits, basis):
            gate = basis_cls(qubit).basis_rotation()
            if gate is not None:
                self.basis.append(gate)

    @staticmethod
    def _get_bitflip_tuple(qubits: Tuple[int], probs: "ProbsType") -> Tuple[float]:
        if isinstance(probs, float):
            if probs < 0 or probs > 1:  # pragma: no cover
                raise_error(ValueError, f"Invalid bitflip probability {probs}.")
            return len(qubits) * (probs,)

        if isinstance(probs, (tuple, list)):
            if len(probs) != len(qubits):
                raise_error(
                    ValueError,
                    f"{len(qubits)} qubits were measured but the given "
                    + f"bitflip probability list contains {len(probs)} values.",
                )
            return tuple(probs)

        if isinstance(probs, dict):
            diff = set(probs.keys()) - set(qubits)
            if diff:
                raise_error(
                    KeyError,
                    f"Bitflip map contains {diff} qubits that are not measured.",
                )
            return tuple(probs[q] if q in probs else 0.0 for q in qubits)

        raise_error(TypeError, "Invalid type {} of bitflip map.".format(probs))

    def _get_bitflip_map(self, p: Optional["ProbsType"] = None) -> Dict[int, float]:
        """Creates dictionary with bitflip probabilities."""
        if p is None:
            return {q: 0 for q in self.qubits}
        pt = self._get_bitflip_tuple(self.qubits, p)
        return dict(zip(self.qubits, pt))

    def has_bitflip_noise(self):
        return (
            sum(self.bitflip_map[0].values()) > 0
            or sum(self.bitflip_map[1].values()) > 0
        )

    def add(self, gate):
        """Adds target qubits to a measurement gate.

        This method is only used for creating the global measurement gate used
        by the `models.Circuit`.
        The user is not supposed to use this method and a `ValueError` is
        raised if he does so.

        Args:
            gate: Measurement gate to add its qubits in the current gate.
        """
        assert isinstance(gate, self.__class__)
        self.target_qubits += gate.target_qubits
        self.bitflip_map[0].update(gate.bitflip_map[0])
        self.bitflip_map[1].update(gate.bitflip_map[1])

    def controlled_by(self, *q):
        """"""
        raise_error(NotImplementedError, "Measurement gates cannot be controlled.")

    def asmatrix(self, backend):
        """"""
        raise_error(
            NotImplementedError, "Measurement gates do not have matrix representation."
        )

    def apply(self, backend, state, nqubits):
        self.result.backend = backend
        if not self.collapse:
            return state

        qubits = sorted(self.target_qubits)
        # measure and get result
        probs = backend.calculate_probabilities(state, qubits, nqubits)
        shot = self.result.add_shot(probs)
        # collapse state
        return backend.collapse_state(state, qubits, shot, nqubits)

    def apply_density_matrix(self, backend, state, nqubits):
        self.result.backend = backend
        if not self.collapse:
            return state

        qubits = sorted(self.target_qubits)
        # measure and get result
        probs = backend.calculate_probabilities_density_matrix(state, qubits, nqubits)
        shot = self.result.add_shot(probs)
        # collapse state
        return backend.collapse_density_matrix(state, qubits, shot, nqubits)
