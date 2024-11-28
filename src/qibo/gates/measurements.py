import json
from typing import Dict, Optional, Tuple, Union

from qibo import gates
from qibo.config import raise_error
from qibo.gates.abstract import Gate
from qibo.gates.gates import Z
from qibo.measurements import MeasurementResult


class M(Gate):
    """The measure gate.

    Args:
        *q (int): id numbers of the qubits to measure.
            It is possible to measure multiple qubits using ``gates.M(0, 1, 2, ...)``.
            If the qubits to measure are held in an iterable (eg. list) the ``*``
            operator can be used, for example ``gates.M(*[0, 1, 4])`` or
            ``gates.M(*range(5))``.
        register_name (str, optional): Optional name of the register to distinguish it
            from other registers when used in circuits.
        collapse (bool): Collapse the state vector after the measurement is
            performed. Can be used only for single shot measurements.
            If ``True`` the collapsed state vector is returned. If ``False``
            the measurement result is returned.
        basis (:class:`qibo.gates.Gate` or str or list, optional): Basis to measure.
            Can be either:
            - a qibo gate
            - the string representing the gate
            - a callable that accepts a qubit, for example: ``lambda q: gates.RX(q, 0.2)``
            - a list of the above, if a different basis will be used for each measurement qubit.
            Defaults is to :class:`qibo.gates.Z`.
        p0 (dict, optional): bitflip probability map. Can be:
            A dictionary that maps each measured qubit to the probability
            that it is flipped, a list or tuple that has the same length
            as the tuple of measured qubits or a single float number.
            If a single float is given the same probability will be used
            for all qubits.
        p1 (dict, optional): bitflip probability map for asymmetric bitflips.
            Same as ``p0`` but controls the 1->0 bitflip probability.
            If ``p1`` is ``None`` then ``p0`` will be used both for 0->1 and
            1->0 bitflips.
    """

    def __init__(
        self,
        *q,
        register_name: Optional[str] = None,
        collapse: bool = False,
        basis: Union[Gate, str] = Z,
        p0: Optional["ProbsType"] = None,  # type: ignore
        p1: Optional["ProbsType"] = None,  # type: ignore
    ):
        super().__init__()
        self.name = "measure"
        self.draw_label = "M"
        self.target_qubits = tuple(q)
        self.register_name = register_name
        self.collapse = collapse
        self.result = MeasurementResult(self.target_qubits)
        # list of measurement pulses implementing the gate
        # relevant for experiments only
        self.pulses = None
        # saving basis for __repr__ ans save to file
        to_gate = lambda x: getattr(gates, x) if isinstance(x, str) else x
        if not isinstance(basis, list):
            self.basis_gates = len(q) * [to_gate(basis)]
            basis = len(self.target_qubits) * [basis]
        elif len(basis) != len(self.target_qubits):
            raise_error(
                ValueError,
                f"Given basis list has length {len(basis)} while "
                f"we are measuring {len(self.target_qubits)} qubits.",
            )
        else:
            self.basis_gates = [to_gate(g) for g in basis]

        self.init_args = q
        self.init_kwargs = {
            "register_name": register_name,
            "collapse": collapse,
            "basis": [g.__name__ for g in self.basis_gates],
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
        self.basis = []
        for qubit, basis_cls in zip(self.target_qubits, self.basis_gates):
            gate = basis_cls(qubit).basis_rotation()
            if gate is not None:
                self.basis.append(gate)

    @property
    def raw(self) -> dict:
        """Serialize to dictionary.

        The values used in the serialization should be compatible with a
        JSON dump (or any other one supporting a minimal set of scalar
        types). Though the specific implementation is up to the specific
        gate.
        """
        encoded_simple = super().raw
        encoded_simple.update({"measurement_result": self.result.raw})
        return encoded_simple

    @staticmethod
    def _get_bitflip_tuple(
        qubits: Tuple[int, ...], probs: "ProbsType"  # type: ignore
    ) -> Tuple[float, ...]:
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

        raise_error(TypeError, f"Invalid type {probs} of bitflip map.")

    def _get_bitflip_map(self, p: Optional["ProbsType"] = None) -> Dict[int, float]:  # type: ignore
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
        by the :class:`qibo.models.Circuit`.
        The user is not supposed to use this method and a ``ValueError`` is
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

    def matrix(self, backend=None):
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
        shot = self.result.add_shot(probs, backend=backend)
        # collapse state
        return backend.collapse_state(state, qubits, shot, nqubits)

    def apply_density_matrix(self, backend, state, nqubits):
        self.result.backend = backend
        if not self.collapse:
            return state

        qubits = sorted(self.target_qubits)
        # measure and get result
        probs = backend.calculate_probabilities_density_matrix(state, qubits, nqubits)
        shot = self.result.add_shot(probs, backend=backend)
        # collapse state
        return backend.collapse_density_matrix(state, qubits, shot, nqubits)

    def apply_clifford(self, backend, state, nqubits):
        self.result.backend = backend
        if not self.collapse:
            return state

        qubits = sorted(self.target_qubits)
        sample = backend.sample_shots(state, qubits, nqubits, 1, self.collapse)
        self.result.add_shot_from_sample(sample[0])
        return state

    @classmethod
    def load(cls, payload):
        """Constructs a measurement gate starting from a json serialized
        one."""
        args = json.loads(payload)
        return cls.from_dict(args)

    # Overload on_qubits to copy also gate.result, controlled by can be removed for measurements
    def on_qubits(self, qubit_map) -> "Gate":
        """Creates the same measurement gate targeting different qubits
        and preserving the measurement result register.

        Args:
            qubit_map (dict): dictionary mapping original qubit indices to new ones.

        Returns:
            :class:`qibo.gates.Gate.M`: object of the original gate type targeting
            the given qubits.

        Example:

            .. testcode::

                from qibo import Circuit, gates

                measurement = gates.M(0, 1)

                circuit = Circuit(3)
                circuit.add(measurement.on_qubits({0: 0, 1: 2}))
                assert circuit.queue[0].result is measurement.result
                circuit.draw()
            .. testoutput::

                0: ─M─
                1: ─|─
                2: ─M─
        """

        qubits = (qubit_map.get(q) for q in self.qubits)
        gate = self.__class__(*qubits, **self.init_kwargs)
        gate.result = self.result
        return gate
