from qibo.config import raise_error
from qibo.gates.abstract import Gate
from typing import Dict, Optional, Tuple


class M(Gate):
    """The Measure Z gate.

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

    def __init__(self, *q, register_name: Optional[str] = None,
                 collapse: bool = False,
                 p0: Optional["ProbsType"] = None,
                 p1: Optional["ProbsType"] = None):
        super().__init__()
        self.name = "measure"
        self.target_qubits = q
        self.register_name = register_name
        self.collapse = collapse
        self.result = None
        self._symbol = None

        self.init_args = q
        self.init_kwargs = {"register_name": register_name,
                            "collapse": collapse,
                            "p0": p0, "p1": p1}
        if collapse and (p0 is not None or p1 is not None):
            raise_error(NotImplementedError, "Bitflip measurement noise is not "
                                             "available when collapsing.")

        if p1 is None: p1 = p0
        if p0 is None: p0 = p1
        self.bitflip_map = (self._get_bitflip_map(p0),
                            self._get_bitflip_map(p1))

    @staticmethod
    def _get_bitflip_tuple(qubits: Tuple[int], probs: "ProbsType"
                           ) -> Tuple[float]:
        if isinstance(probs, float):
            if probs < 0 or probs > 1:
                raise_error(ValueError, "Invalid bitflip probability {}."
                                        "".format(probs))
            return len(qubits) * (probs,)

        if isinstance(probs, (tuple, list)):
            if len(probs) != len(qubits):
                raise_error(ValueError, "{} qubits were measured but the given "
                                        "bitflip probability list contains {} "
                                        "values.".format(
                                            len(qubits), len(probs)))
            return tuple(probs)

        if isinstance(probs, dict):
            diff = set(probs.keys()) - set(qubits)
            if diff:
                raise_error(KeyError, "Bitflip map contains {} qubits that are "
                                      "not measured.".format(diff))
            return tuple(probs[q] if q in probs else 0.0 for q in qubits)

        raise_error(TypeError, "Invalid type {} of bitflip map.".format(probs))

    def _get_bitflip_map(self, p: Optional["ProbsType"] = None) -> Dict[int, float]:
        """Creates dictionary with bitflip probabilities."""
        if p is None:
            return {q: 0 for q in self.qubits}
        pt = self._get_bitflip_tuple(self.qubits, p)
        return {q: p for q, p in zip(self.qubits, pt)}

    def has_bitflip_noise(self):
        return sum(self.bitflip_map[0].values()) > 0 or sum(self.bitflip_map[1].values()) > 0

    @staticmethod
    def einsum_string(qubits, nqubits, measuring=False):
        """Generates einsum string for partial trace of density matrices.

        Args:
            qubits (list): Set of qubit ids that are traced out.
            nqubits (int): Total number of qubits in the state.
            measuring (bool): If True non-traced-out indices are multiplied and
                the output has shape (nqubits - len(qubits),).
                If False the output has shape 2 * (nqubits - len(qubits),).

        Returns:
            String to use in einsum for performing partial density of a
            density matrix.
        """
        # TODO: Move this somewhere else (if it is needed at all)
        from qibo.config import EINSUM_CHARS
        if (2 - int(measuring)) * nqubits > len(EINSUM_CHARS): # pragma: no cover
            # case not tested because it requires large instance
            raise_error(NotImplementedError, "Not enough einsum characters.")

        left_in, right_in, left_out, right_out = [], [], [], []
        for i in range(nqubits):
            left_in.append(EINSUM_CHARS[i])
            if i in qubits:
                right_in.append(EINSUM_CHARS[i])
            else:
                left_out.append(EINSUM_CHARS[i])
                if measuring:
                    right_in.append(EINSUM_CHARS[i])
                else:
                    right_in.append(EINSUM_CHARS[i + nqubits])
                    right_out.append(EINSUM_CHARS[i + nqubits])

        left_in, left_out = "".join(left_in), "".join(left_out)
        right_in, right_out = "".join(right_in), "".join(right_out)
        return f"{left_in}{right_in}->{left_out}{right_out}"

    def symbol(self):
        """Returns symbol containing measurement outcomes for ``collapse=True`` gates."""
        return self._symbol

    def add(self, gate: "M"):
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

    @property
    def matrix(self):
        """"""
        raise_error(NotImplementedError, "Measurement gates do not have matrix representation.")