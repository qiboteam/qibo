import sympy
from qibo.config import raise_error
from qibo.gates.abstract import Gate
from typing import Dict, Optional, Tuple


class MeasurementResult:

    def __init__(self, qubits):
        self.qubits = qubits
        self.samples = []

    def append(self, shot):
        self.samples.append(shot)


class MeasurementSymbol(sympy.Symbol):
    """``sympy.Symbol`` connected to measurement results.

    Used by :class:`qibo.gates.measurements.M` with ``collapse=True`` to allow
    controlling subsequent gates from the measurement results.
    """
    _counter = 0

    def __new__(cls, *args, **kwargs):
        name = "m{}".format(cls._counter)
        cls._counter += 1
        return super().__new__(cls=cls, name=name)

    def __init__(self, index, result):
        self.index = index
        self.result = result

    def __getstate__(self):
        return {
            "index": self.index,
            "result": self.result,
            "name": self.name
        }

    def __setstate__(self, data):
        self.index = data.get("index")
        self.result = data.get("result")
        self.name = data.get("name")

    def outcome(self):
        return self.result.samples[-1][self.index]

    def evaluate(self, expr):
        """Substitutes the symbol's value in the given expression.

        Args:
            expr (sympy.Expr): Sympy expression that involves the current
                measurement symbol.
        """
        return expr.subs(self, self.outcome())


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
        self.target_qubits = tuple(q)
        self.register_name = register_name
        self.collapse = collapse
        self.result = None

        self.init_args = q
        self.init_kwargs = {"register_name": register_name,
                            "collapse": collapse,
                            "p0": p0, "p1": p1}
        if collapse:
            if p0 is not None or p1 is not None:
                raise_error(NotImplementedError, "Bitflip measurement noise is not "
                                                "available when collapsing.")
            self.result = MeasurementResult(self.target_qubits)

        if p1 is None: p1 = p0
        if p0 is None: p0 = p1
        self.bitflip_map = (self._get_bitflip_map(p0),
                            self._get_bitflip_map(p1))

    def get_symbols(self):
        symbols = []
        for i, q in enumerate(self.target_qubits):
            symbols.append(MeasurementSymbol(i, self.result))
        if len(self.target_qubits) > 1:
            return symbols
        else:
            return symbols[0]

    @staticmethod
    def _get_bitflip_tuple(qubits: Tuple[int], probs: "ProbsType"
                           ) -> Tuple[float]:
        if isinstance(probs, float):
            if probs < 0 or probs > 1:  # pragma: no cover
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

    @property
    def matrix(self):
        """"""
        raise_error(NotImplementedError, "Measurement gates do not have matrix representation.")

    def apply(self, backend, state, nqubits):
        qubits = sorted(self.target_qubits)
        # measure and get result
        probs = backend.calculate_probabilities(state, qubits, nqubits)
        shot = backend.sample_shots(probs, 1)
        # update the gate's result with the measurement outcome
        binshot = backend.samples_to_binary(shot, len(qubits))[0]
        self.result.backend = backend
        self.result.append(binshot)
        # collapse state
        return backend.collapse_state(state, qubits, shot, nqubits)

    def apply_density_matrix(self, backend, state, nqubits):
        qubits = sorted(self.target_qubits)
        # measure and get result
        probs = backend.calculate_probabilities_density_matrix(state, qubits, nqubits)
        shot = backend.sample_shots(probs, 1)
        binshot = backend.samples_to_binary(shot, len(qubits))[0]
        # update the gate's result with the measurement outcome
        self.result.backend = backend
        self.result.append(binshot)
        # collapse state
        return backend.collapse_density_matrix(state, qubits, shot, nqubits)
