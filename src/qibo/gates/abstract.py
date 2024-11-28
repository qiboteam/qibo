import collections
import json
from typing import List, Sequence, Tuple

import sympy

from qibo.backends import _check_backend
from qibo.config import raise_error

REQUIRED_FIELDS = [
    "name",
    "init_args",
    "init_kwargs",
    "_target_qubits",
    "_control_qubits",
]
REQUIRED_FIELDS_INIT_KWARGS = [
    "theta",
    "phi",
    "lam",
    "phi0",
    "phi1",
    "register_name",
    "collapse",
    "basis",
    "p0",
    "p1",
]


class Gate:
    """The base class for gate implementation.

    All base gates should inherit this class.

    Attributes:
        name (str): Name of the gate.
        draw_label (str): Optional label for drawing the gate in a circuit
            with :meth:`qibo.models.Circuit.draw`.
        is_controlled_by (bool): ``True`` if the gate was created using the
            :meth:`qibo.gates.abstract.Gate.controlled_by` method,
            otherwise ``False``.
        init_args (list): Arguments used to initialize the gate.
        init_kwargs (dict): Arguments used to initialize the gate.
        target_qubits (tuple): Tuple with ids of target qubits.
        control_qubits (tuple): Tuple with ids of control qubits sorted in
            increasing order.
    """

    def __init__(self):
        from qibo import config

        self.name = None
        self.draw_label = None
        self.is_controlled_by = False
        # args for creating gate
        self.init_args = []
        self.init_kwargs = {}

        self.unitary = False
        self._target_qubits = ()
        self._control_qubits = ()
        self._parameters = ()
        config.ALLOW_SWITCHERS = False

        self.symbolic_parameters = {}

        # for distributed circuits
        self.device_gates = set()
        self.original_gate = None

    @property
    def clifford(self):
        """Return boolean value representing if a Gate is Clifford or not."""
        return False

    @property
    def raw(self) -> dict:
        """Serialize to dictionary.

        The values used in the serialization should be compatible with a
        JSON dump (or any other one supporting a minimal set of scalar
        types). Though the specific implementation is up to the specific
        gate.
        """
        encoded = self.__dict__

        encoded_simple = {
            key: value for key, value in encoded.items() if key in REQUIRED_FIELDS
        }

        encoded_simple["init_kwargs"] = {
            key: value
            for key, value in encoded_simple["init_kwargs"].items()
            if key in REQUIRED_FIELDS_INIT_KWARGS
        }

        encoded_simple["_class"] = type(self).__name__

        return encoded_simple

    @staticmethod
    def from_dict(raw: dict):
        """Load from serialization.

        Essentially the counter-part of :meth:`raw`.
        """
        from qibo.gates import gates, measurements

        for mod in (gates, measurements):
            try:
                cls = getattr(mod, raw["_class"])
                break
            except AttributeError:
                # gate not found in given module, try next
                pass
        else:
            raise ValueError(f"Unknown gate {raw['_class']}")

        gate = cls(*raw["init_args"], **raw["init_kwargs"])
        if raw["_class"] == "M":
            if raw["measurement_result"]["samples"] is not None:
                gate.result.register_samples(raw["measurement_result"]["samples"])
            return gate
        try:
            return gate.controlled_by(*raw["_control_qubits"])
        except RuntimeError as e:
            if "controlled" in e.args[0]:
                return gate
            raise e

    def to_json(self):
        """Dump gate to JSON.

        Note:
            Consider using :meth:`raw` directly.
        """
        return json.dumps(self.raw)

    @property
    def target_qubits(self) -> Tuple[int, ...]:
        """Tuple with ids of target qubits."""
        return self._target_qubits

    @property
    def control_qubits(self) -> Tuple[int, ...]:
        """Tuple with ids of control qubits sorted in increasing order."""
        return tuple(sorted(self._control_qubits))

    @property
    def qubits(self) -> Tuple[int, ...]:
        """Tuple with ids of all qubits (control and target) that the gate acts."""
        return self.control_qubits + self.target_qubits

    @property
    def qasm_label(self):
        """String corresponding to OpenQASM operation of the gate."""
        raise_error(
            NotImplementedError,
            f"{self.__class__.__name__} is not supported by OpenQASM",
        )

    def _set_target_qubits(self, qubits: Sequence[int]):
        """Helper method for setting target qubits."""
        self._target_qubits = tuple(qubits)
        if len(self._target_qubits) != len(set(qubits)):
            repeated = self._find_repeated(qubits)
            raise_error(
                ValueError,
                f"Target qubit {repeated} was given twice for gate {self.__class__.__name__}.",
            )

    def _set_control_qubits(self, qubits: Sequence[int]):
        """Helper method for setting control qubits."""
        if len(set(qubits)) != len(qubits):
            repeated = self._find_repeated(qubits)
            raise_error(
                ValueError,
                f"Control qubit {repeated} was given twice for gate {self.__class__.__name__}.",
            )
        self._control_qubits = qubits

    @target_qubits.setter
    def target_qubits(self, qubits: Sequence[int]):
        """Sets target qubits tuple."""
        self._set_target_qubits(qubits)
        self._check_control_target_overlap()

    @control_qubits.setter
    def control_qubits(self, qubits: Sequence[int]):
        """Sets control qubits set."""
        self._set_control_qubits(qubits)
        self._check_control_target_overlap()

    def _set_targets_and_controls(
        self, target_qubits: Sequence[int], control_qubits: Sequence[int]
    ):
        """Sets target and control qubits simultaneously.

        This is used for the reduced qubit updates in the distributed
        circuits because using the individual setters may raise errors
        due to temporary overlap of control and target qubits.
        """
        self._set_target_qubits(target_qubits)
        self._set_control_qubits(control_qubits)
        self._check_control_target_overlap()

    @staticmethod
    def _find_repeated(qubits: Sequence[int]) -> int:
        """Finds the first qubit id that is repeated in a sequence of qubit ids."""
        temp_set = set()
        for qubit in qubits:
            if qubit in temp_set:
                return qubit
            temp_set.add(qubit)

    def _check_control_target_overlap(self):
        """Checks that there are no qubits that are both target and
        controls."""
        control_and_target = self._control_qubits + self._target_qubits
        common = len(set(control_and_target)) != len(control_and_target)
        if common:
            raise_error(
                ValueError,
                f"{set(self._target_qubits) & set(self._control_qubits)}"
                + "qubits are both targets and controls "
                + f"for gate {self.__class__.__name__}.",
            )

    @property
    def parameters(self):
        """Returns a tuple containing the current value of gate's parameters."""
        return self._parameters

    def commutes(self, gate: "Gate") -> bool:
        """Checks if two gates commute.

        Args:
            gate: Gate to check if it commutes with the current gate.

        Returns:
            bool: ``True`` if the gates commute, ``False`` otherwise.
        """
        if isinstance(gate, SpecialGate):  # pragma: no cover
            return False
        t1 = set(self.target_qubits)
        t2 = set(gate.target_qubits)
        a = self.__class__ == gate.__class__ and t1 == t2
        b = not (t1 & set(gate.qubits) or t2 & set(self.qubits))
        return a or b

    def on_qubits(self, qubit_map) -> "Gate":
        """Creates the same gate targeting different qubits.

        Args:
            qubit_map (int): Dictionary mapping original qubit indices to new ones.

        Returns:
            A :class:`qibo.gates.Gate` object of the original gate
            type targeting the given qubits.

        Example:

            .. testcode::

                from qibo import Circuit, gates
                circuit = Circuit(4)
                # Add some CNOT gates
                circuit.add(gates.CNOT(2, 3).on_qubits({2: 2, 3: 3})) # equivalent to gates.CNOT(2, 3)
                circuit.add(gates.CNOT(2, 3).on_qubits({2: 3, 3: 0})) # equivalent to gates.CNOT(3, 0)
                circuit.add(gates.CNOT(2, 3).on_qubits({2: 1, 3: 3})) # equivalent to gates.CNOT(1, 3)
                circuit.add(gates.CNOT(2, 3).on_qubits({2: 2, 3: 1})) # equivalent to gates.CNOT(2, 1)
                circuit.draw()
            .. testoutput::

                0: ───X─────
                1: ───|─o─X─
                2: ─o─|─|─o─
                3: ─X─o─X───
        """
        if self.is_controlled_by:
            targets = (qubit_map.get(q) for q in self.target_qubits)
            controls = (qubit_map.get(q) for q in self.control_qubits)
            gate = self.__class__(*targets, **self.init_kwargs)
            gate = gate.controlled_by(*controls)
        else:
            qubits = (qubit_map.get(q) for q in self.qubits)
            gate = self.__class__(*qubits, **self.init_kwargs)
        return gate

    def _dagger(self) -> "Gate":
        """Helper method for :meth:`qibo.gates.Gate.dagger`."""
        # By default the ``_dagger`` method creates an equivalent gate, assuming
        # that the gate is Hermitian (true for common gates like H or Paulis).
        # If the gate is not Hermitian the ``_dagger`` method should be modified.
        return self.__class__(*self.init_args, **self.init_kwargs)

    def dagger(self) -> "Gate":
        """Returns the dagger (conjugate transpose) of the gate.

        Note that dagger is not persistent for parametrized gates.
        For example, applying a dagger to an :class:`qibo.gates.gates.RX` gate
        will change the sign of its parameter at the time of application.
        However, if the parameter is updated after that, for example using
        :meth:`qibo.models.circuit.Circuit.set_parameters`, then the
        action of dagger will be lost.

        Returns:
            :class:`qibo.gates.Gate`: object representing the dagger of the original gate.
        """
        new_gate = self._dagger()
        new_gate.is_controlled_by = self.is_controlled_by
        new_gate.control_qubits = self.control_qubits
        return new_gate

    def check_controls(func):  # pylint: disable=E0213
        def wrapper(self, *args):
            if self.control_qubits:
                raise_error(
                    RuntimeError,
                    "Cannot use `controlled_by` method "
                    + f"on gate {self} because it is already "
                    + f"controlled by {self.control_qubits}.",
                )
            return func(self, *args)  # pylint: disable=E1102

        return wrapper

    @check_controls
    def controlled_by(self, *qubits: int) -> "Gate":
        """Controls the gate on (arbitrarily many) qubits.

        To see how this method affects the underlying matrix representation of a gate,
        please see the documentation of :meth:`qibo.gates.Gate.matrix`.

        .. note::
            Some gate classes default to another gate class depending on the number of controls
            present. For instance, an :math:`1`-controlled :class:`qibo.gates.X` gate
            will default to a :class:`qibo.gates.CNOT` gate, while a :math:`2`-controlled
            :class:`qibo.gates.X` gate defaults to a :class:`qibo.gates.TOFFOLI` gate.
            Other gates affected by this method are: :class:`qibo.gates.Y`, :class:`qibo.gates.Z`,
            :class:`qibo.gates.RX`, :class:`qibo.gates.RY`, :class:`qibo.gates.RZ`,
            :class:`qibo.gates.U1`, :class:`qibo.gates.U2`, and :class:`qibo.gates.U3`.

        Args:
            *qubits (int): Ids of the qubits that the gate will be controlled on.

        Returns:
            :class:`qibo.gates.Gate`: object in with the corresponding
                gate being controlled in the given qubits.
        """
        if qubits:
            self.is_controlled_by = True
            self.control_qubits = qubits
        return self

    def decompose(self, *free) -> List["Gate"]:
        """Decomposes multi-control gates to gates supported by OpenQASM.

        Decompositions are based on `arXiv:9503016 <https://arxiv.org/abs/quant-ph/9503016>`_.

        Args:
            free: Ids of free qubits to use for the gate decomposition.

        Returns:
            list: gates that have the same effect as applying the original gate.
        """
        # TODO: Implement this method for all gates not supported by OpenQASM.
        # Currently this is implemented only for multi-controlled X gates.
        # If it is used on a different gate it will just return a deep copy
        # of the same gate.
        return [self.__class__(*self.init_args, **self.init_kwargs)]

    def matrix(self, backend=None):
        """Returns the matrix representation of the gate.

        If gate has controlled qubits inserted by :meth:`qibo.gates.Gate.controlled_by`,
        then :meth:`qibo.gates.Gate.matrix` returns the matrix of the original gate.

        .. code-block:: python

            from qibo import gates

            gate = gates.SWAP(3, 4).controlled_by(0, 1, 2)
            print(gate.matrix())

        To return the full matrix that takes the control qubits into account,
        one should use :meth:`qibo.models.Circuit.unitary`, e.g.

        .. code-block:: python

            from qibo import Circuit, gates

            nqubits = 5
            circuit = Circuit(nqubits)
            circuit.add(gates.SWAP(3, 4).controlled_by(0, 1, 2))
            print(circuit.unitary())

        Args:
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend
                to be used in the execution. If ``None``, it uses
                the current backend. Defaults to ``None``.

        Returns:
            ndarray: Matrix representation of gate.

        .. note::
            ``Gate.matrix`` was defined as an atribute in ``qibo`` versions prior to  ``0.2.0``.
            From ``0.2.0`` on, it has been converted into a method and has replaced the ``asmatrix`` method.
        """
        backend = _check_backend(backend)

        return backend.matrix(self)

    def generator_eigenvalue(self):
        """This function returns the eigenvalues of the gate's generator.

        Returns:
            float: eigenvalue of the generator.
        """

        raise_error(
            NotImplementedError,
            f"Generator eigenvalue is not implemented for {self.__class__.__name__}",
        )

    def basis_rotation(self):
        """Transformation required to rotate the basis for measuring the gate."""
        raise_error(
            NotImplementedError,
            f"Basis rotation is not implemented for {self.__class__.__name__}",
        )

    def apply(self, backend, state, nqubits):
        return backend.apply_gate(self, state, nqubits)

    def apply_density_matrix(self, backend, state, nqubits):
        return backend.apply_gate_density_matrix(self, state, nqubits)

    def apply_clifford(self, backend, state, nqubits):
        return backend.apply_gate_clifford(self, state, nqubits)


class SpecialGate(Gate):
    """Abstract class for special gates."""

    def commutes(self, gate):
        return False

    def on_qubits(self, qubit_map):
        raise_error(NotImplementedError, "Cannot use special gates on subroutines.")

    def matrix(self, backend=None):  # pragma: no cover
        raise_error(
            NotImplementedError, "Special gates do not have matrix representation."
        )


class ParametrizedGate(Gate):
    """Base class for parametrized gates.

    Implements the basic functionality of parameter setters and getters.
    """

    def __init__(self, trainable=True):
        super().__init__()
        self.parameter_names = "theta"
        self.nparams = 1
        self.trainable = trainable

    @Gate.parameters.setter
    def parameters(self, x):
        """Updates the values of gate's parameters."""
        if isinstance(self.parameter_names, str):
            nparams = 1
            names = [self.parameter_names]
            if not isinstance(x, collections.abc.Iterable):
                x = [x]
            else:
                # Captures the ``Unitary`` gate case where the given parameter
                # can be an array
                try:
                    if len(x) != 1:  # pragma: no cover
                        x = [x]
                except TypeError:  # tf.Variable case
                    s = tuple(x.shape)
                    if not s or s[0] != 1:
                        x = [x]
        else:
            nparams = len(self.parameter_names)
            names = self.parameter_names

        if not self._parameters:
            params = nparams * [None]
        else:
            params = list(self._parameters)
        if len(x) != nparams:
            raise_error(
                ValueError,
                f"Parametrized gate has {nparams} parameters "
                + f"but {len(x)} update values were given.",
            )
        for i, v in enumerate(x):
            if isinstance(v, sympy.Expr):
                self.symbolic_parameters[i] = v
            params[i] = v
        self._parameters = tuple(params)
        self.init_kwargs.update(
            {n: v for n, v in zip(names, self._parameters) if n in self.init_kwargs}
        )

        # set parameters in device gates
        for gate in self.device_gates:  # pragma: no cover
            gate.parameters = x

    def on_qubits(self, qubit_map):
        gate = super().on_qubits(qubit_map)
        gate.parameters = self.parameters
        return gate

    def substitute_symbols(self):
        params = list(self._parameters)
        for i, param in self.symbolic_parameters.items():
            for symbol in param.free_symbols:
                param = symbol.evaluate(param)
            params[i] = float(param)
        self.parameters = tuple(params)

    def matrix(self, backend=None):
        backend = _check_backend(backend)

        return backend.matrix_parametrized(self)
