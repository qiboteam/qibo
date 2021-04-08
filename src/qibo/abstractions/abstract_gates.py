# -*- coding: utf-8 -*-
import collections
import sympy
from abc import ABC, abstractmethod
from qibo import get_device, config
from qibo.config import raise_error
from collections.abc import Iterable
from typing import List, Sequence, Tuple


class Gate:
    """The base class for gate implementation.

    All base gates should inherit this class.

    Attributes:
        name (str): Name of the gate.
        is_controlled_by (bool): ``True`` if the gate was created using the
            :meth:`qibo.abstractions.abstract_gates.Gate.controlled_by` method,
            otherwise ``False``.
        init_args (list): Arguments used to initialize the gate.
        init_kwargs (dict): Arguments used to initialize the gate.
        target_qubits (tuple): Tuple with ids of target qubits.
        control_qubits (tuple): Tuple with ids of control qubits sorted in
            increasing order.
        nqubits (int): Number of qubits that this gate acts on.
        nstates (int): Size of state vectors that this gate acts on.
        density_matrix (bool): Controls if the gate acts on state vectors or
            density matrices.
    """
    from qibo.abstractions import gates as module

    def __init__(self):
        self.name = None
        self.is_controlled_by = False
        # args for creating gate
        self.init_args = []
        self.init_kwargs = {}

        self._target_qubits = tuple()
        self._control_qubits = set()

        self._nqubits = None
        self._nstates = None
        config.ALLOW_SWITCHERS = False

        self.is_prepared = False
        self.well_defined = True
        # Keeps track of whether parametrized gates are well-defined
        # (parameter value is known during circuit creation) or if they are
        # measurement dependent so the parameter value is determined during
        # execution

        # Using density matrices or state vectors
        self._density_matrix = False
        self._active_call = "state_vector_call"

    @property
    def target_qubits(self) -> Tuple[int]:
        """Tuple with ids of target qubits."""
        return self._target_qubits

    @property
    def control_qubits(self) -> Tuple[int]:
        """Tuple with ids of control qubits sorted in increasing order."""
        return tuple(sorted(self._control_qubits))

    @property
    def qubits(self) -> Tuple[int]:
        """Tuple with ids of all qubits (control and target) that the gate acts."""
        return self.control_qubits + self.target_qubits

    def _set_target_qubits(self, qubits: Sequence[int]):
        """Helper method for setting target qubits."""
        self._target_qubits = tuple(qubits)
        if len(self._target_qubits) != len(set(qubits)):
            repeated = self._find_repeated(qubits)
            raise_error(ValueError, "Target qubit {} was given twice for gate {}."
                                    "".format(repeated, self.name))

    def _set_control_qubits(self, qubits: Sequence[int]):
        """Helper method for setting control qubits."""
        self._control_qubits = set(qubits)
        if len(self._control_qubits) != len(qubits):
            repeated = self._find_repeated(qubits)
            raise_error(ValueError, "Control qubit {} was given twice for gate {}."
                                    "".format(repeated, self.name))

    @target_qubits.setter
    def target_qubits(self, qubits: Sequence[int]):
        """Sets control qubits tuple."""
        self._set_target_qubits(qubits)
        self._check_control_target_overlap()

    @control_qubits.setter
    def control_qubits(self, qubits: Sequence[int]):
        """Sets control qubits set."""
        self._set_control_qubits(qubits)
        self._check_control_target_overlap()

    def set_targets_and_controls(self, target_qubits: Sequence[int],
                                 control_qubits: Sequence[int]):
        """Sets target and control qubits simultaneously.

        This is used for the reduced qubit updates in the distributed circuits
        because using the individual setters may raise errors due to temporary
        overlap of control and target qubits.
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
        """Checks that there are no qubits that are both target and controls."""
        common = set(self._target_qubits) & self._control_qubits
        if common:
            raise_error(ValueError, "{} qubits are both targets and controls for "
                                    "gate {}.".format(common, self.name))

    @property
    def nqubits(self) -> int:
        """Number of qubits that this gate acts on."""
        if self._nqubits is None:
            raise_error(ValueError, "Accessing number of qubits for gate {} but "
                                    "this is not yet set.".format(self))
        return self._nqubits

    @property
    def nstates(self) -> int:
        """Size of the state vectors that this gate acts on."""
        if self._nstates is None:
            raise_error(ValueError, "Accessing number of qubits for gate {} but "
                                    "this is not yet set.".format(self))
        return self._nstates

    @nqubits.setter
    def nqubits(self, n: int):
        """Sets the total number of qubits that this gate acts on.

        This setter is used by `circuit.add` if the gate is added in a circuit
        or during `__call__` if the gate is called directly on a state.
        The user is not supposed to set `nqubits` by hand.
        """
        if self._nqubits is not None and n != self.nqubits:
            raise_error(ValueError, "Cannot set gate number of qubits to {} "
                                    "because it is already set to {}."
                                    "".format(n, self.nqubits))
        self._nqubits = n
        self._nstates = 2**n

    @property
    def density_matrix(self) -> bool:
        """Controls if the gate acts on state vectors or density matrices."""
        return self._density_matrix

    @density_matrix.setter
    def density_matrix(self, x: bool):
        """Density matrix flag switcher."""
        if self.is_prepared:
            raise_error(RuntimeError,
                        "Density matrix mode cannot be switched after "
                        "preparing the gate for execution.")
        self._density_matrix = x
        if x:
            self._active_call = "density_matrix_call"
        else:
            self._active_call = "state_vector_call"

    def commutes(self, gate: "Gate") -> bool:
        """Checks if two gates commute.

        Args:
            gate: Gate to check if it commutes with the current gate.

        Returns:
            ``True`` if the gates commute, otherwise ``False``.
        """
        if isinstance(gate, SpecialGate):
            return False
        t1 = set(self.target_qubits)
        t2 = set(gate.target_qubits)
        a = self.__class__ == gate.__class__ and t1 == t2
        b = not (t1 & set(gate.qubits) or t2 & set(self.qubits))
        return a or b

    def on_qubits(self, *q) -> "Gate":
        """Creates the same gate targeting different qubits.

        Args:
            q (int): Qubit index (or indeces) that the new gate should act on.
        """
        if self.is_controlled_by:
            targets = (q[i] for i in self.target_qubits)
            controls = (q[i] for i in self.control_qubits)
            gate = self.__class__(*targets, **self.init_kwargs)
            gate = gate.controlled_by(*controls)
        else:
            qubits = (q[i] for i in self.qubits)
            gate = self.__class__(*qubits, **self.init_kwargs)
        return gate

    def _dagger(self) -> "Gate":
        """Helper method for :meth:`qibo.abstractions.gates.Gate.dagger`."""
        # By default the ``_dagger`` method creates an equivalent gate, assuming
        # that the gate is Hermitian (true for common gates like H or Paulis).
        # If the gate is not Hermitian the ``_dagger`` method should be modified.
        return self.__class__(*self.init_args, **self.init_kwargs)

    def dagger(self) -> "Gate":
        """Returns the dagger (conjugate transpose) of the gate.

        Returns:
            A :class:`qibo.abstractions.gates.Gate` object representing the dagger of
            the original gate.
        """
        new_gate = self._dagger()
        new_gate.is_controlled_by = self.is_controlled_by
        new_gate.control_qubits = self.control_qubits
        return new_gate

    def check_controls(func): # pylint: disable=E0213
        def wrapper(self, *args):
            if self.control_qubits:
                raise_error(RuntimeError, "Cannot use `controlled_by` method "
                                          "on gate {} because it is already "
                                          "controlled by {}."
                                          "".format(self, self.control_qubits))
            if self._nqubits is not None:
                raise_error(RuntimeError, "Cannot use controlled_by on a gate "
                                          "for which the number of qubits is "
                                          "set.")
            return func(self, *args) # pylint: disable=E1102
        return wrapper

    @check_controls
    def controlled_by(self, *qubits: int) -> "Gate":
        """Controls the gate on (arbitrarily many) qubits.

        Args:
            *qubits (int): Ids of the qubits that the gate will be controlled on.

        Returns:
            A :class:`qibo.abstractions.gates.Gate` object in with the corresponding
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
            List with gates that have the same effect as applying the original gate.
        """
        # TODO: Implement this method for all gates not supported by OpenQASM.
        # Currently this is implemented only for multi-controlled X gates.
        # If it is used on a different gate it will just return a deep copy
        # of the same gate.
        return [self.__class__(*self.init_args, **self.init_kwargs)]


class SpecialGate(Gate):
    """Abstract class for special gates.

    Current special gates are :class:`qibo.abstractions.gates.CallbackGate` and
    :class:`qibo.abstractions.gates.Flatten`.
    """

    def commutes(self, gate):
        return False

    def on_qubits(self, *q):
        raise_error(NotImplementedError,
                    "Cannot use special gates on subroutines.")


class Channel(Gate):
    """Abstract class for channels."""

    def __init__(self):
        super().__init__()
        self.gates = tuple()
        # create inversion gates to restore the original state vector
        # because of the in-place updates used in custom operators
        self._inverse_gates = None

    @property
    def inverse_gates(self):
        if self._inverse_gates is None:
            self._inverse_gates = self.calculate_inverse_gates()
            for gate in self._inverse_gates:
                if gate is not None:
                    if self._nqubits is not None:
                        gate.nqubits = self._nqubits
                    gate.density_matrix = self.density_matrix
        return self._inverse_gates

    @abstractmethod
    def calculate_inverse_gates(self): # pragma: no cover
        raise_error(NotImplementedError)

    @Gate.nqubits.setter
    def nqubits(self, n: int):
        Gate.nqubits.fset(self, n) # pylint: disable=no-member
        for gate in self.gates:
            gate.nqubits = n
        if self._inverse_gates is not None:
            for gate in self._inverse_gates:
                if gate is not None:
                    gate.nqubits = n

    @Gate.density_matrix.setter
    def density_matrix(self, x):
        Gate.density_matrix.fset(self, x) # pylint: disable=no-member
        for gate in self.gates:
            gate.density_matrix = x
        if self._inverse_gates is not None:
            for gate in self._inverse_gates:
                if gate is not None:
                    gate.density_matrix = x

    def controlled_by(self, *q):
        """"""
        raise_error(ValueError, "Noise channel cannot be controlled on qubits.")

    def on_qubits(self, *q): # pragma: no cover
        # future TODO
        raise_error(NotImplementedError, "`on_qubits` method is not available "
                                         "for the `GeneralChannel` gate.")


class ParametrizedGate(Gate):
    """Base class for parametrized gates.

    Implements the basic functionality of parameter setters and getters.
    """

    def __init__(self, trainable=True):
        super(ParametrizedGate, self).__init__()
        self.parameter_names = "theta"
        self.nparams = 1
        self.trainable = trainable
        self._parameters = []
        self.symbolic_parameters = {}

    @property
    def parameters(self):
        """Returns a tuple containing the current value of gate's parameters."""
        if isinstance(self.parameter_names, str):
            return self._parameters[0]
        return tuple(self._parameters)

    @parameters.setter
    def parameters(self, x):
        """Updates the values of gate's parameters."""
        if isinstance(self.parameter_names, str):
            nparams = 1
            if not isinstance(x, collections.abc.Iterable):
                x = [x]
            else:
                # Captures the ``Unitary`` gate case where the given parameter
                # can be an array
                try:
                    if len(x) != 1:
                        x = [x]
                except TypeError: # tf.Variable case
                    s = tuple(x.shape)
                    if not s or s[0] != 1:
                        x = [x]
        else:
            nparams = len(self.parameter_names)

        if not self._parameters:
            self._parameters = nparams * [None]
        if len(x) != nparams:
            raise_error(ValueError, "Parametrized gate has {} parameters "
                                    "but {} update values were given."
                                    "".format(nparams, len(x)))
        for i, v in enumerate(x):
            if isinstance(v, sympy.Expr):
                self.well_defined = False
                self.symbolic_parameters[i] = v
            self._parameters[i] = v

        # This part uses ``BackendGate`` attributes (see below), assuming
        # that the gate was initialized using a calculation backend.
        # I could not find a cleaner way to write this so that the
        # ``circuit.set_parameters`` method works properly.
        # pylint: disable=E1101
        if isinstance(self, BaseBackendGate):
            self._unitary = None
            self._matrix = None
            for devgate in self.device_gates:
                devgate.parameters = x

    def substitute_symbols(self):
        params = list(self._parameters)
        for i, param in self.symbolic_parameters.items():
            for symbol in param.free_symbols:
                param = symbol.evaluate(param)
            params[i] = float(param)
        self.parameters = params


class BaseBackendGate(Gate, ABC):
    """Abstract class for gate objects that can be used in calculations.

    Attributes:
        unitary: Unitary matrix representation of the gate in the computational
            basis.
        is_prepared: ``True`` if the gate is prepared for action to states.
            A gate is prepared when its matrix and/or other tensors required
            in the computation are calculated.
            See :meth:`qibo.abstractions.abstract_gates.BackendGate.prepare` for more
            details.
            Note that gate preparation is triggered automatically when a gate
            is added to a circuit or when it acts on a state.
        device: Hardware device to use in order to simulate this gate.
        density_matrix: ``True`` if the gate will act on density matrices,
            ``False`` if the gate will act on state vectors.
    """
    module = None

    def __init__(self):
        Gate.__init__(self)
        self._matrix = None
        self._unitary = None
        self._cache = None
        # Cast gate matrices to the proper device
        self.device = get_device()
        # Reference to copies of this gate that are casted in devices when
        # a distributed circuit is used
        self.device_gates = set()
        self.original_gate = None

    @property
    def unitary(self):
        """Unitary matrix representing the gate in the computational basis."""
        if len(self.qubits) > 2:
            raise_error(NotImplementedError, "Cannot calculate unitary matrix for "
                                             "gates that target more than two qubits.")
        if self._unitary is None:
            self._unitary = self.construct_unitary()
        if self.is_controlled_by and tuple(self._unitary.shape) == (2, 2):
            self._unitary = self.control_unitary(self._unitary)
        return self._unitary

    def __matmul__(self, other: "Gate") -> "Gate":
        """Gate multiplication."""
        if self.qubits != other.qubits:
            raise_error(NotImplementedError, "Cannot multiply gates that target "
                                             "different qubits.")
        if self.__class__.__name__ == other.__class__.__name__:
            square_identity = {"H", "X", "Y", "Z", "CNOT", "CZ", "SWAP"}
            if self.__class__.__name__ in square_identity:
                from qibo.gates import I
                return I(*self.qubits)
        return self.module.Unitary(self.unitary @ other.unitary, *self.qubits)

    def __rmatmul__(self, other): # pragma: no cover
        # always falls back to left ``__matmul__``
        return self.__matmul__(other)

    @staticmethod
    @abstractmethod
    def control_unitary(unitary): # pragma: no cover
        """Updates the unitary matrix of the gate if it is controlled."""
        raise_error(NotImplementedError)

    @abstractmethod
    def construct_unitary(self): # pragma: no cover
        """Constructs the gate's unitary matrix."""
        return raise_error(NotImplementedError)

    @property
    @abstractmethod
    def cache(self): # pragma: no cover
        raise_error(NotImplementedError)

    @abstractmethod
    def set_nqubits(self, state): # pragma: no cover
        """Sets ``gate.nqubits`` and prepares gates for application to states.

        This method is used only when gates are called directly on states
        without being a part of circuit. If a gate is added in a circuit it
        is automatically prepared and this method is not required.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def state_vector_call(self, state): # pragma: no cover
        """Applies the gate on a state vector."""
        raise_error(NotImplementedError)

    @abstractmethod
    def density_matrix_call(self, state): # pragma: no cover
        """Applies the gate on a density matrix."""
        raise_error(NotImplementedError)

    def __call__(self, state):
        """Applies the gate on a state.

        Falls back to a state vector or density matrix call according to the
        current value of the ``gate.density_matrix`` flag.
        It automatically prepares the gate if it is not already prepared.
        """
        if not self.is_prepared:
            self.set_nqubits(state)
        if not self.well_defined:
            self.substitute_symbols() # pylint: disable=E1101
            # method available only for parametrized gates
        return getattr(self, self._active_call)(state)
