import collections
from typing import List, Sequence, Tuple

import sympy

from qibo.config import raise_error


class Gate:
    """The base class for gate implementation.

    All base gates should inherit this class.
    """

    def __init__(self):
        """
        Attributes:
            name (str): Name of the gate.
            draw_label (str): Optional label for drawing the gate in a circuit
                with :func:`qibo.models.Circuit.draw`.
            is_controlled_by (bool): ``True`` if the gate was created using the
                :meth:`qibo.gates.abstract.Gate.controlled_by` method,
                otherwise ``False``.
            init_args (list): Arguments used to initialize the gate.
            init_kwargs (dict): Arguments used to initialize the gate.
            target_qubits (tuple): Tuple with ids of target qubits.
            control_qubits (tuple): Tuple with ids of control qubits sorted in
                increasing order.
        """
        from qibo import config

        self.name = None
        self.draw_label = None
        self.is_controlled_by = False
        # args for creating gate
        self.init_args = []
        self.init_kwargs = {}

        self.clifford = False
        self.unitary = False
        self._target_qubits = tuple()
        self._control_qubits = set()
        self._parameters = tuple()
        config.ALLOW_SWITCHERS = False

        self.symbolic_parameters = {}

        # for distributed circuits
        self.device_gates = set()
        self.original_gate = None

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
        self._control_qubits = set(qubits)
        if len(self._control_qubits) != len(qubits):
            repeated = self._find_repeated(qubits)
            raise_error(
                ValueError,
                f"Control qubit {repeated} was given twice for gate {self.__class__.__name__}.",
            )

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
            raise_error(
                ValueError,
                f"{common} qubits are both targets and controls "
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
            ``True`` if the gates commute, otherwise ``False``.
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

                from qibo import models, gates
                c = models.Circuit(4)
                # Add some CNOT gates
                c.add(gates.CNOT(2, 3).on_qubits({2: 2, 3: 3})) # equivalent to gates.CNOT(2, 3)
                c.add(gates.CNOT(2, 3).on_qubits({2: 3, 3: 0})) # equivalent to gates.CNOT(3, 0)
                c.add(gates.CNOT(2, 3).on_qubits({2: 1, 3: 3})) # equivalent to gates.CNOT(1, 3)
                c.add(gates.CNOT(2, 3).on_qubits({2: 2, 3: 1})) # equivalent to gates.CNOT(2, 1)
                print(c.draw())
            .. testoutput::

                q0: ───X─────
                q1: ───|─o─X─
                q2: ─o─|─|─o─
                q3: ─X─o─X───
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
            A :class:`qibo.gates.Gate` object representing the dagger of
            the original gate.
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

        Args:
            *qubits (int): Ids of the qubits that the gate will be controlled on.

        Returns:
            A :class:`qibo.gates.Gate` object in with the corresponding
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

    def asmatrix(self, backend):
        return backend.asmatrix(self)

    def generator_eigenvalue(self):
        """
        This function returns the eigenvalues of the gate's generator.

        Returns:
            np.float generator's eigenvalue or raise an error if not implemented.
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

    @property
    def matrix(self):
        from qibo.backends import GlobalBackend

        backend = GlobalBackend()
        return self.asmatrix(backend)

    def apply(self, backend, state, nqubits):
        return backend.apply_gate(self, state, nqubits)

    def apply_density_matrix(self, backend, state, nqubits):
        return backend.apply_gate_density_matrix(self, state, nqubits)


class SpecialGate(Gate):
    """Abstract class for special gates."""

    def commutes(self, gate):
        return False

    def on_qubits(self, qubit_map):
        raise_error(NotImplementedError, "Cannot use special gates on subroutines.")

    def asmatrix(self, backend):  # pragma: no cover
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

    def asmatrix(self, backend):
        return backend.asmatrix_parametrized(self)
