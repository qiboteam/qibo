# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
"""
Gates use ``einsum`` to apply gates to state vectors. The einsum string that
specifies the contraction indices is created and cached when a gate is created
so that it is not recalculated every time the gate is called on a state. This
functionality is implemented in :class:`qibo.core.einsum.DefaultEinsum`.

Due to an `issue <https://github.com/tensorflow/tensorflow/issues/37307>`_
with automatic differentiation and complex numbers in ``einsum``, we have
implemented an alternative calculation backend based on ``matmul`` in
:class:`qibo.core.einsum.MatmulEinsum`. Note that this is slower than
the default ``einsum`` on GPU but slightly faster on CPU.

The user can switch the default einsum used by the gates by changing the
``einsum`` variable in `config.py`. It is recommended to use the default unless
automatic differentiation is required. For the latter case, we refer to our
examples.
"""
from abc import ABC, abstractmethod
from qibo.config import raise_error
from typing import Dict, List, Optional, Sequence


class BaseCache:
    """Base cache object for einsum backends defined in `einsum.py`.

    ``circuit.calculation_cache`` is an object of this class.

    ``self.vector`` returns the cache elements required for state vector
    calculations.
    ``self.left``, ``self.right``, ``self.left0`` and ``self.right0`` return the cache
    elements required for density matrix calculations.
    """

    def __init__(self, nqubits, ncontrol: Optional[int] = None):
        self.nqubits = nqubits
        self.ncontrol = ncontrol
        # Cache for state vectors
        self._vector = None
        # Cache for density matrices
        self._left = None
        self._right = None
        self._left0 = None
        self._right0 = None

    @property
    def vector(self):
        if self._vector is None: # pragma: no cover
            # abstract method
            raise_error(NotImplementedError, "Vector cache should be defined in __init__.")
        return self._vector

    @property
    def left(self):
        if self._left is None: # pragma: no cover
            self._calculate_density_matrix()
        return self._left

    @property
    def right(self):
        if self._right is None:
            self._calculate_density_matrix()
        return self._right

    @property
    def left0(self):
        if self._left0 is None: # pragma: no cover
            self._calculate_density_matrix_controlled()
        return self._left0

    @property
    def right0(self):
        if self._right0 is None:
            self._calculate_density_matrix_controlled()
        return self._right0

    def cast_shapes(self, cast_func):
        pass

    @abstractmethod
    def _calculate_density_matrix(self): # pragma: no cover
        """Calculates `left` and `right` elements."""
        raise_error(NotImplementedError)

    @abstractmethod
    def _calculate_density_matrix_controlled(self): # pragma: no cover
        """Calculates `left0` and `right0` elements."""
        raise_error(NotImplementedError)


class DefaultEinsumCache(BaseCache):
    """Cache object required by the :class:`qibo.core.einsum.DefaultEinsum` backend.

    The ``vector``, ``left``, ``right``, ``left0``, ``right0`` properties are
    strings that hold the einsum indices.

    Args:
        qubits (list): List with the qubit indices that the gate is applied to.
        nqubits (int): Total number of qubits in the circuit / state vector.
        ncontrol (int): Number of control qubits for `controlled_by` gates.
    """
    from qibo.config import EINSUM_CHARS as _chars

    def __init__(self, qubits: Sequence[int], nqubits: int,
                 ncontrol: Optional[int] = None):
        super(DefaultEinsumCache, self).__init__(nqubits, ncontrol)

        if nqubits + len(qubits) > len(self._chars): # pragma: no cover
            raise_error(NotImplementedError, "Not enough einsum characters.")

        input_state = list(self._chars[:nqubits])
        output_state = input_state[:]
        gate_chars = list(self._chars[nqubits : nqubits + len(qubits)])

        for i, q in enumerate(qubits):
            gate_chars.append(input_state[q])
            output_state[q] = gate_chars[i]

        self.input = "".join(input_state)
        self.output = "".join(output_state)
        self.gate = "".join(gate_chars)
        self.rest = self._chars[nqubits + len(qubits):]

        self._vector = f"{self.input},{self.gate}->{self.output}"

    def _calculate_density_matrix(self):
        if self.nqubits > len(self.rest): # pragma: no cover
            raise_error(NotImplementedError, "Not enough einsum characters.")

        rest = self.rest[:self.nqubits]
        self._left = f"{self.input}{rest},{self.gate}->{self.output}{rest}"
        self._right = f"{rest}{self.input},{self.gate}->{rest}{self.output}"

    def _calculate_density_matrix_controlled(self):
        if self.nqubits + 1 > len(self.rest): # pragma: no cover
            raise_error(NotImplementedError, "Not enough einsum characters.")
        rest, c = self.rest[:self.nqubits], self.rest[self.nqubits]
        self._left0 = f"{c}{self.input}{rest},{self.gate}->{c}{self.output}{rest}"
        self._right0 = f"{c}{rest}{self.input},{self.gate}->{c}{rest}{self.output}"


class MatmulEinsumCache(BaseCache):
    """Cache object required by the :class:`qibo.core.einsum.MatmulEinsum` backend.

    The ``vector``, ``left``, ``right``, ``left0``, ``right0`` properties are dictionaries
    that hold the following keys:

    * ``ids``: Indices for the transposition before matmul.
    * ``inverse_ids``: Indices for the transposition after matmul.
    * ``shapes``: Tuple with four shapes that are required for ``tf.reshape`` in the ``__call__`` method of :class:`qibo.core.einsum.MatmulEinsum`.

    Args:
        qubits (list): List with the qubit indices that the gate is applied to.
        nqubits (int): Total number of qubits in the circuit / state vector.
        ncontrol (int): Number of control qubits for `controlled_by` gates.
    """

    def __init__(self, qubits: Sequence[int], nqubits: int,
                 ncontrol: Optional[int] = None):
        super(MatmulEinsumCache, self).__init__(nqubits, ncontrol)
        self.ntargets = len(qubits)
        self.nrest = nqubits - self.ntargets
        self.nstates = 2 ** nqubits

        last_index = 0
        target_ids, rest_ids = {}, []
        self.shape = []
        for q in sorted(qubits):
            if q > last_index:
                self.shape.append(2 ** (q - last_index))
                rest_ids.append(len(self.shape) - 1)
            self.shape.append(2)
            target_ids[q] = len(self.shape) - 1
            last_index = q + 1
        if last_index < self.nqubits:
            self.shape.append(2 ** (self.nqubits - last_index))
            rest_ids.append(len(self.shape) - 1)

        self.ids = [target_ids[q] for q in qubits] + rest_ids
        self.transposed_shape = []
        self.inverse_ids = len(self.ids) * [0]
        for i, r in enumerate(self.ids):
            self.inverse_ids[r] = i
            self.transposed_shape.append(self.shape[r])

        self.shape = tuple(self.shape)
        self.transposed_shape = tuple(self.transposed_shape)
        self._vector = {"ids": self.ids, "inverse_ids": self.inverse_ids,
                        "shapes": (self.shape,
                                   (2 ** self.ntargets, 2 ** self.nrest),
                                   self.transposed_shape,
                                   self.nqubits * (2,)),
                        "conjugate": False}

    def cast_shapes(self, cast_func):
        for attr in ["_vector", "_left", "_right", "_left0", "_right0"]:
            d = getattr(self, attr)
            if d is not None:
                d["shapes"] = tuple(cast_func(s) for s in d["shapes"])

    def _calculate_density_matrix(self):
        self._left = {"ids": self.ids + [len(self.ids)],
                      "inverse_ids": self.inverse_ids + [len(self.ids)],
                      "shapes": (self.shape + (self.nstates,),
                                 (2 ** self.ntargets, (2 ** self.nrest) * self.nstates),
                                 self.transposed_shape + (self.nstates,),
                                 2 * self.nqubits * (2,)),
                      "conjugate": False}

        self._right = dict(self._left)
        self._right["inverse_ids"] = [len(self.ids)] + self.inverse_ids
        self._right["conjugate"] = True

    def _calculate_density_matrix_controlled(self): # pragma: no cover
        # `MatmulEinsum` falls back to `DefaultEinsum` if `controlled_by`
        # and density matrices are used simultaneously due to an error
        raise_error(NotImplementedError,
                    "MatmulEinsum backend is not implemented when multicontrol "
                    "gates are used on density matrices.")


class ControlCache:
    """Helper tools for `controlled_by` gates.

    This class contains:

    * an `order` that is used to transpose `state` so that control legs are moved in the front
    * a `targets` list which is equivalent to the `target_qubits` tuple but each index is reduced by the amount of control qubits that preceed it.

    This method is called by the `nqubits` setter so that the loop runs
    once per gate (and not every time the gate is called).
    """

    def __init__(self, gate):
        self.ncontrol = len(gate.control_qubits)
        self._order, self.targets = self.calculate(gate)
        # Calculate the reverse order for transposing the state legs so that
        # control qubits are back to their original positions
        self._reverse = self.revert(self._order)

        self._order_dm = None
        self._reverse_dm = None

    def order(self, is_density_matrix: bool = False):
        if not is_density_matrix:
            return self._order

        if self._order_dm is None:
            self.calculate_dm()
        return self._order_dm

    def reverse(self, is_density_matrix: bool = False):
        if not is_density_matrix:
            return self._reverse

        if self._reverse_dm is None: # pragma: no cover
            self.calculate_dm()
        return self._reverse_dm

    @staticmethod
    def calculate(gate):
        loop_start = 0
        order = list(gate.control_qubits)
        targets = list(gate.target_qubits)
        for control in gate.control_qubits:
            for i in range(loop_start, control):
                order.append(i)
            loop_start = control + 1

            for i, t in enumerate(gate.target_qubits):
                if t > control:
                    targets[i] -= 1
        for i in range(loop_start, gate.nqubits):
            order.append(i)

        return order, targets

    def calculate_dm(self):
        additional_order = [x + len(self._order) for x in self._order]
        self._order_dm = (self._order[:self.ncontrol] +
                          list(additional_order[:self.ncontrol]) +
                          self._order[self.ncontrol:] +
                          list(additional_order[self.ncontrol:]))
        self._reverse_dm = self.revert(self._order_dm)

    @staticmethod
    def revert(transpose_order) -> List[int]:
        reverse_order = len(transpose_order) * [0]
        for i, r in enumerate(transpose_order):
            reverse_order[r] = i
        return reverse_order
