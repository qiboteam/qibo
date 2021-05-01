# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
"""
Gates use ``einsum`` to apply gates to state vectors. The einsum string that
specifies the contraction indices is created and cached when a gate is created
so that it is not recalculated every time the gate is called on a state. This
functionality is implemented in :class:`qibo.backends.numpy.NumpyBackend`.
"""
from qibo.config import raise_error


class EinsumCache:
    """Cache object required to apply gates using ``einsum``.

    ``circuit.calculation_cache`` is an object of this class.

    ``self.vector`` returns the cache elements required for state vector
    calculations.
    ``self.left``, ``self.right``, ``self.left0`` and ``self.right0`` return the cache
    elements required for density matrix calculations.

    Args:
        qubits (list): List with the qubit indices that the gate is applied to.
        nqubits (int): Total number of qubits in the circuit / state vector.
        ncontrol (int): Number of control qubits for `controlled_by` gates.
    """
    from qibo.config import EINSUM_CHARS as _chars

    def __init__(self, qubits, nqubits, ncontrol=None):
        self.nqubits = nqubits
        self.ncontrol = ncontrol
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

        # Cache for state vectors
        self._vector = f"{self.input},{self.gate}->{self.output}"

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

    def _calculate_density_matrix(self):
        """Calculates `left` and `right` elements."""
        if self.nqubits > len(self.rest): # pragma: no cover
            raise_error(NotImplementedError, "Not enough einsum characters.")

        rest = self.rest[:self.nqubits]
        self._left = f"{self.input}{rest},{self.gate}->{self.output}{rest}"
        self._right = f"{rest}{self.input},{self.gate}->{rest}{self.output}"

    def _calculate_density_matrix_controlled(self):
        """Calculates `left0` and `right0` elements."""
        if self.nqubits + 1 > len(self.rest): # pragma: no cover
            raise_error(NotImplementedError, "Not enough einsum characters.")
        rest, c = self.rest[:self.nqubits], self.rest[self.nqubits]
        self._left0 = f"{c}{self.input}{rest},{self.gate}->{c}{self.output}{rest}"
        self._right0 = f"{c}{rest}{self.input},{self.gate}->{c}{rest}{self.output}"


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
    def revert(transpose_order):
        reverse_order = len(transpose_order) * [0]
        for i, r in enumerate(transpose_order):
            reverse_order[r] = i
        return reverse_order
