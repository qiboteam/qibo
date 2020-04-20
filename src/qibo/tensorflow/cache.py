from qibo.base import gates as base_gates
from typing import List, Optional, Sequence


class BaseCache:

    def __init__(self):
        # Cache for state vectors
        self._vector = None
        # Cache for density matrices
        self._left = None
        self._right = None
        self._left0 = None
        self._right0 = None

    @property
    def vector(self):
        if self._vector is None:
            self._calculate_state_vector()
        return self._vector

    @property
    def left(self):
        if self._left is None:
            self._calculate_density_matrix()
        return self._left

    @property
    def right(self):
        if self._right is None:
            self._calculate_density_matrix()
        return self._right

    @property
    def left0(self):
        if self._left0 is None:
            self._calculate_density_matrix(is_controlled_by=True)
        return self._left0

    @property
    def right0(self):
        if self._right0 is None:
            self._calculate_density_matrix(is_controlled_by=True)
        return self._right0

    def _calculate_state_vector(self):
        raise NotImplementedError

    def _calculate_density_matrix(self, is_controlled_by: bool = False):
        raise NotImplementedError


class DefaultEinsumCache(BaseCache):

    _chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, qubits: Sequence[int], nqubits: int,
                 ncontrol: Optional[int] = None):
      """Creates index string for `tf.einsum`.

      Args:
          qubits (list): List with the qubit indices that the gate is applied to.
          nqubits (int): Total number of qubits in the circuit / state vector.

      Returns:
          String formated as {input state}{gate matrix}->{output state}.
      """
      super(DefaultEinsumCache, self).__init__()
      self.nqubits = nqubits

      if nqubits + len(qubits) > len(self._chars):
          raise NotImplementedError("Not enough einsum characters.")

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

    def _calculate_density_matrix(self, is_controlled_by: bool = False):
        if self.nqubits > len(self.rest):
            raise NotImplementedError("Not enough einsum characters.")

        rest = self.rest[:self.nqubits]

        self._left = f"{self.input}{rest},{self.gate}->{self.output}{rest}"
        self._right = f"{rest}{self.input},{self.gate}->{rest}{self.output}"

        if is_controlled_by:
            if self.nqubits + 1 > len(self.rest):
                raise NotImplementedError("Not enough einsum characters.")
            c = self.rest[self.nqubits]
            self._left0 = f"{c}{self.input}{rest},{self.gate}->{c}{self.output}{rest}"
            self._right0 = f"{c}{rest}{self.input},{self.gate}->{c}{rest}{self.output}"


class MatmulEinsumCache(BaseCache):

    def __init__(self, qubits: Sequence[int], nqubits: int,
                 ncontrol: Optional[int] = None):
        """Creates indeces and shapes required for gate application with matmul.

        Args:
            qubits (tuple): Tuple with the qubit indices that the gate is applied to.
            nqubits (int): Total number of qubits in the circuit / state vector.

        Returns:
            Indices for the first transposition (before matmul) and the inverse
            transposition (after matmul) and the four reshape shapes.
        """
        super(MatmulEinsumCache, self).__init__()
        self.nqubits = nqubits
        self.ntargets = len(qubits)
        self.ncontrol = ncontrol
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

    def _calculate_density_matrix(self, is_controlled_by: bool = False):
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

        if is_controlled_by:
            raise NotImplementedError
            c_dim = 2 ** self.ncontrol - 1
            self._left0 = {}
            self._left0["ids"] = [len(self.left["ids"])] + self.left["ids"]
            self._left0["shapes"] = ((c_dim,) + self.shape + (self.nstates,),
                                     (2 ** self.ntargets, (2 ** self.nrest) * self.nstates * cdim),
                                     self.transposed_shape + (self.nstates, cdim)
                                     )


class ControlCache:
    """Helper tools for `controlled_by` gates.

    This class contains:
      A) an `order` that is used to transpose `state`
         so that control legs are moved in the front
      B) a `targets` list which is equivalent to the
         `target_qubits` tuple but each index is reduced
         by the amount of control qubits that preceed it.
    This method is called by the `nqubits` setter so that the loop runs
    once per gate (and not every time the gate is called).
    """

    def __init__(self, gate: base_gates.Gate):
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

        if self._reverse_dm is None:
            self.calculate_dm()
        return self._reverse_dm

    @staticmethod
    def calculate(gate: base_gates.Gate):
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
        additional_order = np.array(self._order) + len(self._order)
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
