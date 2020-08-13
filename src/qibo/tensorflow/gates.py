# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import gates as base_gates
from qibo.base import cache
from qibo.config import tfmatrices as matrices
from qibo.config import BACKEND, DTYPES, raise_error
from typing import Dict, List, Optional, Sequence, Tuple


class TensorflowGate(base_gates.Gate):
    """The base Tensorflow gate.

    **Properties:**
        matrix: The matrix that represents the gate to be applied.
            This is (2, 2) for 1-qubit gates and (4, 4) for 2-qubit gates.
        qubits: List with the qubits that the gate is applied to.
    """

    import sys
    module = sys.modules[__name__]

    def __init__(self):
        self.calculation_cache = None
        # For `controlled_by` gates (see `cache.ControlCache` for more details)
        self.control_cache = None
        # Gate matrices
        self.matrix = None
        # Einsum backend
        self.einsum = BACKEND.get('EINSUM')

    def _prepare(self):
        matrix = self.construct_unitary()
        rank = int(np.log2(int(matrix.shape[0])))
        self.matrix = tf.reshape(matrix, 2 * rank * (2,))

    def __matmul__(self, other: "TensorflowGate") -> "TensorflowGate":
        gate = base_gates.Gate.__matmul__(self, other)
        if gate is None:
            gate = Unitary(tf.matmul(self.unitary, other.unitary), *self.qubits)
        return gate

    @staticmethod
    def control_unitary(unitary: tf.Tensor) -> tf.Tensor:
        from qibo.tensorflow import cgates
        return cgates.TensorflowGate.control_unitary(unitary)

    @base_gates.Gate.nqubits.setter
    def nqubits(self, n: int):
        """Sets the number of qubit that this gate acts on.

        This is called automatically by the `Circuit.add` method if the gate
        is used on a `Circuit`. If the gate is called on a state then `nqubits`
        is set during the first `__call__`.
        When `nqubits` is set we also calculate the einsum string so that it
        is calculated only once per gate.
        """
        base_gates.Gate.nqubits.fset(self, n) # pylint: disable=no-member
        if self.is_controlled_by:
            self.control_cache = cache.ControlCache(self)
            nactive = n - len(self.control_qubits)
            targets = self.control_cache.targets
            self.calculation_cache = self.einsum.create_cache(targets, nactive, ncontrol=len(self.control_qubits))
        else:
            self.calculation_cache = self.einsum.create_cache(self.qubits, n)
        self.calculation_cache.cast_shapes(lambda x: tf.cast(x, dtype=DTYPES.get('DTYPEINT')))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        """Implements the `Gate` on a given state."""
        if self._nqubits is None:
            if is_density_matrix:
                self.nqubits = len(tuple(state.shape)) // 2
            else:
                self.nqubits = len(tuple(state.shape))

        if self.is_controlled_by:
            return self._controlled_by_call(state, is_density_matrix)

        if is_density_matrix:
            state = self.einsum(self.calculation_cache.right, state,
                                tf.math.conj(self.matrix))
            state = self.einsum(self.calculation_cache.left, state, self.matrix)
            return state

        return self.einsum(self.calculation_cache.vector, state, self.matrix)

    def _controlled_by_call(self, state: tf.Tensor,
                            is_density_matrix: bool = False) -> tf.Tensor:
        """Gate __call__ method for `controlled_by` gates."""
        ncontrol = len(self.control_qubits)
        nactive = self.nqubits - ncontrol

        transpose_order = self.control_cache.order(is_density_matrix)
        reverse_transpose_order = self.control_cache.reverse(is_density_matrix)

        state = tf.transpose(state, transpose_order)
        if is_density_matrix:
            state = tf.reshape(state, 2 * (2 ** ncontrol,) + 2 * nactive * (2,))

            updates01 = self.einsum(self.calculation_cache.right0,
                                    state[:-1, -1],
                                    tf.math.conj(self.matrix))
            updates10 = self.einsum(self.calculation_cache.left0,
                                    state[-1, :-1],
                                    self.matrix)

            updates11 = self.einsum(self.calculation_cache.right,
                                    state[-1, -1], tf.math.conj(self.matrix))
            updates11 = self.einsum(self.calculation_cache.left,
                                    updates11, self.matrix)

            updates01 = tf.concat([state[:-1, :-1], updates01[:, tf.newaxis]], axis=1)
            updates10 = tf.concat([updates10, updates11[tf.newaxis]], axis=0)
            state = tf.concat([updates01, updates10[tf.newaxis]], axis=0)
            state = tf.reshape(state, 2 * self.nqubits * (2,))

        else:
            # Apply `einsum` only to the part of the state where all controls
            # are active. This should be `state[-1]`
            state = tf.reshape(state, (2 ** ncontrol,) + nactive * (2,))
            updates = self.einsum(self.calculation_cache.vector, state[-1],
                                  self.matrix)
            # Concatenate the updated part of the state `updates` with the
            # part of of the state that remained unaffected `state[:-1]`.
            state = tf.concat([state[:-1], updates[tf.newaxis]], axis=0)
            state = tf.reshape(state, self.nqubits * (2,))

        return tf.transpose(state, reverse_transpose_order)


class H(TensorflowGate, base_gates.H):

    def __init__(self, q):
        base_gates.H.__init__(self, q)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        return matrices.H


class X(TensorflowGate, base_gates.X):

    def __init__(self, q):
        base_gates.X.__init__(self, q)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        return matrices.X

    def controlled_by(self, *q):
        """Fall back to CNOT and Toffoli if controls are one or two."""
        gate = base_gates.X.controlled_by(self, *q)
        gate.einsum = self.einsum
        return gate


class Y(TensorflowGate, base_gates.Y):

    def __init__(self, q):
        base_gates.Y.__init__(self, q)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        return matrices.Y


class Z(TensorflowGate, base_gates.Z):

    def __init__(self, q):
        base_gates.Z.__init__(self, q)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        return matrices.Z


class RX(TensorflowGate, base_gates.RX):

    def __init__(self, q, theta):
        base_gates.RX.__init__(self, q, theta)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        t = tf.cast(self.parameter, dtype=DTYPES.get('DTYPECPX'))
        return tf.cos(t / 2.0) * matrices.I - 1j * tf.sin(t / 2.0) * matrices.X


class RY(TensorflowGate, base_gates.RY):

    def __init__(self, q, theta):
        base_gates.RY.__init__(self, q, theta)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        t = tf.cast(self.parameter, dtype=DTYPES.get('DTYPECPX'))
        return tf.cos(t / 2.0) * matrices.I - 1j * tf.sin(t / 2.0) * matrices.Y


class RZ(TensorflowGate, base_gates.RZ):

    def __init__(self, q, theta):
        base_gates.RZ.__init__(self, q, theta)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        t = tf.cast(self.parameter, dtype=DTYPES.get('DTYPECPX'))
        phase = tf.exp(1j * t / 2.0)[tf.newaxis]
        diag = tf.concat([tf.math.conj(phase), phase], axis=0)
        return tf.linalg.diag(diag)


class ZPow(TensorflowGate, base_gates.ZPow):

    def __init__(self, q, theta):
        base_gates.ZPow.__init__(self, q, theta)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        t = tf.cast(self.parameter, dtype=DTYPES.get('DTYPECPX'))
        phase = tf.exp(1j * t)
        diag = tf.concat([1, phase], axis=0)
        return tf.linalg.diag(diag)


class CNOT(TensorflowGate, base_gates.CNOT):

    def __init__(self, q0, q1):
        base_gates.CNOT.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        return tf.reshape(matrices.CNOT, (4, 4))


class CZ(TensorflowGate, base_gates.CZ):

    def __init__(self, q0, q1):
        base_gates.CZ.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        diag = tf.cast(tf.concat([tf.ones(3), [-1]], axis=0),
                       dtype=DTYPES.get('DTYPECPX'))
        return tf.linalg.diag(diag)


class CZPow(TensorflowGate, base_gates.CZPow):

    def __init__(self, q0, q1, theta):
        base_gates.CZPow.__init__(self, q0, q1, theta)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        dtype = DTYPES.get('DTYPECPX')
        th = tf.cast(self.parameter, dtype=dtype)
        phase = tf.exp(1j * th)[tf.newaxis]
        diag = tf.concat([tf.ones(3, dtype=dtype), phase], axis=0)
        return tf.linalg.diag(diag)


class SWAP(TensorflowGate, base_gates.SWAP):

    def __init__(self, q0, q1):
        base_gates.SWAP.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        return tf.cast([[1, 0, 0, 0], [0, 0, 1, 0],
                        [0, 1, 0, 0], [0, 0, 0, 1]],
                       dtype=DTYPES.get('DTYPECPX'))


class fSim(TensorflowGate, base_gates.fSim):

    def __init__(self, q0, q1, theta, phi):
        base_gates.fSim.__init__(self, q0, q1, theta, phi)
        TensorflowGate.__init__(self)

    def construct_unitary(self):
        from qibo.tensorflow import cgates
        return cgates.fSim.construct_unitary(self)


class GeneralizedfSim(TensorflowGate, base_gates.GeneralizedfSim):

    def __init__(self, q0, q1, unitary, phi):
        base_gates.GeneralizedfSim.__init__(self, q0, q1, unitary, phi)
        TensorflowGate.__init__(self)

    def construct_unitary(self):
        from qibo.tensorflow import cgates
        return cgates.GeneralizedfSim.construct_unitary(self)


class TOFFOLI(TensorflowGate, base_gates.TOFFOLI):

    def __init__(self, q0, q1, q2):
        base_gates.TOFFOLI.__init__(self, q0, q1, q2)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        return tf.reshape(matrices.TOFFOLI, (8, 8))


class Unitary(TensorflowGate, base_gates.Unitary):

    def __init__(self, unitary, *q, name: Optional[str] = None):
        if not isinstance(unitary, (np.ndarray, tf.Tensor)):
            raise_error(TypeError, "Unknown type {} of unitary matrix."
                                   "".format(type(unitary)))
        base_gates.Unitary.__init__(self, unitary, *q, name=name)
        TensorflowGate.__init__(self)
        self._unitary = self.construct_unitary()

    def construct_unitary(self):
        unitary = self.parameter
        rank = int(np.log2(int(unitary.shape[0])))
        dtype = DTYPES.get('DTYPECPX')
        if isinstance(unitary, tf.Tensor):
            matrix = tf.identity(tf.cast(unitary, dtype=dtype))
        elif isinstance(unitary, np.ndarray):
            matrix = tf.convert_to_tensor(unitary, dtype=dtype)
        return matrix


class VariationalLayer(TensorflowGate, base_gates.VariationalLayer):

    from qibo.tensorflow import cgates

    def __init__(self, qubits: List[int], pairs: List[Tuple[int, int]],
                 one_qubit_gate, two_qubit_gate,
                 params: List[float],
                 params2: Optional[List[float]] = None,
                 name: Optional[str] = None):
        base_gates.VariationalLayer.__init__(self, qubits, pairs,
                                             one_qubit_gate, two_qubit_gate,
                                             params, params2,
                                             name=name)
        TensorflowGate.__init__(self)
        self.unitary_constructor = Unitary

    @staticmethod
    def _tfkron(m1, m2):
        m = tf.transpose(tf.tensordot(m1, m2, axes=0), [0, 2, 1, 3])
        return tf.reshape(m, (4, 4))

    def _calculate_unitaries(self):
        matrices = tf.stack([self._tfkron(
            self.one_qubit_gate(q1, theta=self.params[q1]).unitary,
            self.one_qubit_gate(q2, theta=self.params[q2]).unitary)
                             for q1, q2 in self.pairs], axis=0)
        entangling_matrix = self.two_qubit_gate(0, 1).unitary
        matrices = tf.matmul(entangling_matrix, matrices)

        additional_matrix = None
        q = self.additional_target
        if q is not None:
            additional_matrix = self.one_qubit_gate(
                q, theta=self.params[q]).unitary

        if self.params2:
            matrices2 = tf.stack([self._tfkron(
                self.one_qubit_gate(q1, theta=self.params2[q1]).unitary,
                self.one_qubit_gate(q2, theta=self.params2[q2]).unitary)
                                for q1, q2 in self.pairs], axis=0)
            matrices = tf.matmul(matrices2, matrices)

            q = self.additional_target
            if q is not None:
                additional_matrix = tf.matmul(
                    self.one_qubit_gate(q, theta=self.params2[q]).unitary,
                    additional_matrix)

        return matrices, additional_matrix

    def _prepare(self):
        matrices, additional_matrix = self._calculate_unitaries()
        self.unitaries = [self.unitary_constructor(matrices[i], *targets)
                          for i, targets in enumerate(self.pairs)]
        if self.additional_target is not None:
            self.additional_unitary = self.unitary_constructor(
                additional_matrix, self.additional_target)

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor: # pragma: no cover
        # impractical case because VariationalLayer is not called by circuits
        return self.cgates.VariationalLayer.__call__(self, state, is_density_matrix)


class TensorflowChannel(TensorflowGate):
    """Base Tensorflow channels.

    All channels should inherit this class.
    """

    def __init__(self):
        super(TensorflowChannel, self).__init__()
        self.gates = []

    def _prepare(self):
        pass

    @base_gates.Gate.nqubits.setter
    def nqubits(self, n: int):
        base_gates.Gate.nqubits.fset(self, n) # pylint: disable=no-member
        for gate in self.gates:
            gate.nqubits = n

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = True
                 ) -> tf.Tensor:
        if not is_density_matrix:
            raise_error(ValueError, "Noise channel can only be applied to density "
                                    "matrices.")
        if self._nqubits is None:
            self.nqubits = len(tuple(state.shape)) // 2

        return self._krauss_sum(state)

    def _krauss_sum(self, state: tf.Tensor) -> tf.Tensor: # pragma: no cover
        """Loops over `self.gates` to calculate sum of Krauss operators."""
        # abstract method
        raise_error(NotImplementedError)


class NoiseChannel(TensorflowChannel, base_gates.NoiseChannel):

    def __init__(self, q: int, px: float = 0, py: float = 0, pz: float = 0):
        base_gates.NoiseChannel.__init__(self, q, px, py, pz)
        TensorflowChannel.__init__(self)
        classes = (X, Y, Z)
        self.gates = [cl(q) for p, cl in zip(self.p, classes) if p > 0]

    def _krauss_sum(self, state: tf.Tensor) -> tf.Tensor:
        new_state = tf.zeros_like(state)
        for p, gate in zip(self.p, self.gates):
            new_state += p * gate(state, is_density_matrix=True)
        return (1 - self.total_p) * state + new_state


class GeneralChannel(TensorflowChannel, base_gates.GeneralChannel):

    def __init__(self, A: Sequence[Tuple[Tuple[int], np.ndarray]]):
        base_gates.GeneralChannel.__init__(self, A)
        TensorflowChannel.__init__(self)
        self.gates = [Unitary(m, *list(q)) for q, m in A]

    def _krauss_sum(self, state: tf.Tensor) -> tf.Tensor:
        new_state = tf.zeros_like(state)
        for gate in self.gates:
            new_state += gate(state, is_density_matrix=True)
        return new_state
