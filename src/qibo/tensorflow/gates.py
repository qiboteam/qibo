# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import sys
import numpy as np
import tensorflow as tf
from qibo.base import gates as base_gates
from qibo.base import cache
from qibo.config import matrices
from qibo.config import BACKEND, DTYPES, GPU_MEASUREMENT_CUTOFF, CPU_NAME
from typing import Dict, List, Optional, Sequence, Tuple


class TensorflowGate(base_gates.Gate):
    """The base Tensorflow gate.

    **Properties:**
        matrix: The matrix that represents the gate to be applied.
            This is (2, 2) for 1-qubit gates and (4, 4) for 2-qubit gates.
        qubits: List with the qubits that the gate is applied to.
    """
    def __init__(self):
        self.calculation_cache = None
        # For `controlled_by` gates (see `cache.ControlCache` for more details)
        self.control_cache = None
        # Gate matrices
        self.matrix = None
        # Einsum backend
        self.einsum = BACKEND.get('EINSUM')

    def _construct_matrix(self):
        self.matrix = self.construct_unitary()

    @staticmethod
    def construct_unitary(*args) -> tf.Tensor:
        """Constructs unitary matrix corresponding to the gate.

        Args:
            *args: Variational parameters for parametrized gates.
        """
        raise NotImplementedError

    @base_gates.Gate.nqubits.setter
    def nqubits(self, n: int):
        """Sets the number of qubit that this gate acts on.

        This is called automatically by the `Circuit.add` method if the gate
        is used on a `Circuit`. If the gate is called on a state then `nqubits`
        is set during the first `__call__`.
        When `nqubits` is set we also calculate the einsum string so that it
        is calculated only once per gate.
        """
        base_gates.Gate.nqubits.fset(self, n)
        if self.is_controlled_by:
            self.control_cache = cache.ControlCache(self)
            nactive = n - len(self.control_qubits)
            targets = self.control_cache.targets
            self.calculation_cache = self.einsum.create_cache(targets, nactive, ncontrol=len(self.control_qubits))
        else:
            self.calculation_cache = self.einsum.create_cache(self.qubits, n)
        self.calculation_cache.cast_shapes(lambda x: tf.cast(x, dtype=DTYPES.get('DTYPEINT')))
        self._construct_matrix()

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

    @staticmethod
    def construct_unitary() -> tf.Tensor:
        return matrices.H


class X(TensorflowGate, base_gates.X):

    _MODULE = sys.modules[__name__]

    def __init__(self, q):
        base_gates.X.__init__(self, q)
        TensorflowGate.__init__(self)

    @staticmethod
    def construct_unitary() -> tf.Tensor:
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

    @staticmethod
    def construct_unitary() -> tf.Tensor:
        return matrices.Y


class Z(TensorflowGate, base_gates.Z):

    _MODULE = sys.modules[__name__]

    def __init__(self, q):
        base_gates.Z.__init__(self, q)
        TensorflowGate.__init__(self)

    @staticmethod
    def construct_unitary() -> tf.Tensor:
        return matrices.Z


class RX(TensorflowGate, base_gates.RX):

    def __init__(self, q, theta):
        base_gates.RX.__init__(self, q, theta)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        self.matrix = self.construct_unitary(self.theta)

    @staticmethod
    def construct_unitary(theta) -> tf.Tensor:
        t = tf.cast(theta, dtype=DTYPES.get('DTYPECPX'))
        return tf.cos(t / 2.0) * matrices.I - 1j * tf.sin(t / 2.0) * matrices.X


class RY(TensorflowGate, base_gates.RY):

    def __init__(self, q, theta):
        base_gates.RY.__init__(self, q, theta)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        self.matrix = self.construct_unitary(self.theta)

    @staticmethod
    def construct_unitary(theta) -> tf.Tensor:
        t = tf.cast(theta, dtype=DTYPES.get('DTYPECPX'))
        return tf.cos(t / 2.0) * matrices.I - 1j * tf.sin(t / 2.0) * matrices.Y


class RZ(TensorflowGate, base_gates.RZ):

    def __init__(self, q, theta):
        base_gates.RZ.__init__(self, q, theta)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        self.matrix = self.construct_unitary(self.theta)

    @staticmethod
    def construct_unitary(theta) -> tf.Tensor:
        t = tf.cast(theta, dtype=DTYPES.get('DTYPECPX'))
        phase = tf.exp(1j * t / 2.0)[tf.newaxis]
        diag = tf.concat([tf.math.conj(phase), phase], axis=0)
        return tf.linalg.diag(diag)


class CNOT(TensorflowGate, base_gates.CNOT):

    def __init__(self, q0, q1):
        base_gates.CNOT.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        self.matrix = matrices.CNOT

    @staticmethod
    def construct_unitary() -> tf.Tensor:
        return tf.reshape(matrices.CNOT, (4, 4))


class CZ(TensorflowGate, base_gates.CZ):

    def __init__(self, q0, q1):
        base_gates.CZ.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        self.matrix = tf.reshape(self.construct_unitary(), 4 * (2,))

    @staticmethod
    def construct_unitary() -> tf.Tensor:
        diag = tf.cast(tf.concat([tf.ones(3), [-1]], axis=0), dtype=DTYPES.get('DTYPECPX'))
        return tf.linalg.diag(diag)


class CZPow(TensorflowGate, base_gates.CZPow):

    def __init__(self, q0, q1, theta):
        base_gates.CZPow.__init__(self, q0, q1, theta)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        self.matrix = tf.reshape(self.construct_unitary(self.theta), 4 * (2,))

    @staticmethod
    def construct_unitary(theta) -> tf.Tensor:
        th = tf.cast(theta, dtype=DTYPES.get('DTYPECPX'))
        phase = tf.exp(1j * th)[tf.newaxis]
        diag = tf.concat([tf.ones(3, dtype=DTYPES.get('DTYPECPX')), phase], axis=0)
        return tf.linalg.diag(diag)


class SWAP(TensorflowGate, base_gates.SWAP):

    def __init__(self, q0, q1):
        base_gates.SWAP.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        self.matrix = matrices.SWAP

    @staticmethod
    def construct_unitary() -> tf.Tensor:
        return tf.cast([[1, 0, 0, 0], [0, 0, 1, 0],
                        [0, 1, 0, 0], [0, 0, 0, 1]], dtype=DTYPES.get('DTYPECPX'))


class fSim(TensorflowGate, base_gates.fSim):

    def __init__(self, q0, q1, theta, phi):
        base_gates.fSim.__init__(self, q0, q1, theta, phi)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        dtype = DTYPES.get("DTYPECPX")
        th = tf.cast(self.theta, dtype=dtype)
        rotation = tf.cos(th) * matrices.I - 1j * tf.sin(th) * matrices.X
        phase = tf.exp(-1j * tf.cast(self.phi, dtype=dtype))
        self.matrix = tf.eye(4, dtype=dtype)
        self.matrix = tf.tensor_scatter_nd_update(self.matrix, [[3, 3]], [phase])
        rotation = tf.reshape(rotation, (4,))
        ids = [[1, 1], [1, 2], [2, 1], [2, 2]]
        self.matrix = tf.tensor_scatter_nd_update(self.matrix, ids, rotation)
        self.matrix = tf.reshape(self.matrix, 4 * (2,))


class GeneralizedfSim(TensorflowGate, base_gates.GeneralizedfSim):

    def __init__(self, q0, q1, unitary, phi):
        base_gates.GeneralizedfSim.__init__(self, q0, q1, unitary, phi)
        TensorflowGate.__init__(self)
        shape = tuple(self.unitary.shape)
        if shape != (2, 2):
            raise ValueError("Invalid shape {} of rotation for generalized "
                             "fSim gate".format(shape))

    def _construct_matrix(self):
        dtype = DTYPES.get("DTYPECPX")
        rotation = tf.cast(self.unitary, dtype=dtype)
        phase = tf.exp(-1j * tf.cast(self.phi, dtype=dtype))
        self.matrix = tf.eye(4, dtype=dtype)
        self.matrix = tf.tensor_scatter_nd_update(self.matrix, [[3, 3]], [phase])
        rotation = tf.reshape(rotation, (4,))
        ids = [[1, 1], [1, 2], [2, 1], [2, 2]]
        self.matrix = tf.tensor_scatter_nd_update(self.matrix, ids, rotation)
        self.matrix = tf.reshape(self.matrix, 4 * (2,))


class TOFFOLI(TensorflowGate, base_gates.TOFFOLI):

    def __init__(self, q0, q1, q2):
        base_gates.TOFFOLI.__init__(self, q0, q1, q2)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        self.matrix = matrices.TOFFOLI

    @staticmethod
    def construct_unitary() -> tf.Tensor:
        return tf.reshape(matrices.TOFFOLI, (8, 8))


class Unitary(TensorflowGate, base_gates.Unitary):

    def __init__(self, unitary, *q, name: Optional[str] = None):
        base_gates.Unitary.__init__(self, unitary, *q, name=name)
        TensorflowGate.__init__(self)

        rank = len(self.target_qubits)
        shape = tuple(self.unitary.shape)
        if shape != (2 ** rank, 2 ** rank):
            raise ValueError("Invalid shape {} of unitary matrix acting on "
                             "{} target qubits.".format(shape, rank))

    def _construct_matrix(self):
        rank = len(self.target_qubits)
        self.matrix = tf.cast(self.unitary, dtype=DTYPES.get("DTYPECPX"))
        self.matrix = tf.reshape(self.matrix, 2 * rank * (2,))


class VariationalLayer(TensorflowGate, base_gates.VariationalLayer):

    def __init__(self, qubit_pairs: List[Tuple[int, int]],
                 one_qubit_gate, two_qubit_gate,
                 params_map: Dict[int, float],
                 params_map2: Optional[Dict[int, float]] = None,
                 name: Optional[str] = None):
        base_gates.VariationalLayer.__init__(self, qubit_pairs,
                                             one_qubit_gate, two_qubit_gate,
                                             params_map, params_map2,
                                             name=name)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        matrices = tf.stack([self._tfkron(
            self.one_qubit_gate.construct_unitary(self.params_map[q1]),
            self.one_qubit_gate.construct_unitary(self.params_map[q2]))
                             for q1, q2 in self.qubit_pairs], axis=0)
        entangling_matrix = self.two_qubit_gate.construct_unitary()
        matrices = tf.matmul(entangling_matrix, matrices)
        if self.additional_target is not None:
            additional_matrix = self.one_qubit_gate.construct_unitary(
                self.params_map[self.additional_target])
        if self.params_map2 is not None:
            matrices2 = tf.stack([self._tfkron(
                self.one_qubit_gate.construct_unitary(self.params_map2[q1]),
                self.one_qubit_gate.construct_unitary(self.params_map2[q2]))
                                for q1, q2 in self.qubit_pairs], axis=0)
            matrices = tf.matmul(matrices2, matrices)
            if self.additional_target is not None:
                additional_matrix = tf.matmul(
                    self.one_qubit_gate.construct_unitary(
                        self.params_map2[self.additional_target]),
                    additional_matrix)

        self.unitaries = [Unitary(matrices[i], *targets)
                          for i, targets in enumerate(self.qubit_pairs)]
        if self.additional_target is not None:
            self.additional_unitary = Unitary(additional_matrix, self.additional_target)
        else:
            self.additional_unitary = None

    @staticmethod
    def _tfkron(m1, m2):
        m = tf.transpose(tf.tensordot(m1, m2, axes=0), [0, 2, 1, 3])
        return tf.reshape(m, (4, 4))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        for i, unitary in enumerate(self.unitaries):
            state = unitary(state, is_density_matrix)
        if self.additional_unitary is not None:
            state = self.additional_unitary(state, is_density_matrix)
        return state


class Flatten(TensorflowGate, base_gates.Flatten):

    def __init__(self, coefficients):
        base_gates.Flatten.__init__(self, coefficients)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        pass

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        if self._nqubits is None:
            if is_density_matrix:
                self.nqubits = len(tuple(state.shape)) // 2
            else:
                self.nqubits = len(tuple(state.shape))

        if is_density_matrix:
            shape = 2 * self.nqubits * (2,)
        else:
            shape = self.nqubits * (2,)

        _state = np.array(self.coefficients).reshape(shape)
        return tf.convert_to_tensor(_state, dtype=DTYPES.get("DTYPECPX"))


class TensorflowChannel(TensorflowGate):
    """Base Tensorflow channels.

    All channels should inherit this class.
    """

    @TensorflowGate.nqubits.setter
    def nqubits(self, n: int):
        self._nqubits = n
        self._nstates = 2 ** n
        for gate in self.gates:
            gate.nqubits = n

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = True
                 ) -> tf.Tensor:
        if not is_density_matrix:
            raise ValueError("Noise channel can only be applied to density "
                             "matrices.")
        if self._nqubits is None:
            self.nqubits = len(tuple(state.shape)) // 2

        return self._krauss_sum(state)

    def _krauss_sum(self, state: tf.Tensor) -> tf.Tensor:
        """Loops over `self.gates` to calculate sum of Krauss operators."""
        raise NotImplementedError


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
