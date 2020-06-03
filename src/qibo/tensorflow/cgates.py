# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import gates as base_gates
from qibo.config import DTYPEINT, DTYPE, DTYPECPX, GPU_MEASUREMENT_CUTOFF, CPU_NAME
from qibo.config import matrices
from qibo.tensorflow import custom_operators as op
from typing import Optional, Sequence, Tuple


class TensorflowGate:

    def __init__(self):
        if not tf.executing_eagerly():
            raise NotImplementedError("Custom operator gates should not be "
                                      "used in compiled mode.")

    def with_backend(self, backend: Optional[str] = None):
        """Used only for test compatibility with native gates.

        Custom kernel gates do not have different backends
        """
        return self

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        """Implements the `Gate` on a given state.

        Args:
            state (tf.Tensor): State vector with shape (2 ** nqubits,).
        """
        if self._nqubits is None:
            self.nqubits = int(np.log2(tuple(state.shape)[0]))


class MatrixGate(TensorflowGate):
    """``TensorflowGate`` that uses matrix to be applied to states."""

    def __init__(self):
        super(MatrixGate, self).__init__()
        self.matrix = None

    @base_gates.Gate.nqubits.setter
    def nqubits(self, n: int):
        base_gates.Gate.nqubits.fset(self, n)
        self._construct_matrix()

    def _construct_matrix(self):
        raise NotImplementedError

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        super(MatrixGate, self).__call__(state, is_density_matrix)
        return op.apply_gate(state, self.matrix, self.nqubits,
                             self.target_qubits[0], self.control_qubits)


class H(MatrixGate, base_gates.H):

    def __init__(self, q):
        base_gates.H.__init__(self, q)
        MatrixGate.__init__(self)

    def _construct_matrix(self):
        self.matrix = matrices.H


class X(TensorflowGate, base_gates.X):

    def __init__(self, q):
        base_gates.X.__init__(self, q)
        TensorflowGate.__init__(self)

    def controlled_by(self, *q):
        """Fall back to CNOT and Toffoli if controls are one or two."""
        if len(q) == 1:
            gate = CNOT(q[0], self.target_qubits[0])
        elif len(q) == 2:
            gate = TOFFOLI(q[0], q[1], self.target_qubits[0])
        else:
            gate = base_gates.X.controlled_by(self, *q)
        return gate

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_x(state, self.nqubits, self.target_qubits[0],
                          self.control_qubits)


class Y(TensorflowGate, base_gates.Y):

    def __init__(self, q):
        base_gates.Y.__init__(self, q)
        TensorflowGate.__init__(self)

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_y(state, self.nqubits, self.target_qubits[0],
                          self.control_qubits)


class Z(TensorflowGate, base_gates.Z):

    def __init__(self, q):
        base_gates.Z.__init__(self, q)
        TensorflowGate.__init__(self)

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_z(state, self.nqubits, self.target_qubits[0],
                          self.control_qubits)


class M(TensorflowGate, base_gates.M):
    from qibo.tensorflow import measurements

    def __init__(self, *q, register_name: Optional[str] = None):
        base_gates.M.__init__(self, *q, register_name=register_name)
        self._traceout = None

    @base_gates.Gate.nqubits.setter
    def nqubits(self, n: int):
        base_gates.Gate.nqubits.fset(self, n)

    @property
    def _traceout_str(self):
        """Einsum string used to trace out when state is density matrix."""
        if self._traceout is None:
            from qibo.tensorflow.einsum import DefaultEinsum
            qubits = set(self.unmeasured_qubits)
            # TODO: Remove ``DefaultEinsum`` dependence here
            self._traceout = DefaultEinsum.partialtrace_str(
              qubits, self.nqubits, measuring=True)
        return self._traceout

    def _calculate_probabilities(self, state: tf.Tensor,
                                 is_density_matrix: bool = False) -> tf.Tensor:
        """Calculates probabilities from state using Born's rule.

        Args:
            state: State vector of shape nqubits * (2,) or density matrix of
                shape 2 * nqubits * (2,).
            is_density_matrix: Flag that specifies whether `state` is a state
                vector or density matrix.

        Returns:
            Probabilities for measured qubits with shape len(target_qubits)* (2,).
        """
        # Trace out unmeasured qubits
        if is_density_matrix:
            probs = tf.cast(tf.einsum(self._traceout_str, state),
                            dtype=DTYPE)
        else:
            probs = tf.reduce_sum(tf.square(tf.abs(state)),
                                  axis=self.unmeasured_qubits)
        # Bring probs in the order specified by the user
        return tf.transpose(probs, perm=self.reduced_target_qubits)

    def __call__(self, state: tf.Tensor, nshots: int,
                 samples_only: bool = False,
                 is_density_matrix: bool = False) -> tf.Tensor:
        TensorflowGate.__call__(self, state, is_density_matrix)
        probs_dim = 2 ** len(self.target_qubits)

        shape = (1 + is_density_matrix) * self.nqubits * (2,)
        probs = self._calculate_probabilities(
            tf.reshape(state, shape), is_density_matrix)
        logits = tf.math.log(tf.reshape(probs, (probs_dim,)))

        if nshots * probs_dim < GPU_MEASUREMENT_CUTOFF:
            # Use default device to perform sampling
            samples_dec = tf.random.categorical(logits[tf.newaxis], nshots,
                                                dtype=DTYPEINT)[0]
        else:
            # Force using CPU to perform sampling because if GPU is used
            # it will cause a `ResourceExhaustedError`
            if CPU_NAME is None:
                raise RuntimeError("Cannot find CPU device to use for sampling.")
            with tf.device(CPU_NAME):
                samples_dec = tf.random.categorical(logits[tf.newaxis], nshots,
                                                    dtype=DTYPEINT)[0]
        if samples_only:
            return samples_dec
        return self.measurements.GateResult(
            self.qubits, state, decimal_samples=samples_dec)


class RX(MatrixGate, base_gates.RX):

    def __init__(self, q, theta):
        base_gates.RX.__init__(self, q, theta)
        MatrixGate.__init__(self)

    def _construct_matrix(self):
        th = tf.cast(self.theta, dtype=DTYPECPX)
        self.matrix = (tf.cos(th / 2.0) * matrices.I -
                       1j * tf.sin(th / 2.0) * matrices.X)

    # TODO: Make `_construct_matrix` a static method for all relevant gates
    @staticmethod
    def construct_unitary(theta):
        t = tf.cast(theta, dtype=DTYPECPX)
        return tf.cos(t / 2.0) * matrices.I - 1j * tf.sin(t / 2.0) * matrices.X


class RY(MatrixGate, base_gates.RY):

    def __init__(self, q, theta):
        base_gates.RY.__init__(self, q, theta)
        MatrixGate.__init__(self)

    def _construct_matrix(self):
        th = tf.cast(self.theta, dtype=DTYPECPX)
        self.matrix = (tf.cos(th / 2.0) * matrices.I -
                       1j * tf.sin(th / 2.0) * matrices.Y)

    @staticmethod
    def construct_unitary(theta):
        t = tf.cast(theta, dtype=DTYPECPX)
        return tf.cos(t / 2.0) * matrices.I - 1j * tf.sin(t / 2.0) * matrices.Y


class RZ(MatrixGate, base_gates.RZ):

    def __init__(self, q, theta):
        base_gates.RZ.__init__(self, q, theta)
        MatrixGate.__init__(self)

    def _construct_matrix(self):
        th = tf.cast(self.theta, dtype=DTYPECPX)
        phase = tf.exp(1j * th / 2.0)[tf.newaxis]
        diag = tf.concat([tf.math.conj(phase), phase], axis=0)
        self.matrix = tf.linalg.diag(diag)

    @staticmethod
    def construct_unitary(theta):
        t = tf.cast(theta, dtype=DTYPECPX)
        phase = tf.exp(1j * t / 2.0)[tf.newaxis]
        diag = tf.concat([tf.math.conj(phase), phase], axis=0)
        return tf.linalg.diag(diag)


class CNOT(TensorflowGate, base_gates.CNOT):

    def __init__(self, q0, q1):
        base_gates.CNOT.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False):
        return X.__call__(self, state, is_density_matrix)


class CZPow(MatrixGate, base_gates.CZPow):

    def __init__(self, q0, q1, theta):
        base_gates.CZPow.__init__(self, q0, q1, theta)

    def _construct_matrix(self):
        if self.theta == np.pi:
            self.matrix = tf.cast(-1, dtype=DTYPECPX)
        else:
            self.matrix = tf.exp(1j * tf.cast(self.theta, dtype=DTYPECPX))

    @staticmethod
    def construct_unitary(theta=np.pi):
        if theta == np.pi:
            m = np.eye(4)
            m[3, 3] = -1
            return tf.cast(m, dtype=DTYPECPX)
        phase = tf.exp(1j * tf.cast(theta, dtype=DTYPECPX))
        diag = tf.concat([tf.eye(3, dtype=DTYPECPX), phase], axis=0)
        return tf.linalg.diag(diag)

    def __call__(self, state, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_zpow(state, self.matrix, self.nqubits,
                             self.target_qubits[0], self.control_qubits)


class SWAP(TensorflowGate, base_gates.SWAP):

    def __init__(self, q0, q1):
        base_gates.SWAP.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    def __call__(self, state, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_swap(state, self.nqubits, self.target_qubits,
                             self.control_qubits)


class fSim(MatrixGate, base_gates.fSim):

    def __init__(self, q0, q1, theta, phi):
        base_gates.fSim.__init__(self, q0, q1, theta, phi)

    def _construct_matrix(self):
        th = tf.cast(self.theta, dtype=DTYPECPX)
        rotation = tf.cos(th) * matrices.I - 1j * tf.sin(th) * matrices.X
        phase = tf.exp(-1j * tf.cast(self.phi, dtype=DTYPECPX))
        self.matrix = tf.concat([tf.reshape(rotation, (4,)), [phase]], axis=0)

    def __call__(self, state, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_fsim(state, self.matrix, self.nqubits,
                             self.target_qubits, self.control_qubits)


class GeneralizedfSim(MatrixGate, base_gates.GeneralizedfSim):

    def __init__(self, q0, q1, unitary, phi):
        base_gates.GeneralizedfSim.__init__(self, q0, q1, unitary, phi)
        shape = tuple(self.unitary.shape)
        if shape != (2, 2):
            raise ValueError("Invalid shape {} of rotation for generalized "
                             "fSim gate".format(shape))

    def _construct_matrix(self):
        rotation = tf.cast(self.unitary, dtype=DTYPECPX)
        phase = tf.exp(-1j * tf.cast(self.phi, dtype=DTYPECPX))
        rotation = tf.reshape(rotation, (4,))
        self.matrix = tf.concat([tf.reshape(rotation, (4,)), [phase]], axis=0)

    def __call__(self, state, is_density_matrix: bool = False):
        return fSim.__call__(self, state, is_density_matrix)


class TOFFOLI(TensorflowGate, base_gates.TOFFOLI):

    def __init__(self, q0, q1, q2):
        base_gates.TOFFOLI.__init__(self, q0, q1, q2)
        TensorflowGate.__init__(self)

    def __call__(self, state, is_density_matrix: bool = False):
        return X.__call__(self, state, is_density_matrix)


class Unitary(MatrixGate, base_gates.Unitary):

    def __init__(self, unitary, *q, name: Optional[str] = None):
        base_gates.Unitary.__init__(self, unitary, *q, name=name)
        MatrixGate.__init__(self)

        rank = self.rank
        if rank > 2:
            raise NotImplementedError("Unitary matrix gate supports only one "
                                      "qubit gates but {} target qubits were "
                                      "given.".format(len(self.target_qubits)))

        shape = tuple(self.unitary.shape)
        if shape != (2 ** rank, 2 ** rank):
            raise ValueError("Invalid shape {} of unitary matrix acting on "
                             "{} target qubits.".format(shape, rank))

    def _construct_matrix(self):
        self.matrix = tf.convert_to_tensor(self.unitary, dtype=DTYPECPX)

    @property
    def rank(self) -> int:
        return len(self.target_qubits)

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        TensorflowGate.__call__(self, state, is_density_matrix)
        if self.rank == 1:
            return op.apply_gate(state, self.matrix, self.nqubits,
                                 self.target_qubits[0],
                                 self.control_qubits)
        if self.rank == 2:
            return op.apply_twoqubit_gate(state, self.matrix, self.nqubits,
                                          self.target_qubits,
                                          self.control_qubits)


class VariationalLayer(MatrixGate, base_gates.VariationalLayer):

    def __init__(self, qubit_pairs, one_qubit_gate, two_qubit_gate, thetas,
                 name: Optional[str] = None):
        base_gates.VariationalLayer.__init__(self, qubit_pairs, one_qubit_gate,
                                             two_qubit_gate, thetas, name=name)
        MatrixGate.__init__(self)

    @staticmethod
    def _tfkron(m1, m2):
        m = tf.transpose(tf.tensordot(m1, m2, axes=0), [0, 2, 1, 3])
        return tf.reshape(m, (4, 4))

    def _construct_matrix(self):
        self.matrix = tf.stack([self._tfkron(
            self.one_qubit_gate.construct_unitary(self.thetas[q1]),
            self.one_qubit_gate.construct_unitary(self.thetas[q2]))
                             for q1, q2 in self.qubit_pairs], axis=0)
        entangling_matrix = self.two_qubit_gate.construct_unitary()
        self.matrix = tf.matmul(entangling_matrix, self.matrix)

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        for i, targets in enumerate(self.qubit_pairs):
            state = op.apply_twoqubit_gate(state, self.matrix[i], self.nqubits, targets)
        return state


class Flatten(TensorflowGate, base_gates.Flatten):

    def __init__(self, coefficients):
        base_gates.Flatten.__init__(self, coefficients)
        TensorflowGate.__init__(self)

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        TensorflowGate.__call__(self, state, is_density_matrix)
        _state = np.array(self.coefficients).reshape(state.shape)
        return tf.convert_to_tensor(_state, dtype=state.dtype)


# TODO: Add channels once density matrices are supported by custom operators
