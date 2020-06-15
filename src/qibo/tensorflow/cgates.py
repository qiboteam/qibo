# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import sys
import numpy as np
import tensorflow as tf
from qibo.base import gates as base_gates
from qibo.config import BACKEND, DTYPES, GPU_MEASUREMENT_CUTOFF, CPU_NAME
from qibo.tensorflow import custom_operators as op
from typing import Dict, List, Optional, Sequence, Tuple


class TensorflowGate:

    def __init__(self):
        if not tf.executing_eagerly():
            raise NotImplementedError("Custom operator gates should not be "
                                      "used in compiled mode.")

    @staticmethod
    def construct_unitary(*args) -> tf.Tensor:
        """Constructs unitary matrix corresponding to the gate.

        This matrix is not necessarily used by ``__call__`` when applying the
        gate to a state vector.

        Args:
            *args: Variational parameters for parametrized gates.
        """
        raise NotImplementedError

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
        self._prepare()

    def _prepare(self):
        """Prepares the gate for application to state vectors.

        Called automatically by the ``nqubits`` setter.
        Calculates the ``matrix`` required to apply the gate to state vectors.
        This is not necessarily the same as the unitary matrix of the gate.
        """
        raise NotImplementedError

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        super(MatrixGate, self).__call__(state, is_density_matrix)
        return op.apply_gate(state, self.matrix, self.nqubits,
                             self.target_qubits[0], self.control_qubits)


class H(MatrixGate, base_gates.H):

    def __new__(cls, q):
        if BACKEND.get('GATES') == 'custom':
            return super(H, cls).__new__(cls)
        else:
            from qibo.tensorflow import gates
            return gates.H(q)

    def __init__(self, q):
        base_gates.H.__init__(self, q)
        MatrixGate.__init__(self)

    @staticmethod
    def construct_unitary() -> tf.Tensor:
        return tf.cast(np.array([[1, 1], [1, -1]]) / np.sqrt(2),
                       dtype=DTYPES.get('DTYPECPX'))

    def _prepare(self):
        self.matrix = self.construct_unitary()


class X(TensorflowGate, base_gates.X):

    _MODULE = sys.modules[__name__]

    def __init__(self, q):
        base_gates.X.__init__(self, q)
        TensorflowGate.__init__(self)

    @staticmethod
    def construct_unitary() -> tf.Tensor:
        return tf.cast([[0, 1], [1, 0]], dtype=DTYPES.get('DTYPECPX'))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_x(state, self.nqubits, self.target_qubits[0],
                          self.control_qubits)


class Y(TensorflowGate, base_gates.Y):

    def __init__(self, q):
        base_gates.Y.__init__(self, q)
        TensorflowGate.__init__(self)

    @staticmethod
    def construct_unitary() -> tf.Tensor:
        return 1j * tf.cast([[0, -1], [1, 0]], dtype=DTYPES.get('DTYPECPX'))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_y(state, self.nqubits, self.target_qubits[0],
                          self.control_qubits)


class Z(TensorflowGate, base_gates.Z):

    _MODULE = sys.modules[__name__]

    def __init__(self, q):
        base_gates.Z.__init__(self, q)
        TensorflowGate.__init__(self)

    @staticmethod
    def construct_unitary() -> tf.Tensor:
        return tf.cast([[1, 0], [0, -1]], dtype=DTYPES.get('DTYPECPX'))

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
                            dtype=DTYPES.get('DTYPE'))
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
                                                dtype=DTYPES.get('DTYPEINT'))[0]
        else:
            # Force using CPU to perform sampling because if GPU is used
            # it will cause a `ResourceExhaustedError`
            if CPU_NAME is None:
                raise RuntimeError("Cannot find CPU device to use for sampling.")
            with tf.device(CPU_NAME):
                samples_dec = tf.random.categorical(logits[tf.newaxis], nshots,
                                                    dtype=DTYPES.get('DTYPEINT'))[0]
        if samples_only:
            return samples_dec
        return self.measurements.GateResult(
            self.qubits, state, decimal_samples=samples_dec)


class RX(MatrixGate, base_gates.RX):

    def __init__(self, q, theta):
        base_gates.RX.__init__(self, q, theta)
        MatrixGate.__init__(self)

    @staticmethod
    def construct_unitary(theta) -> tf.Tensor:
        dtype = DTYPES.get('DTYPECPX')
        t = tf.cast(theta, dtype=dtype)
        I = tf.eye(2, dtype=dtype)
        X = tf.cast([[0, 1], [1, 0]], dtype=dtype)
        return tf.cos(t / 2.0) * I - 1j * tf.sin(t / 2.0) * X

    def _prepare(self):
        self.matrix = self.construct_unitary(self.theta)


class RY(MatrixGate, base_gates.RY):

    def __init__(self, q, theta):
        base_gates.RY.__init__(self, q, theta)
        MatrixGate.__init__(self)

    @staticmethod
    def construct_unitary(theta) -> tf.Tensor:
        dtype = DTYPES.get('DTYPECPX')
        t = tf.cast(theta, dtype=dtype)
        I = tf.eye(2, dtype=dtype)
        iY = tf.cast([[0, 1], [-1, 0]], dtype=dtype)
        return tf.cos(t / 2.0) * I - tf.sin(t / 2.0) * iY

    def _prepare(self):
        self.matrix = self.construct_unitary(self.theta)


class RZ(MatrixGate, base_gates.RZ):

    def __init__(self, q, theta):
        base_gates.RZ.__init__(self, q, theta)
        MatrixGate.__init__(self)

    @staticmethod
    def construct_unitary(theta) -> tf.Tensor:
        t = tf.cast(theta, dtype=DTYPES.get('DTYPECPX'))
        phase = tf.exp(1j * t / 2.0)[tf.newaxis]
        diag = tf.concat([tf.math.conj(phase), phase], axis=0)
        return tf.linalg.diag(diag)

    def _prepare(self):
        self.matrix = self.construct_unitary(self.theta)


class CNOT(TensorflowGate, base_gates.CNOT):

    def __init__(self, q0, q1):
        base_gates.CNOT.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    @staticmethod
    def construct_unitary() -> tf.Tensor:
        return tf.cast([[1, 0, 0, 0], [0, 1, 0, 0],
                        [0, 0, 0, 1], [0, 0, 1, 0]], dtype=DTYPES.get('DTYPECPX'))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False):
        return X.__call__(self, state, is_density_matrix)


class CZ(TensorflowGate, base_gates.CZ):

    def __init__(self, q0, q1):
        base_gates.CZ.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    @staticmethod
    def construct_unitary() -> tf.Tensor:
        diag = tf.cast(tf.concat([tf.ones(3), [-1]], axis=0), dtype=DTYPES.get('DTYPECPX'))
        return tf.linalg.diag(diag)

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False):
        return Z.__call__(self, state, is_density_matrix)


class CZPow(MatrixGate, base_gates.CZPow):

    def __init__(self, q0, q1, theta):
        base_gates.CZPow.__init__(self, q0, q1, theta)

    def _prepare(self):
        self.matrix = tf.exp(1j * tf.cast(self.theta, dtype=DTYPES.get('DTYPECPX')))

    @staticmethod
    def construct_unitary(theta) -> tf.Tensor:
        phase = tf.exp(1j * tf.cast(theta, dtype=DTYPES.get('DTYPECPX')))
        diag = tf.concat([tf.ones(3, dtype=DTYPES.get('DTYPECPX')), [phase]], axis=0)
        return tf.linalg.diag(diag)

    def __call__(self, state, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_zpow(state, self.matrix, self.nqubits,
                             self.target_qubits[0], self.control_qubits)


class SWAP(TensorflowGate, base_gates.SWAP):

    def __init__(self, q0, q1):
        base_gates.SWAP.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    @staticmethod
    def construct_unitary() -> tf.Tensor:
        return tf.cast([[1, 0, 0, 0], [0, 0, 1, 0],
                        [0, 1, 0, 0], [0, 0, 0, 1]], dtype=DTYPES.get('DTYPECPX'))

    def __call__(self, state, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_swap(state, self.nqubits, self.target_qubits,
                             self.control_qubits)


class fSim(MatrixGate, base_gates.fSim):

    def __init__(self, q0, q1, theta, phi):
        base_gates.fSim.__init__(self, q0, q1, theta, phi)

    def _prepare(self):
        dtype = DTYPES.get('DTYPECPX')
        th = tf.cast(self.theta, dtype=dtype)
        I = tf.eye(2, dtype=dtype)
        X = tf.cast([[0, 1], [1, 0]], dtype=dtype)
        rotation = tf.cos(th) * I - 1j * tf.sin(th) * X
        phase = tf.exp(-1j * tf.cast(self.phi, dtype=dtype))
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

    def _prepare(self):
        rotation = tf.cast(self.unitary, dtype=DTYPES.get('DTYPECPX'))
        phase = tf.exp(-1j * tf.cast(self.phi, dtype=DTYPES.get('DTYPECPX')))
        rotation = tf.reshape(rotation, (4,))
        self.matrix = tf.concat([tf.reshape(rotation, (4,)), [phase]], axis=0)

    def __call__(self, state, is_density_matrix: bool = False):
        return fSim.__call__(self, state, is_density_matrix)


class TOFFOLI(TensorflowGate, base_gates.TOFFOLI):

    def __init__(self, q0, q1, q2):
        base_gates.TOFFOLI.__init__(self, q0, q1, q2)
        TensorflowGate.__init__(self)

    @staticmethod
    def construct_unitary() -> tf.Tensor:
        return tf.cast([[1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1, 0]], dtype=DTYPES.get('DTYPECPX'))

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

    @property
    def rank(self) -> int:
        return len(self.target_qubits)

    @staticmethod
    def construct_unitary(unitary) -> tf.Tensor:
        return tf.convert_to_tensor(unitary, dtype=DTYPES.get('DTYPECPX'))

    def _prepare(self):
        self.matrix = self.construct_unitary(self.unitary)

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

    def __init__(self, qubit_pairs: List[Tuple[int, int]],
                 one_qubit_gate, two_qubit_gate,
                 params_map: Dict[int, float],
                 params_map2: Optional[Dict[int, float]] = None,
                 name: Optional[str] = None):
        base_gates.VariationalLayer.__init__(self, qubit_pairs,
                                             one_qubit_gate, two_qubit_gate,
                                             params_map, params_map2,
                                             name=name)
        MatrixGate.__init__(self)
        self.additional_matrix = None

    @staticmethod
    def _tfkron(m1, m2):
        m = tf.transpose(tf.tensordot(m1, m2, axes=0), [0, 2, 1, 3])
        return tf.reshape(m, (4, 4))

    def _prepare(self):
        self.matrix = tf.stack([self._tfkron(
            self.one_qubit_gate.construct_unitary(self.params_map[q1]),
            self.one_qubit_gate.construct_unitary(self.params_map[q2]))
                             for q1, q2 in self.qubit_pairs], axis=0)
        entangling_matrix = self.two_qubit_gate.construct_unitary()
        self.matrix = tf.matmul(entangling_matrix, self.matrix)
        if self.additional_target is not None:
            self.additional_matrix = self.one_qubit_gate.construct_unitary(
                self.params_map[self.additional_target])

        if self.params_map2 is not None:
            matrix2 = tf.stack([self._tfkron(
                self.one_qubit_gate.construct_unitary(self.params_map2[q1]),
                self.one_qubit_gate.construct_unitary(self.params_map2[q2]))
                                for q1, q2 in self.qubit_pairs], axis=0)
            self.matrix = tf.matmul(matrix2, self.matrix)
            if self.additional_target is not None:
                self.additional_matrix = tf.matmul(
                    self.one_qubit_gate.construct_unitary(
                        self.params_map2[self.additional_target]),
                    self.additional_matrix)

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        for i, targets in enumerate(self.qubit_pairs):
            state = op.apply_twoqubit_gate(state, self.matrix[i], self.nqubits, targets)
        if self.additional_matrix is not None:
            state = op.apply_gate(state, self.additional_matrix, self.nqubits,
                                  self.additional_target)
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
