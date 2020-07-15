# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import gates as base_gates
from qibo.config import BACKEND, DTYPES, DEVICES
from qibo.tensorflow import custom_operators as op
from typing import Dict, List, Optional, Sequence, Tuple


class TensorflowGate(base_gates.Gate):

    import sys
    module = sys.modules[__name__]

    def __new__(cls, *args, **kwargs):
        cgate_only = {"I", "M", "Flatten", "CallbackGate"}
        if BACKEND.get('GATES') == 'custom' or cls.__name__ in cgate_only:
            return super(TensorflowGate, cls).__new__(cls)
        else:
            from qibo.tensorflow import gates
            return getattr(gates, cls.__name__)(*args, **kwargs)

    def __init__(self):
        if not tf.executing_eagerly():
            raise NotImplementedError("Custom operator gates should not be "
                                      "used in compiled mode.")

    def __matmul__(self, other: "TensorflowGate") -> "TensorflowGate":
        gate = base_gates.Gate.__matmul__(self, other)
        if gate is None:
            gate = Unitary(tf.matmul(self.unitary, other.unitary), *self.qubits)
        return gate

    @staticmethod
    def control_unitary(unitary: tf.Tensor) -> tf.Tensor:
        shape = tuple(unitary.shape)
        if shape != (2, 2): # pragma: no cover
            raise ValueError("Cannot use ``control_unitary`` method for input "
                             "matrix of shape {}.".format(shape))
        matrix = tf.eye(4, dtype=DTYPES.get('DTYPECPX'))
        ids = [[2, 2], [2, 3], [3, 2], [3, 3]]
        values = tf.reshape(unitary, (4,))
        return tf.tensor_scatter_nd_update(matrix, ids, values)

    def _calculate_qubits_tensor(self) -> tf.Tensor:
        """Calculates ``qubits`` tensor required for applying gates using custom operators."""
        qubits = list(self.nqubits - np.array(self.control_qubits) - 1)
        qubits.extend(self.nqubits - np.array(self.target_qubits) - 1)
        qubits = sorted(qubits)
        return tf.convert_to_tensor(qubits, dtype=tf.int32)

    def _prepare(self):
        """Prepares the gate for application to state vectors.

        Called automatically by the ``nqubits`` setter.
        Calculates the ``matrix`` required to apply the gate to state vectors.
        This is not necessarily the same as the unitary matrix of the gate.
        """
        pass

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

    def _prepare(self):
        self.matrix = self.construct_unitary()

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        super(MatrixGate, self).__call__(state, is_density_matrix)
        return op.apply_gate(state, self.matrix, self.qubits_tensor,
                             self.nqubits, self.target_qubits[0])


class H(MatrixGate, base_gates.H):

    def __init__(self, q):
        base_gates.H.__init__(self, q)
        MatrixGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        return tf.cast(np.array([[1, 1], [1, -1]]) / np.sqrt(2),
                       dtype=DTYPES.get('DTYPECPX'))


class X(TensorflowGate, base_gates.X):

    def __init__(self, q):
        base_gates.X.__init__(self, q)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        return tf.cast([[0, 1], [1, 0]], dtype=DTYPES.get('DTYPECPX'))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_x(state, self.qubits_tensor, self.nqubits,
                          self.target_qubits[0])


class Y(TensorflowGate, base_gates.Y):

    def __init__(self, q):
        base_gates.Y.__init__(self, q)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        return 1j * tf.cast([[0, -1], [1, 0]], dtype=DTYPES.get('DTYPECPX'))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_y(state, self.qubits_tensor, self.nqubits,
                          self.target_qubits[0])


class Z(TensorflowGate, base_gates.Z):

    def __init__(self, q):
        base_gates.Z.__init__(self, q)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        return tf.cast([[1, 0], [0, -1]], dtype=DTYPES.get('DTYPECPX'))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_z(state, self.qubits_tensor, self.nqubits,
                          self.target_qubits[0])


class I(TensorflowGate, base_gates.I):

    def __init__(self, *q):
        base_gates.I.__init__(self, *q)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        n = tf.cast(2 ** len(self.target_qubits), dtype=DTYPES.get('DTYPEINT'))
        return tf.eye(n, dtype=DTYPES.get('DTYPECPX'))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False):
        return state


class M(TensorflowGate, base_gates.M):
    from qibo.tensorflow import measurements

    def __init__(self, *q, register_name: Optional[str] = None):
        base_gates.M.__init__(self, *q, register_name=register_name)
        self._traceout = None

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
        probs_dim = tf.cast((2 ** len(self.target_qubits),),
                            dtype=DTYPES.get('DTYPEINT'))
        def sample():
            shape = (1 + is_density_matrix) * self.nqubits * (2,)
            probs = self._calculate_probabilities(
                tf.reshape(state, shape), is_density_matrix)
            logits = tf.math.log(tf.reshape(probs, probs_dim))
            return tf.random.categorical(logits[tf.newaxis], nshots,
                                         dtype=DTYPES.get('DTYPEINT'))[0]

        oom_error = tf.python.framework.errors_impl.ResourceExhaustedError
        try:
            samples_dec = sample()
        except oom_error: # pragma: no cover
            # Force using CPU to perform sampling
            if not DEVICES['CPU']:
                raise RuntimeError("Cannot find CPU device to use for sampling.")
            with tf.device(DEVICES['CPU'][0]):
                samples_dec = sample()

        if samples_only:
            return samples_dec
        return self.measurements.GateResult(
            self.qubits, state, decimal_samples=samples_dec)


class ParametrizedMatrixGate(MatrixGate, base_gates.ParametrizedGate):

    @property
    def parameter(self):
        return self._theta

    @parameter.setter
    def parameter(self, x):
        self._theta = x
        self._prepare()


class RX(ParametrizedMatrixGate, base_gates.RX):

    def __init__(self, q, theta):
        base_gates.RX.__init__(self, q, theta)
        ParametrizedMatrixGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        dtype = DTYPES.get('DTYPECPX')
        t = tf.cast(self.parameter, dtype=dtype)
        I = tf.eye(2, dtype=dtype)
        X = tf.cast([[0, 1], [1, 0]], dtype=dtype)
        return tf.cos(t / 2.0) * I - 1j * tf.sin(t / 2.0) * X


class RY(ParametrizedMatrixGate, base_gates.RY):

    def __init__(self, q, theta):
        base_gates.RY.__init__(self, q, theta)
        ParametrizedMatrixGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        dtype = DTYPES.get('DTYPECPX')
        t = tf.cast(self.parameter, dtype=dtype)
        I = tf.eye(2, dtype=dtype)
        iY = tf.cast([[0, 1], [-1, 0]], dtype=dtype)
        return tf.cos(t / 2.0) * I - tf.sin(t / 2.0) * iY


class RZ(ParametrizedMatrixGate, base_gates.RZ):

    def __init__(self, q, theta):
        base_gates.RZ.__init__(self, q, theta)
        ParametrizedMatrixGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        t = tf.cast(self.parameter, dtype=DTYPES.get('DTYPECPX'))
        phase = tf.exp(1j * t / 2.0)[tf.newaxis]
        diag = tf.concat([tf.math.conj(phase), phase], axis=0)
        return tf.linalg.diag(diag)


class ZPow(ParametrizedMatrixGate, base_gates.ZPow):

    def __init__(self, q, theta):
        base_gates.ZPow.__init__(self, q, theta)
        ParametrizedMatrixGate.__init__(self)

    def _prepare(self):
        self.matrix = tf.exp(1j * tf.cast(self.parameter, dtype=DTYPES.get('DTYPECPX')))

    def construct_unitary(self) -> tf.Tensor:
        t = tf.cast(self.parameter, dtype=DTYPES.get('DTYPECPX'))
        phase = tf.exp(1j * t)
        diag = tf.concat([1, phase], axis=0)
        return tf.linalg.diag(diag)

    def __call__(self, state, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_z_pow(state, self.matrix, self.qubits_tensor,
                              self.nqubits, self.target_qubits[0])


class CNOT(TensorflowGate, base_gates.CNOT):

    def __init__(self, q0, q1):
        base_gates.CNOT.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        return tf.cast([[1, 0, 0, 0], [0, 1, 0, 0],
                        [0, 0, 0, 1], [0, 0, 1, 0]], dtype=DTYPES.get('DTYPECPX'))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False):
        return X.__call__(self, state, is_density_matrix)


class CZ(TensorflowGate, base_gates.CZ):

    def __init__(self, q0, q1):
        base_gates.CZ.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        diag = tf.cast(tf.concat([tf.ones(3), [-1]], axis=0), dtype=DTYPES.get('DTYPECPX'))
        return tf.linalg.diag(diag)

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False):
        return Z.__call__(self, state, is_density_matrix)


class CZPow(ParametrizedMatrixGate, base_gates.CZPow):

    def __init__(self, q0, q1, theta):
        base_gates.CZPow.__init__(self, q0, q1, theta)
        ParametrizedMatrixGate.__init__(self)

    def _prepare(self):
        ZPow._prepare(self)

    def construct_unitary(self) -> tf.Tensor:
        dtype = DTYPES.get('DTYPECPX')
        phase = tf.exp(1j * tf.cast(self.parameter, dtype=dtype))
        diag = tf.concat([tf.ones(3, dtype=dtype), [phase]], axis=0)
        return tf.linalg.diag(diag)

    def __call__(self, state, is_density_matrix: bool = False):
        return ZPow.__call__(self, state, is_density_matrix)


class SWAP(TensorflowGate, base_gates.SWAP):

    def __init__(self, q0, q1):
        base_gates.SWAP.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        return tf.cast([[1, 0, 0, 0], [0, 0, 1, 0],
                        [0, 1, 0, 0], [0, 0, 0, 1]],
                       dtype=DTYPES.get('DTYPECPX'))

    def __call__(self, state, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_swap(state, self.qubits_tensor, self.nqubits,
                             *self.target_qubits)


class fSim(MatrixGate, base_gates.fSim):

    def __init__(self, q0, q1, theta, phi):
        base_gates.fSim.__init__(self, q0, q1, theta, phi)
        TensorflowGate.__init__(self)

    def _prepare(self):
        dtype = DTYPES.get('DTYPECPX')
        th = tf.cast(self.theta, dtype=dtype)
        I = tf.eye(2, dtype=dtype)
        X = tf.cast([[0, 1], [1, 0]], dtype=dtype)
        rotation = tf.cos(th) * I - 1j * tf.sin(th) * X
        phase = tf.exp(-1j * tf.cast(self.phi, dtype=dtype))
        self.matrix = tf.concat([tf.reshape(rotation, (4,)), [phase]], axis=0)

    def construct_unitary(self):
        dtype = DTYPES.get("DTYPECPX")
        th = tf.cast(self.theta, dtype=dtype)
        eyemat = tf.eye(2, dtype=dtype)
        xmat = tf.cast([[0, 1], [1, 0]], dtype=dtype)
        rotation = tf.cos(th) * eyemat - 1j * tf.sin(th) * xmat
        phase = tf.exp(-1j * tf.cast(self.phi, dtype=dtype))
        matrix = tf.eye(4, dtype=dtype)
        matrix = tf.tensor_scatter_nd_update(matrix, [[3, 3]], [phase])
        rotation = tf.reshape(rotation, (4,))
        ids = [[1, 1], [1, 2], [2, 1], [2, 2]]
        return tf.tensor_scatter_nd_update(matrix, ids, rotation)

    def __call__(self, state, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_fsim(state, self.matrix, self.qubits_tensor,
                             self.nqubits, *self.target_qubits)


class GeneralizedfSim(MatrixGate, base_gates.GeneralizedfSim):

    def __init__(self, q0, q1, unitary, phi):
        base_gates.GeneralizedfSim.__init__(self, q0, q1, unitary, phi)
        TensorflowGate.__init__(self)
        shape = tuple(self.given_unitary.shape)
        if shape != (2, 2):
            raise ValueError("Invalid shape {} of rotation for generalized "
                             "fSim gate".format(shape))

    def _prepare(self):
        dtype = DTYPES.get('DTYPECPX')
        rotation = tf.cast(self.given_unitary, dtype=dtype)
        phase = tf.exp(-1j * tf.cast(self.phi, dtype=dtype))
        rotation = tf.reshape(rotation, (4,))
        self.matrix = tf.concat([tf.reshape(rotation, (4,)), [phase]], axis=0)

    def construct_unitary(self):
        dtype = DTYPES.get("DTYPECPX")
        rotation = tf.cast(self.given_unitary, dtype=dtype)
        phase = tf.exp(-1j * tf.cast(self.phi, dtype=dtype))
        matrix = tf.eye(4, dtype=dtype)
        matrix = tf.tensor_scatter_nd_update(matrix, [[3, 3]], [phase])
        rotation = tf.reshape(rotation, (4,))
        ids = [[1, 1], [1, 2], [2, 1], [2, 2]]
        return tf.tensor_scatter_nd_update(matrix, ids, rotation)

    def __call__(self, state, is_density_matrix: bool = False):
        return fSim.__call__(self, state, is_density_matrix)


class TOFFOLI(TensorflowGate, base_gates.TOFFOLI):

    def __init__(self, q0, q1, q2):
        base_gates.TOFFOLI.__init__(self, q0, q1, q2)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
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

        shape = tuple(self.given_unitary.shape)
        if shape != (2 ** rank, 2 ** rank):
            raise ValueError("Invalid shape {} of unitary matrix acting on "
                             "{} target qubits.".format(shape, rank))

        self._unitary = self.construct_unitary()

    @property
    def rank(self) -> int:
        return len(self.target_qubits)

    def construct_unitary(self) -> tf.Tensor:
        unitary = self.given_unitary
        if isinstance(unitary, tf.Tensor):
            return tf.identity(tf.cast(unitary, dtype=DTYPES.get('DTYPECPX')))
        elif isinstance(unitary, np.ndarray):
            return tf.convert_to_tensor(unitary, dtype=DTYPES.get('DTYPECPX'))
        raise TypeError("Unknown type {} of unitary matrix".format(type(unitary)))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        TensorflowGate.__call__(self, state, is_density_matrix)
        if self.rank == 1:
            return op.apply_gate(state, self.matrix, self.qubits_tensor,
                                 self.nqubits, self.target_qubits[0])
        if self.rank == 2:
            return op.apply_two_qubit_gate(state, self.matrix, self.qubits_tensor,
                                           self.nqubits, *self.target_qubits)


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
        self.unitary_constructor = Unitary

    @staticmethod
    def _tfkron(m1, m2):
        m = tf.transpose(tf.tensordot(m1, m2, axes=0), [0, 2, 1, 3])
        return tf.reshape(m, (4, 4))

    def _prepare(self):
        matrices = tf.stack([self._tfkron(
            self.one_qubit_gate(q1, theta=self.params_map[q1]).unitary,
            self.one_qubit_gate(q2, theta=self.params_map[q2]).unitary)
                             for q1, q2 in self.qubit_pairs], axis=0)
        entangling_matrix = self.two_qubit_gate(0, 1).unitary
        matrices = tf.matmul(entangling_matrix, matrices)

        q = self.additional_target
        if q is not None:
            additional_matrix = self.one_qubit_gate(
                q, theta=self.params_map[q]).unitary

        if self.params_map2 is not None:
            matrices2 = tf.stack([self._tfkron(
                self.one_qubit_gate(q1, theta=self.params_map2[q1]).unitary,
                self.one_qubit_gate(q2, theta=self.params_map2[q2]).unitary)
                                for q1, q2 in self.qubit_pairs], axis=0)
            matrices = tf.matmul(matrices2, matrices)

            q = self.additional_target
            if q is not None:
                additional_matrix = tf.matmul(
                    self.one_qubit_gate(q, theta=self.params_map2[q]).unitary,
                    additional_matrix)

        self.unitaries = [self.unitary_constructor(matrices[i], *targets)
                          for i, targets in enumerate(self.qubit_pairs)]
        if self.additional_target is not None: # pragma: no cover
            self.additional_unitary = self.unitary_constructor(
                additional_matrix, self.additional_target)

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        TensorflowGate.__call__(self, state, is_density_matrix)
        for i, unitary in enumerate(self.unitaries):
            state = unitary(state, is_density_matrix)
        if self.additional_unitary is not None:
            state = self.additional_unitary(state, is_density_matrix)
        return state


class Flatten(TensorflowGate, base_gates.Flatten):

    def __init__(self, coefficients):
        base_gates.Flatten.__init__(self, coefficients)
        TensorflowGate.__init__(self)
        self.swap_reset = []

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        shape = tuple(state.shape)
        if self._nqubits is None:
            if is_density_matrix:
                self.nqubits = len(shape) // 2
            else:
                self.nqubits = len(shape)
        _state = np.array(self.coefficients).reshape(shape)
        return tf.convert_to_tensor(_state, dtype=DTYPES.get("DTYPECPX"))


class CallbackGate(TensorflowGate, base_gates.CallbackGate):

    def __init__(self, callback):
        base_gates.CallbackGate.__init__(self, callback)
        TensorflowGate.__init__(self)
        self.swap_reset = []

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        TensorflowGate.__call__(self, state, is_density_matrix)
        self.callback.append(self.callback(state, is_density_matrix))
        return state


# Density matrices are not supported by custom operators yet so channels fall
# back to native tensorflow gates
class TensorflowChannel(TensorflowGate):

    def __new__(cls, *args, **kwargs):
        if BACKEND.get('GATES') == 'custom': # pragma: no cover
            raise NotImplementedError("Density matrices are not supported by "
                                      "custom operator gates.")
        else:
            from qibo.tensorflow import gates
            return getattr(gates, cls.__name__)(*args, **kwargs)


class NoiseChannel(TensorflowChannel, base_gates.NoiseChannel):
    pass


class GeneralChannel(TensorflowChannel, base_gates.NoiseChannel):
    pass
