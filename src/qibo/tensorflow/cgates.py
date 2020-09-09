# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import gates as base_gates
from qibo.config import BACKEND, DTYPES, DEVICES, raise_error
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
            raise_error(NotImplementedError, "Custom operator gates should not be "
                                      "used in compiled mode.")

    def __matmul__(self, other: "TensorflowGate") -> "TensorflowGate":
        gate = base_gates.Gate.__matmul__(self, other)
        if gate is None:
            gate = Unitary(self.unitary @ other.unitary, *self.qubits)
        return gate

    @staticmethod
    def control_unitary(unitary: tf.Tensor) -> tf.Tensor:
        shape = tuple(unitary.shape)
        if shape != (2, 2):
            raise_error(ValueError, "Cannot use ``control_unitary`` method for "
                                    "input matrix of shape {}.".format(shape))
        dtype = DTYPES.get('DTYPECPX')
        zeros = tf.zeros((2, 2), dtype=dtype)
        part1 = tf.concat([tf.eye(2, dtype=dtype), zeros], axis=0)
        part2 = tf.concat([zeros, unitary], axis=0)
        return tf.concat([part1, part2], axis=1)

    def _calculate_qubits_tensor(self) -> tf.Tensor:
        """Calculates ``qubits`` tensor required for applying gates using custom operators."""
        qubits = list(self.nqubits - np.array(self.control_qubits) - 1)
        qubits.extend(self.nqubits - np.array(self.target_qubits) - 1)
        qubits = sorted(qubits)
        with tf.device(self.device):
            self.qubits_tensor = tf.convert_to_tensor(qubits, dtype=tf.int32)

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
        with tf.device(self.device):
            self.matrix = tf.constant(self.construct_unitary(),
                                      dtype=DTYPES.get('DTYPECPX'))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        super(MatrixGate, self).__call__(state, is_density_matrix)
        return op.apply_gate(state, self.matrix, self.qubits_tensor,
                             self.nqubits, self.target_qubits[0])


class H(MatrixGate, base_gates.H):

    def __init__(self, q):
        base_gates.H.__init__(self, q)
        MatrixGate.__init__(self)

    def construct_unitary(self) -> np.ndarray:
        return (np.array([[1, 1], [1, -1]], dtype=DTYPES.get('NPTYPECPX'))
                / np.sqrt(2))


class X(TensorflowGate, base_gates.X):

    def __init__(self, q):
        base_gates.X.__init__(self, q)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> np.ndarray:
        return np.array([[0, 1], [1, 0]], dtype=DTYPES.get('NPTYPECPX'))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_x(state, self.qubits_tensor, self.nqubits,
                          self.target_qubits[0])


class Y(TensorflowGate, base_gates.Y):

    def __init__(self, q):
        base_gates.Y.__init__(self, q)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> np.ndarray:
        return np.array([[0, -1j], [1j, 0]], dtype=DTYPES.get('NPTYPECPX'))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_y(state, self.qubits_tensor, self.nqubits,
                          self.target_qubits[0])


class Z(TensorflowGate, base_gates.Z):

    def __init__(self, q):
        base_gates.Z.__init__(self, q)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> np.ndarray:
        return np.array([[1, 0], [0, -1]], dtype=DTYPES.get('NPTYPECPX'))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_z(state, self.qubits_tensor, self.nqubits,
                          self.target_qubits[0])


class I(TensorflowGate, base_gates.I):

    def __init__(self, *q):
        base_gates.I.__init__(self, *q)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> np.ndarray:
        dim = 2 ** len(self.target_qubits)
        return np.eye(dim, dtype=DTYPES.get('NPTYPECPX'))

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

    def _get_cpu(self): # pragma: no cover
        # case not covered by GitHub workflows because it requires OOM
        if not DEVICES['CPU']:
            raise_error(RuntimeError, "Cannot find CPU device to use for sampling.")
        return DEVICES['CPU'][0]

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

        device = DEVICES['DEFAULT']
        if np.log2(nshots) + len(self.target_qubits) > 31: # pragma: no cover
            # case not covered by GitHub workflows because it requires large example
            # Use CPU to avoid "aborted" error
            device = self._get_cpu()

        oom_error = tf.python.framework.errors_impl.ResourceExhaustedError
        try:
            with tf.device(device):
                samples_dec = sample()
        except oom_error: # pragma: no cover
            # case not covered by GitHub workflows because it requires OOM
            # Force using CPU to perform sampling
            device = self._get_cpu()
            with tf.device(device):
                samples_dec = sample()

        if samples_only:
            return samples_dec
        return self.measurements.GateResult(
            self.qubits, state, decimal_samples=samples_dec)


class RX(MatrixGate, base_gates.RX):

    def __init__(self, q, theta):
        base_gates.RX.__init__(self, q, theta)
        MatrixGate.__init__(self)

    def construct_unitary(self) -> np.ndarray:
        cos, isin = np.cos(self.parameter / 2.0), -1j * np.sin(self.parameter / 2.0)
        return np.array([[cos, isin], [isin, cos]], dtype=DTYPES.get('NPTYPECPX'))


class RY(MatrixGate, base_gates.RY):

    def __init__(self, q, theta):
        base_gates.RY.__init__(self, q, theta)
        MatrixGate.__init__(self)

    def construct_unitary(self) -> np.ndarray:
        cos, sin = np.cos(self.parameter / 2.0), np.sin(self.parameter / 2.0)
        return np.array([[cos, -sin], [sin, cos]], dtype=DTYPES.get('NPTYPECPX'))


class RZ(MatrixGate, base_gates.RZ):

    def __init__(self, q, theta):
        base_gates.RZ.__init__(self, q, theta)
        MatrixGate.__init__(self)

    def construct_unitary(self) -> np.ndarray:
        phase = np.exp(1j * self.parameter / 2.0)
        return np.diag([phase.conj(), phase]).astype(DTYPES.get('NPTYPECPX'))


class ZPow(MatrixGate, base_gates.ZPow):

    def __init__(self, q, theta):
        base_gates.ZPow.__init__(self, q, theta)
        MatrixGate.__init__(self)

    def _prepare(self):
        with tf.device(self.device):
            self.matrix = tf.constant(np.exp(1j * self.parameter),
                                      dtype=DTYPES.get('DTYPECPX'))

    def construct_unitary(self) -> np.ndarray:
        return np.diag([1, np.exp(1j * self.parameter)]).astype(
            DTYPES.get('NPTYPECPX'))

    def __call__(self, state, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_z_pow(state, self.matrix, self.qubits_tensor,
                              self.nqubits, self.target_qubits[0])


class CNOT(TensorflowGate, base_gates.CNOT):

    def __init__(self, q0, q1):
        base_gates.CNOT.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                         [0, 0, 0, 1], [0, 0, 1, 0]],
                        dtype=DTYPES.get('NPTYPECPX'))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False):
        return X.__call__(self, state, is_density_matrix)


class CZ(TensorflowGate, base_gates.CZ):

    def __init__(self, q0, q1):
        base_gates.CZ.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> np.ndarray:
        return np.diag([1, 1, 1, -1]).astype(DTYPES.get('NPTYPECPX'))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False):
        return Z.__call__(self, state, is_density_matrix)


class CZPow(MatrixGate, base_gates.CZPow):

    def __init__(self, q0, q1, theta):
        base_gates.CZPow.__init__(self, q0, q1, theta)
        MatrixGate.__init__(self)

    def _prepare(self):
        ZPow._prepare(self)

    def construct_unitary(self) -> np.ndarray:
        return np.diag([1, 1, 1, np.exp(1j * self.parameter)]).astype(
            DTYPES.get('NPTYPECPX'))

    def __call__(self, state, is_density_matrix: bool = False):
        return ZPow.__call__(self, state, is_density_matrix)


class SWAP(TensorflowGate, base_gates.SWAP):

    def __init__(self, q0, q1):
        base_gates.SWAP.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                         [0, 1, 0, 0], [0, 0, 0, 1]],
                        dtype=DTYPES.get('NPTYPECPX'))

    def __call__(self, state, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_swap(state, self.qubits_tensor, self.nqubits,
                             *self.target_qubits)


class fSim(MatrixGate, base_gates.fSim):

    def __init__(self, q0, q1, theta, phi):
        base_gates.fSim.__init__(self, q0, q1, theta, phi)
        MatrixGate.__init__(self)

    def _prepare(self):
        theta, phi = self.parameter
        cos, isin = np.cos(theta), -1j * np.sin(theta)
        phase = np.exp(-1j * phi)
        matrix = np.array([cos, isin, isin, cos, phase],
                          dtype=DTYPES.get('NPTYPECPX'))
        with tf.device(self.device):
            self.matrix = tf.constant(matrix, dtype=DTYPES.get('DTYPECPX'))

    def construct_unitary(self) -> np.ndarray:
        theta, phi = self.parameter
        cos, isin = np.cos(theta), -1j * np.sin(theta)
        matrix = np.eye(4, dtype=DTYPES.get('NPTYPECPX'))
        matrix[1, 1], matrix[2, 2] = cos, cos
        matrix[1, 2], matrix[2, 1] = isin, isin
        matrix[3, 3] = np.exp(-1j * phi)
        return matrix

    def __call__(self, state, is_density_matrix: bool = False):
        TensorflowGate.__call__(self, state, is_density_matrix)
        return op.apply_fsim(state, self.matrix, self.qubits_tensor,
                             self.nqubits, *self.target_qubits)


class GeneralizedfSim(MatrixGate, base_gates.GeneralizedfSim):

    def __init__(self, q0, q1, unitary, phi):
        base_gates.GeneralizedfSim.__init__(self, q0, q1, unitary, phi)
        TensorflowGate.__init__(self)

    def _prepare(self):
        unitary, phi = self.parameter
        matrix = np.zeros(5, dtype=DTYPES.get("NPTYPECPX"))
        matrix[:4] = np.reshape(unitary, (4,))
        matrix[4] = np.exp(-1j * phi)
        with tf.device(self.device):
            self.matrix = tf.constant(matrix, dtype=DTYPES.get('DTYPECPX'))

    def construct_unitary(self) -> np.ndarray:
        unitary, phi = self.parameter
        matrix = np.eye(4, dtype=DTYPES.get('NPTYPECPX'))
        matrix[1:3, 1:3] = np.reshape(unitary, (2, 2))
        matrix[3, 3] = np.exp(-1j * phi)
        return matrix

    def __call__(self, state, is_density_matrix: bool = False):
        return fSim.__call__(self, state, is_density_matrix)


class TOFFOLI(TensorflowGate, base_gates.TOFFOLI):

    def __init__(self, q0, q1, q2):
        base_gates.TOFFOLI.__init__(self, q0, q1, q2)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> np.ndarray:
        matrix = np.eye(8, dtype=DTYPES.get('NPTYPECPX'))
        matrix[-2, -2], matrix[-2, -1] = 0, 1
        matrix[-1, -2], matrix[-1, -1] = 1, 0
        return matrix

    def __call__(self, state, is_density_matrix: bool = False):
        return X.__call__(self, state, is_density_matrix)


class Unitary(MatrixGate, base_gates.Unitary):

    def __init__(self, unitary, *q, name: Optional[str] = None):
        if not isinstance(unitary, (np.ndarray, tf.Tensor)):
            raise_error(TypeError, "Unknown type {} of unitary matrix."
                                   "".format(type(unitary)))
        base_gates.Unitary.__init__(self, unitary, *q, name=name)
        MatrixGate.__init__(self)
        rank = self.rank
        if rank > 2:
            raise_error(NotImplementedError, "Unitary matrix gate supports only one "
                                      "qubit gates but {} target qubits were "
                                      "given.".format(len(self.target_qubits)))
        self._unitary = self.construct_unitary()

    def construct_unitary(self) -> np.ndarray:
        unitary = self.parameter
        if isinstance(unitary, np.ndarray):
            return unitary.astype(DTYPES.get('NPTYPECPX'))
        if isinstance(unitary, tf.Tensor):
            return tf.identity(tf.cast(unitary, dtype=DTYPES.get('DTYPECPX')))

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

    def __init__(self, qubits: List[int], pairs: List[Tuple[int, int]],
                 one_qubit_gate, two_qubit_gate,
                 params: List[float], params2: Optional[List[float]] = None,
                 name: Optional[str] = None):
        base_gates.VariationalLayer.__init__(self, qubits, pairs,
                                             one_qubit_gate, two_qubit_gate,
                                             params, params2,
                                             name=name)
        MatrixGate.__init__(self)
        self.unitary_constructor = Unitary

    def _calculate_unitaries(self):
        matrices = np.stack([np.kron(
            self.one_qubit_gate(q1, theta=self.params[q1]).unitary,
            self.one_qubit_gate(q2, theta=self.params[q2]).unitary)
                             for q1, q2 in self.pairs], axis=0)
        entangling_matrix = self.two_qubit_gate(0, 1).unitary
        matrices = entangling_matrix @ matrices

        additional_matrix = None
        q = self.additional_target
        if q is not None:
            additional_matrix = self.one_qubit_gate(
                q, theta=self.params[q]).unitary

        if self.params2:
            matrices2 = np.stack([np.kron(
                self.one_qubit_gate(q1, theta=self.params2[q1]).unitary,
                self.one_qubit_gate(q2, theta=self.params2[q2]).unitary)
                                for q1, q2 in self.pairs], axis=0)
            matrices = matrices2 @ matrices

            q = self.additional_target
            if q is not None:
                _new = self.one_qubit_gate(q, theta=self.params2[q]).unitary
                additional_matrix = _new @ additional_matrix

        return matrices, additional_matrix

    def _prepare(self):
        matrices, additional_matrix = self._calculate_unitaries()
        self.unitaries = [self.unitary_constructor(matrices[i], *targets)
                          for i, targets in enumerate(self.pairs)]
        if additional_matrix is not None:
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
            # future TODO
            raise_error(NotImplementedError, "Density matrices are not supported by "
                                      "custom operator gates.")
        else:
            from qibo.tensorflow import gates
            return getattr(gates, cls.__name__)(*args, **kwargs)


class NoiseChannel(TensorflowChannel, base_gates.NoiseChannel):
    pass


class GeneralChannel(TensorflowChannel, base_gates.NoiseChannel):
    pass
