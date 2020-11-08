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
        cgate_only = {"I", "M", "Flatten", "CallbackGate", "ZPow", "CZPow",
                      "ProbabilisticNoiseChannel"}
        if BACKEND.get('GATES') == 'custom' or cls.__name__ in cgate_only:
            return super(TensorflowGate, cls).__new__(cls)
        else:
            from qibo.tensorflow import gates
            return getattr(gates, cls.__name__)(*args, **kwargs)

    def __init__(self):
        if not tf.executing_eagerly():
            raise_error(NotImplementedError,
                        "Custom operator gates should not be used in compiled "
                        "mode.")
        self.gate_op = op.apply_gate
        self._density_matrix = False
        self._active_call = "_state_vector_call"

        self.qubits_tensor = None
        self.qubits_tensor_dm = None
        self.target_qubits_dm = None

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
            if self.density_matrix:
                self.target_qubits_dm = tuple(np.array(self.target_qubits) +
                                              self.nqubits)
                self.qubits_tensor_dm = self.qubits_tensor + self.nqubits

    def _prepare(self):
        """Prepares the gate for application to state vectors.

        Called automatically by the ``nqubits`` setter.
        Calculates the ``matrix`` required to apply the gate to state vectors.
        This is not necessarily the same as the unitary matrix of the gate.
        """
        pass

    def _set_nqubits(self, state: tf.Tensor):
        """Sets ``gate.nqubits`` from state, if not already set."""
        if self._nqubits is None:
            self.nqubits = int(np.log2(tuple(state.shape)[0]))

    def _state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        return self.gate_op(state, self.qubits_tensor, self.nqubits,
                            *self.target_qubits)

    def _density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        state = self.gate_op(state, self.qubits_tensor_dm, 2 * self.nqubits,
                             *self.target_qubits)
        state = self.gate_op(state, self.qubits_tensor, 2 * self.nqubits,
                             *self.target_qubits_dm)
        return state

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        """Implements the `Gate` on a given state.

        Args:
            state (tf.Tensor): State vector with shape (2 ** nqubits,).
        """
        self._set_nqubits(state)
        return getattr(self, self._active_call)(state)


class MatrixGate(TensorflowGate):
    """``TensorflowGate`` that uses matrix to be applied to states."""

    def __init__(self):
        super(MatrixGate, self).__init__()
        self.matrix = None

    def _prepare(self):
        with tf.device(self.device):
            self.matrix = tf.constant(self.construct_unitary(),
                                      dtype=DTYPES.get('DTYPECPX'))

    def _state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
       return self.gate_op(state, self.matrix, self.qubits_tensor,
                           self.nqubits, *self.target_qubits)

    def _density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        state = self.gate_op(state, self.matrix, self.qubits_tensor_dm,
                             2 * self.nqubits, *self.target_qubits)
        adjmatrix = tf.math.conj(self.matrix)
        state = self.gate_op(state, adjmatrix, self.qubits_tensor,
                             2 * self.nqubits, *self.target_qubits_dm)
        return state


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
        self.gate_op = op.apply_x

    def construct_unitary(self) -> np.ndarray:
        return np.array([[0, 1], [1, 0]], dtype=DTYPES.get('NPTYPECPX'))


class Y(TensorflowGate, base_gates.Y):

    def __init__(self, q):
        base_gates.Y.__init__(self, q)
        TensorflowGate.__init__(self)
        self.gate_op = op.apply_y

    def construct_unitary(self) -> np.ndarray:
        return np.array([[0, -1j], [1j, 0]], dtype=DTYPES.get('NPTYPECPX'))

    def _density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        return -TensorflowGate._density_matrix_call(self, state)


class Z(TensorflowGate, base_gates.Z):

    def __init__(self, q):
        base_gates.Z.__init__(self, q)
        TensorflowGate.__init__(self)
        self.gate_op = op.apply_z

    def construct_unitary(self) -> np.ndarray:
        return np.array([[1, 0], [0, -1]], dtype=DTYPES.get('NPTYPECPX'))


class I(TensorflowGate, base_gates.I):

    def __init__(self, *q):
        base_gates.I.__init__(self, *q)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> np.ndarray:
        dim = 2 ** len(self.target_qubits)
        return np.eye(dim, dtype=DTYPES.get('NPTYPECPX'))

    def __call__(self, state: tf.Tensor):
        return state


class Collapse(TensorflowGate, base_gates.Collapse):

    def __init__(self, *q: int, result: List[int] = 0):
        base_gates.Collapse.__init__(self, *q, result=result)
        TensorflowGate.__init__(self)
        self.result_tensor = None
        self.gate_op = op.collapse_state

    @staticmethod
    def _result_to_list(res):
        if isinstance(res, np.ndarray):
            return list(res.astype(np.int))
        if isinstance(res, tf.Tensor):
            return list(res.numpy().astype(np.int))
        return list(res)

    def _prepare(self):
        n = len(self.result)
        result = sum(2 ** (n - i - 1) * r for i, r in enumerate(self.result))
        self.result_tensor = tf.cast(result, dtype=DTYPES.get('DTYPEINT'))

    def _state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        return self.gate_op(state, self.qubits_tensor, self.result_tensor,
                            self.nqubits, self.normalize)

    def _density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        state = self.gate_op(state, self.qubits_tensor_dm, self.result_tensor,
                             2 * self.nqubits, False)
        state = self.gate_op(state, self.qubits_tensor, self.result_tensor,
                             2 * self.nqubits, False)
        return state / tf.linalg.trace(state)


class M(TensorflowGate, base_gates.M):
    from qibo.tensorflow import distutils
    from qibo.tensorflow import measurements

    def __init__(self, *q, register_name: Optional[str] = None,
                 p0: Optional["ProbsType"] = None,
                 p1: Optional["ProbsType"] = None):
        base_gates.M.__init__(self, *q, register_name=register_name,
                              p0=p0, p1=p1)
        self.qubits_tensor = None
        self._density_matrix = False
        self._traceout = None

    def _calculate_probabilities_dm(self, state: tf.Tensor) -> tf.Tensor:
        if self._traceout is None:
            from qibo.tensorflow.einsum import DefaultEinsum
            qubits = set(self.unmeasured_qubits)
            # TODO: Remove ``DefaultEinsum`` dependence here
            self._traceout = DefaultEinsum.partialtrace_str(
              qubits, self.nqubits, measuring=True)
        return tf.cast(tf.einsum(self._traceout, state),
                       dtype=DTYPES.get('DTYPE'))

    def _calculate_probabilities(self, state: tf.Tensor) -> tf.Tensor:
        """Calculates probabilities from state using Born's rule.

        Args:
            state: State vector of shape nqubits * (2,) or density matrix of
                shape 2 * nqubits * (2,).

        Returns:
            Probabilities for measured qubits with shape len(target_qubits)* (2,).
        """
        # Trace out unmeasured qubits
        if self.density_matrix:
            probs = self._calculate_probabilities_dm(state)
        else:
            probs = tf.reduce_sum(tf.square(tf.abs(state)),
                                  axis=self.unmeasured_qubits)
        # Bring probs in the order specified by the user
        return tf.transpose(probs, perm=self.reduced_target_qubits)

    def _sample(self, state: tf.Tensor, nshots: int) -> tf.Tensor:
        dtype = DTYPES.get('DTYPEINT')
        probs_dim = tf.cast((2 ** len(self.target_qubits),), dtype=dtype)
        shape = (1 + self.density_matrix) * self.nqubits * (2,)
        probs = self._calculate_probabilities(tf.reshape(state, shape))
        logits = tf.math.log(tf.reshape(probs, probs_dim))[tf.newaxis]
        samples_dec = tf.random.categorical(logits, nshots, dtype=dtype)[0]
        result = self.measurements.GateResult(
            self.qubits, decimal_samples=samples_dec)
        # optional bitflip noise
        if sum(sum(x.values()) for x in self.bitflip_map) > 0:
            result = result.apply_bitflips(*self.bitflip_map)
        return result

    def _get_cpu(self): # pragma: no cover
        # case not covered by GitHub workflows because it requires OOM
        if not DEVICES['CPU']:
            raise_error(RuntimeError, "Cannot find CPU device to use for sampling.")
        return DEVICES['CPU'][0]

    def __call__(self, state: tf.Tensor, nshots: int) -> tf.Tensor:
        if isinstance(state, self.distutils.DistributedState):
            with tf.device(state.device):
                state = state.vector
        TensorflowGate._set_nqubits(self, state)
        if np.log2(nshots) + len(self.target_qubits) > 31: # pragma: no cover
            # case not covered by GitHub workflows because it requires large example
            # Use CPU to avoid "aborted" error
            device = self._get_cpu()

        oom_error = tf.python.framework.errors_impl.ResourceExhaustedError
        try:
            with tf.device(self.device):
                result = self._sample(state, nshots)
        except oom_error: # pragma: no cover
            # case not covered by GitHub workflows because it requires OOM
            # Force using CPU to perform sampling
            device = self._get_cpu()
            with tf.device(device):
                result = self._sample(state, nshots)
        return result


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


class U1(MatrixGate, base_gates.U1):

    def __init__(self, q, theta):
        base_gates.U1.__init__(self, q, theta)
        MatrixGate.__init__(self)
        self.gate_op = op.apply_z_pow

    def _prepare(self):
        with tf.device(self.device):
            self.matrix = tf.constant(np.exp(1j * self.parameter),
                                      dtype=DTYPES.get('DTYPECPX'))

    def construct_unitary(self) -> np.ndarray:
        return np.diag([1, np.exp(1j * self.parameter)]).astype(
            DTYPES.get('NPTYPECPX'))


class U2(MatrixGate, base_gates.U2):

    def __init__(self, q, phi, lam):
        base_gates.U2.__init__(self, q, phi, lam)
        MatrixGate.__init__(self)

    def construct_unitary(self) -> np.ndarray:
        eplus = np.exp(1j * (self._phi + self._lam) / 2.0)
        eminus = np.exp(1j * (self._phi - self._lam) / 2.0)
        return np.array([[eplus.conj(), - eminus.conj()],
                         [eminus, eplus]],
                        dtype=DTYPES.get('NPTYPECPX')) / np.sqrt(2)


class U3(MatrixGate, base_gates.U3):

    def __init__(self, q, theta, phi, lam):
        base_gates.U3.__init__(self, q, theta, phi, lam)
        MatrixGate.__init__(self)

    def construct_unitary(self) -> np.ndarray:
        cost = np.cos(self._theta / 2)
        sint = np.sin(self._theta / 2)
        eplus = np.exp(1j * (self._phi + self._lam) / 2.0)
        eminus = np.exp(1j * (self._phi - self._lam) / 2.0)
        return np.array([[eplus.conj() * cost, - eminus.conj() * sint],
                         [eminus * sint, eplus * cost]],
                        dtype=DTYPES.get('NPTYPECPX'))


class ZPow(MatrixGate, base_gates.ZPow):

  def __new__(cls, q, theta):
      if BACKEND.get('GATES') == 'custom':
          return U1(q, theta)
      else:
          from qibo.tensorflow import gates
          return gates.U1(q, theta)


class CNOT(TensorflowGate, base_gates.CNOT):

    def __init__(self, q0, q1):
        base_gates.CNOT.__init__(self, q0, q1)
        TensorflowGate.__init__(self)
        self.gate_op = op.apply_x

    def construct_unitary(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                         [0, 0, 0, 1], [0, 0, 1, 0]],
                        dtype=DTYPES.get('NPTYPECPX'))


class CZ(TensorflowGate, base_gates.CZ):

    def __init__(self, q0, q1):
        base_gates.CZ.__init__(self, q0, q1)
        TensorflowGate.__init__(self)
        self.gate_op = op.apply_z

    def construct_unitary(self) -> np.ndarray:
        return np.diag([1, 1, 1, -1]).astype(DTYPES.get('NPTYPECPX'))


class _CUn_(MatrixGate):
    base = U1

    def __init__(self, q0, q1, **params):
        MatrixGate.__init__(self)
        cbase = "C{}".format(self.base.__name__)
        getattr(base_gates, cbase).__init__(self, q0, q1, **params)

    def _prepare(self):
        with tf.device(self.device):
            self.matrix = tf.constant(self.base.construct_unitary(self),
                                      dtype=DTYPES.get('DTYPECPX'))

    def construct_unitary(self) -> tf.Tensor:
        return MatrixGate.control_unitary(self.base.construct_unitary(self))

    def __call__(self, state):
        return self.base.__call__(self, state)


class CRX(_CUn_, base_gates.CRX):
    base = RX

    def __init__(self, q0, q1, theta):
        _CUn_.__init__(self, q0, q1, theta=theta)


class CRY(_CUn_, base_gates.CRY):
    base = RY

    def __init__(self, q0, q1, theta):
        _CUn_.__init__(self, q0, q1, theta=theta)


class CRZ(_CUn_, base_gates.CRZ):
    base = RZ

    def __init__(self, q0, q1, theta):
        _CUn_.__init__(self, q0, q1, theta=theta)


class CU1(_CUn_, base_gates.CU1):
    base = U1

    def __init__(self, q0, q1, theta):
        _CUn_.__init__(self, q0, q1, theta=theta)
        self.gate_op = op.apply_z_pow

    def _prepare(self):
        U1._prepare(self)


class CU2(_CUn_, base_gates.CU2):
    base = U2

    def __init__(self, q0, q1, phi, lam):
        _CUn_.__init__(self, q0, q1, phi=phi, lam=lam)


class CU3(_CUn_, base_gates.CU3):
    base = U3

    def __init__(self, q0, q1, theta, phi, lam):
        _CUn_.__init__(self, q0, q1, theta=theta, phi=phi, lam=lam)


class CZPow(MatrixGate, base_gates.CZPow):

  def __new__(cls, q0, q1, theta):
      if BACKEND.get('GATES') == 'custom':
          return CU1(q0, q1, theta)
      else:
          from qibo.tensorflow import gates
          return gates.CU1(q0, q1, theta)


class SWAP(TensorflowGate, base_gates.SWAP):

    def __init__(self, q0, q1):
        base_gates.SWAP.__init__(self, q0, q1)
        TensorflowGate.__init__(self)
        self.gate_op = op.apply_swap

    def construct_unitary(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                         [0, 1, 0, 0], [0, 0, 0, 1]],
                        dtype=DTYPES.get('NPTYPECPX'))


class fSim(MatrixGate, base_gates.fSim):

    def __init__(self, q0, q1, theta, phi):
        base_gates.fSim.__init__(self, q0, q1, theta, phi)
        MatrixGate.__init__(self)
        self.gate_op = op.apply_fsim

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


class GeneralizedfSim(MatrixGate, base_gates.GeneralizedfSim):

    def __init__(self, q0, q1, unitary, phi):
        base_gates.GeneralizedfSim.__init__(self, q0, q1, unitary, phi)
        TensorflowGate.__init__(self)
        self.gate_op = op.apply_fsim

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

    def _dagger(self) -> "GenerelizedfSim":
        unitary, phi = self.parameter
        if isinstance(unitary, tf.Tensor):
            ud = tf.math.conj(tf.transpose(unitary))
        else:
            ud = unitary.conj().T
        q0, q1 = self.target_qubits
        return self.__class__(q0, q1, ud, -phi)


class TOFFOLI(TensorflowGate, base_gates.TOFFOLI):

    def __init__(self, q0, q1, q2):
        base_gates.TOFFOLI.__init__(self, q0, q1, q2)
        TensorflowGate.__init__(self)
        self.gate_op = op.apply_x

    def construct_unitary(self) -> np.ndarray:
        matrix = np.eye(8, dtype=DTYPES.get('NPTYPECPX'))
        matrix[-2, -2], matrix[-2, -1] = 0, 1
        matrix[-1, -2], matrix[-1, -1] = 1, 0
        return matrix


class Unitary(MatrixGate, base_gates.Unitary):

    def __init__(self, unitary, *q, name: Optional[str] = None):
        if not isinstance(unitary, (np.ndarray, tf.Tensor)):
            raise_error(TypeError, "Unknown type {} of unitary matrix."
                                   "".format(type(unitary)))
        base_gates.Unitary.__init__(self, unitary, *q, name=name)
        MatrixGate.__init__(self)
        rank = self.rank
        if rank == 1:
            self.gate_op = op.apply_gate
        elif rank == 2:
            self.gate_op = op.apply_two_qubit_gate
        else:
            n = len(self.target_qubits)
            raise_error(NotImplementedError, "Unitary gate supports one or two-"
                                             "qubit gates when using custom "
                                             "operators, but {} target qubits "
                                             "were given. Please switch to a "
                                             "Tensorflow backend to execute "
                                             "this operation.".format(n))
        self._unitary = self.construct_unitary()

    def construct_unitary(self) -> np.ndarray:
        unitary = self.parameter
        if isinstance(unitary, np.ndarray):
            return unitary.astype(DTYPES.get('NPTYPECPX'))
        if isinstance(unitary, tf.Tensor):
            return tf.identity(tf.cast(unitary, dtype=DTYPES.get('DTYPECPX')))

    def _dagger(self) -> "Unitary":
        unitary = self.parameter
        if isinstance(unitary, tf.Tensor):
            ud = tf.math.conj(tf.transpose(unitary))
        else:
            ud = unitary.conj().T
        return self.__class__(ud, *self.target_qubits, **self.init_kwargs)


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

    def _unitary_constructor(self, matrix, *targets):
        gate = Unitary(matrix, *targets)
        gate.density_matrix = self.density_matrix
        return gate

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
        if not self.is_dagger:
            self.unitaries = [self._unitary_constructor(matrices[i], *targets)
                              for i, targets in enumerate(self.pairs)]

            if additional_matrix is not None:
                self.additional_unitary = self._unitary_constructor(
                    additional_matrix, self.additional_target)
                self.additional_unitary.density_matrix = self.density_matrix

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        TensorflowGate._set_nqubits(self, state)
        for i, unitary in enumerate(self.unitaries):
            state = unitary(state)
        if self.additional_unitary is not None:
            state = self.additional_unitary(state)
        return state


class Flatten(TensorflowGate, base_gates.Flatten):

    def __init__(self, coefficients):
        base_gates.Flatten.__init__(self, coefficients)
        TensorflowGate.__init__(self)
        self.swap_reset = []

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        shape = tuple(state.shape)
        if self._nqubits is None:
            if self.density_matrix:
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

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        TensorflowGate._set_nqubits(self, state)
        self.callback.append(self.callback(state, self.density_matrix))
        return state


class ProbabilisticNoiseChannel(TensorflowGate, base_gates.ProbabilisticNoiseChannel):

    def __init__(self, q, px=0, py=0, pz=0, seed=None):
        base_gates.ProbabilisticNoiseChannel.__init__(self, q, px, py, pz, seed)
        TensorflowGate.__init__(self)

    def _prepare(self):
        self._create_gates()
        if self.seed is not None:
            np.random.seed(self.seed)

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        TensorflowGate._set_nqubits(self, state)
        for p, gate in self.gates:
            if np.random.random() < p:
                state = gate(state)
        return state


class TensorflowChannel(TensorflowGate):
    """Base Tensorflow channel.

    All channels should inherit this class.
    """

    def __init__(self):
        super(TensorflowChannel, self).__init__()

    def _prepare(self):
        if not self.density_matrix:
            raise_error(ValueError, "Channels cannot be used on state vectors.")
        base_cls = getattr(base_gates, self.__class__.__name__)
        base_cls._create_gates(self)

    def _state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        raise_error(ValueError, "Channels cannot be used on state vectors.")

    def _density_matrix_call(self, state: tf.Tensor) -> tf.Tensor: # pragma: no cover
        """Loops over `self.gates` to calculate sum of Krauss operators."""
        # abstract method
        raise_error(NotImplementedError)


class NoiseChannel(TensorflowChannel, base_gates.NoiseChannel):

    def __init__(self, q: int, px: float = 0, py: float = 0, pz: float = 0):
        TensorflowChannel.__init__(self)
        base_gates.NoiseChannel.__init__(self, q, px, py, pz)

    def _density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        new_state = tf.zeros_like(state)
        for p, gate in self.gates:
            new_state += p * gate(state)
            gate(state) # reset to the original state vector
        return (1 - self.total_p) * state + new_state


class GeneralChannel(TensorflowChannel, base_gates.GeneralChannel):

    def __init__(self, A: Sequence[Tuple[Tuple[int], np.ndarray]]):
        TensorflowChannel.__init__(self)
        base_gates.GeneralChannel.__init__(self, A)
        self.dagger_gates = tuple()

    @staticmethod
    def _invert(gate):
        """Creates invert gates of each Ak to reset to the original state."""
        matrix = gate.parameter
        if isinstance(matrix, np.ndarray):
            inv_matrix = np.linalg.inv(matrix)
        elif isinstance(matrix, tf.Tensor):
            inv_matrix = np.linalg.inv(matrix)
        inv_gate = Unitary(inv_matrix, *gate.target_qubits)
        inv_gate.density_matrix = True
        inv_gate.device = gate.device
        inv_gate.nqubits = gate.nqubits
        return inv_gate

    def _prepare(self):
        TensorflowChannel._prepare(self)
        # create invert gates for resetting to the original state vector
        self.inv_gates = tuple(self._invert(gate) for gate in self.gates)

    def _density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        new_state = tf.zeros_like(state)
        assert len(self.gates) == len(self.inv_gates)
        for gate, inv_gate in zip(self.gates, self.inv_gates):
            new_state += gate(state)
            inv_gate(state)
        return new_state
