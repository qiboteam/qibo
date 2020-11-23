# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import sys
import numpy as np
import tensorflow as tf
from qibo.base import gates
from qibo.base.abstract_gates import BackendGate
from qibo.config import BACKEND, DTYPES, DEVICES, NUMERIC_TYPES, raise_error
from qibo.tensorflow import custom_operators as op
from typing import Dict, List, Optional, Sequence, Tuple


class TensorflowGate(BackendGate):
    module = sys.modules[__name__]

    def __new__(cls, *args, **kwargs):
        cgate_only = {"I", "M", "Flatten", "CallbackGate", "ZPow", "CZPow"}
        # TODO: Move these to a different file and refactor
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
        super().__init__()
        self.gate_op = op.apply_gate
        self.qubits_tensor = None
        self.qubits_tensor_dm = None
        self.target_qubits_dm = None

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

    def reprepare(self):
        raise_error(RuntimeError, "Cannot reprepare non-parametrized gate.")

    def prepare(self):
        """Prepares the gate for application to state vectors."""
        self.is_prepared = True
        qubits = list(self.nqubits - np.array(self.control_qubits) - 1)
        qubits.extend(self.nqubits - np.array(self.target_qubits) - 1)
        qubits = sorted(qubits)
        with tf.device(self.device):
            self.qubits_tensor = tf.convert_to_tensor(qubits, dtype=tf.int32)
            if self.density_matrix:
                self.target_qubits_dm = tuple(np.array(self.target_qubits) +
                                              self.nqubits)
                self.qubits_tensor_dm = self.qubits_tensor + self.nqubits

    def set_nqubits(self, state: tf.Tensor):
        self.nqubits = int(np.log2(tuple(state.shape)[0]))
        self.prepare()

    def state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        return self.gate_op(state, self.qubits_tensor, self.nqubits,
                            *self.target_qubits)

    def density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        state = self.gate_op(state, self.qubits_tensor_dm, 2 * self.nqubits,
                             *self.target_qubits)
        state = self.gate_op(state, self.qubits_tensor, 2 * self.nqubits,
                             *self.target_qubits_dm)
        return state


class MatrixGate(TensorflowGate):
    """Gate that uses matrix multiplication to be applied to states."""

    def __init__(self):
        super().__init__()
        self.matrix = None

    def reprepare(self):
        with tf.device(self.device):
            self.matrix = tf.constant(self.construct_unitary(),
                                      dtype=DTYPES.get('DTYPECPX'))

    def prepare(self):
        super().prepare()
        self.reprepare()

    def state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        return self.gate_op(state, self.matrix, self.qubits_tensor,
                            self.nqubits, *self.target_qubits)

    def density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        state = self.gate_op(state, self.matrix, self.qubits_tensor_dm,
                             2 * self.nqubits, *self.target_qubits)
        adjmatrix = tf.math.conj(self.matrix)
        state = self.gate_op(state, adjmatrix, self.qubits_tensor,
                             2 * self.nqubits, *self.target_qubits_dm)
        return state


class H(MatrixGate, gates.H):

    def __init__(self, q):
        MatrixGate.__init__(self)
        gates.H.__init__(self, q)

    def construct_unitary(self) -> np.ndarray:
        return (np.array([[1, 1], [1, -1]], dtype=DTYPES.get('NPTYPECPX'))
                / np.sqrt(2))


class X(TensorflowGate, gates.X):

    def __init__(self, q):
        TensorflowGate.__init__(self)
        gates.X.__init__(self, q)
        self.gate_op = op.apply_x

    def construct_unitary(self) -> np.ndarray:
        return np.array([[0, 1], [1, 0]], dtype=DTYPES.get('NPTYPECPX'))


class Y(TensorflowGate, gates.Y):

    def __init__(self, q):
        TensorflowGate.__init__(self)
        gates.Y.__init__(self, q)
        self.gate_op = op.apply_y

    def construct_unitary(self) -> np.ndarray:
        return np.array([[0, -1j], [1j, 0]], dtype=DTYPES.get('NPTYPECPX'))

    def density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        return -TensorflowGate.density_matrix_call(self, state)


class Z(TensorflowGate, gates.Z):

    def __init__(self, q):
        TensorflowGate.__init__(self)
        gates.Z.__init__(self, q)
        self.gate_op = op.apply_z

    def construct_unitary(self) -> np.ndarray:
        return np.array([[1, 0], [0, -1]], dtype=DTYPES.get('NPTYPECPX'))


class I(TensorflowGate, gates.I):

    def __init__(self, *q):
        TensorflowGate.__init__(self)
        gates.I.__init__(self, *q)

    def construct_unitary(self) -> np.ndarray:
        dim = 2 ** len(self.target_qubits)
        return np.eye(dim, dtype=DTYPES.get('NPTYPECPX'))

    def state_vector_call(self, state: tf.Tensor):
        return state

    def density_matrix_call(self, state: tf.Tensor):
        return state


class Collapse(TensorflowGate, gates.Collapse):

    def __init__(self, *q: int, result: List[int] = 0):
        TensorflowGate.__init__(self)
        gates.Collapse.__init__(self, *q, result=result)
        self.result_tensor = None
        self.gate_op = op.collapse_state

    def _result_to_list(self, res):
        if isinstance(res, np.ndarray):
            return list(res.astype(np.int))
        if isinstance(res, tf.Tensor):
            return list(res.numpy().astype(np.int))
        if isinstance(res, int) or isinstance(res, NUMERIC_TYPES):
            return len(self.target_qubits) * [res]
        return list(res)

    def prepare(self):
        TensorflowGate.prepare(self)
        n = len(self.result)
        result = sum(2 ** (n - i - 1) * r for i, r in enumerate(self.result))
        self.result_tensor = tf.cast(result, dtype=DTYPES.get('DTYPEINT'))

    def construct_unitary(self):
        raise_error(ValueError, "Collapse gate does not have unitary "
                                "representation.")

    def state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        return self.gate_op(state, self.qubits_tensor, self.result_tensor,
                            self.nqubits, self.normalize)

    def density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        state = self.gate_op(state, self.qubits_tensor_dm, self.result_tensor,
                             2 * self.nqubits, False)
        state = self.gate_op(state, self.qubits_tensor, self.result_tensor,
                             2 * self.nqubits, False)
        return state / tf.linalg.trace(state)


class M(TensorflowGate, gates.M):
    from qibo.tensorflow import distutils, measurements

    def __init__(self, *q, register_name: Optional[str] = None,
                 p0: Optional["ProbsType"] = None,
                 p1: Optional["ProbsType"] = None):
        TensorflowGate.__init__(self)
        gates.M.__init__(self, *q, register_name=register_name, p0=p0, p1=p1)
        self.traceout = None
        self.unmeasured_qubits = None # Tuple
        self.reduced_target_qubits = None # List

    def prepare(self):
        self.is_prepared = True
        target_qubits = set(self.target_qubits)
        unmeasured_qubits = []
        reduced_target_qubits = dict()
        for i in range(self.nqubits):
            if i in target_qubits:
                reduced_target_qubits[i] = i - len(unmeasured_qubits)
            else:
                unmeasured_qubits.append(i)
        self.unmeasured_qubits = tuple(unmeasured_qubits)
        self.reduced_target_qubits = list(
            reduced_target_qubits[i] for i in self.target_qubits)
        if self.density_matrix:
            from qibo.tensorflow.einsum import DefaultEinsum
            qubits = set(self.unmeasured_qubits)
            # TODO: Remove ``DefaultEinsum`` dependence here
            self.traceout = DefaultEinsum.partialtrace_str(
                qubits, self.nqubits, measuring=True)

    def _get_cpu(self): # pragma: no cover
        # case not covered by GitHub workflows because it requires OOM
        if not DEVICES['CPU']:
            raise_error(RuntimeError, "Cannot find CPU device to use for sampling.")
        return DEVICES['CPU'][0]

    def construct_unitary(self):
        raise_error(ValueError, "Measurement gate does not have unitary "
                                "representation.")

    def state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        shape = self.nqubits * (2,)
        x = tf.reshape(tf.square(tf.abs(state)), shape)
        return tf.reduce_sum(x, axis=self.unmeasured_qubits)

    def density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        shape = 2 * self.nqubits * (2,)
        x = tf.einsum(self.traceout, tf.reshape(state, shape))
        return tf.cast(x, dtype=DTYPES.get('DTYPE'))

    def sample(self, state: tf.Tensor, nshots: int) -> tf.Tensor:
        probs = getattr(self, self._active_call)(state)
        probs = tf.transpose(probs, perm=self.reduced_target_qubits)

        dtype = DTYPES.get('DTYPEINT')
        probs_dim = tf.cast((2 ** len(self.target_qubits),), dtype=dtype)
        logits = tf.math.log(tf.reshape(probs, probs_dim))[tf.newaxis]

        samples_dec = tf.random.categorical(logits, nshots, dtype=dtype)[0]
        result = self.measurements.GateResult(
            self.qubits, decimal_samples=samples_dec)
        # optional bitflip noise
        if sum(sum(x.values()) for x in self.bitflip_map) > 0:
            result = result.apply_bitflips(*self.bitflip_map)
        return result

    def __call__(self, state: tf.Tensor, nshots: int) -> tf.Tensor:
        if isinstance(state, self.distutils.DistributedState):
            with tf.device(state.device):
                state = state.vector

        if not self.is_prepared:
            self.set_nqubits(state)

        if np.log2(nshots) + len(self.target_qubits) > 31: # pragma: no cover
            # case not covered by GitHub workflows because it requires large example
            # Use CPU to avoid "aborted" error
            device = self._get_cpu()

        oom_error = tf.python.framework.errors_impl.ResourceExhaustedError
        try:
            with tf.device(self.device):
                result = self.sample(state, nshots)
        except oom_error: # pragma: no cover
            # case not covered by GitHub workflows because it requires OOM
            # Force using CPU to perform sampling
            device = self._get_cpu()
            with tf.device(device):
                result = self.sample(state, nshots)
        return result


class RX(MatrixGate, gates.RX):

    def __init__(self, q, theta):
        MatrixGate.__init__(self)
        gates.RX.__init__(self, q, theta)

    def construct_unitary(self) -> np.ndarray:
        theta = self.parameters
        cos, isin = np.cos(theta / 2.0), -1j * np.sin(theta / 2.0)
        return np.array([[cos, isin], [isin, cos]], dtype=DTYPES.get('NPTYPECPX'))


class RY(MatrixGate, gates.RY):

    def __init__(self, q, theta):
        MatrixGate.__init__(self)
        gates.RY.__init__(self, q, theta)

    def construct_unitary(self) -> np.ndarray:
        theta = self.parameters
        cos, sin = np.cos(theta / 2.0), np.sin(theta / 2.0)
        return np.array([[cos, -sin], [sin, cos]], dtype=DTYPES.get('NPTYPECPX'))


class RZ(MatrixGate, gates.RZ):

    def __init__(self, q, theta):
        MatrixGate.__init__(self)
        gates.RZ.__init__(self, q, theta)

    def construct_unitary(self) -> np.ndarray:
        phase = np.exp(1j * self.parameters / 2.0)
        return np.diag([phase.conj(), phase]).astype(DTYPES.get('NPTYPECPX'))


class U1(MatrixGate, gates.U1):

    def __init__(self, q, theta):
        MatrixGate.__init__(self)
        gates.U1.__init__(self, q, theta)
        self.gate_op = op.apply_z_pow

    def reprepare(self):
        with tf.device(self.device):
            self.matrix = tf.constant(np.exp(1j * self.parameters),
                                      dtype=DTYPES.get('DTYPECPX'))

    def construct_unitary(self) -> np.ndarray:
        return np.diag([1, np.exp(1j * self.parameters)]).astype(
            DTYPES.get('NPTYPECPX'))


class U2(MatrixGate, gates.U2):

    def __init__(self, q, phi, lam):
        MatrixGate.__init__(self)
        gates.U2.__init__(self, q, phi, lam)

    def construct_unitary(self) -> np.ndarray:
        phi, lam = self.parameters
        eplus = np.exp(1j * (phi + lam) / 2.0)
        eminus = np.exp(1j * (phi - lam) / 2.0)
        return np.array([[eplus.conj(), - eminus.conj()], [eminus, eplus]],
                        dtype=DTYPES.get('NPTYPECPX')) / np.sqrt(2)


class U3(MatrixGate, gates.U3):

    def __init__(self, q, theta, phi, lam):
        MatrixGate.__init__(self)
        gates.U3.__init__(self, q, theta, phi, lam)

    def construct_unitary(self) -> np.ndarray:
        theta, phi, lam = self.parameters
        cost = np.cos(theta / 2)
        sint = np.sin(theta / 2)
        eplus = np.exp(1j * (phi + lam) / 2.0)
        eminus = np.exp(1j * (phi - lam) / 2.0)
        return np.array([[eplus.conj() * cost, - eminus.conj() * sint],
                         [eminus * sint, eplus * cost]],
                        dtype=DTYPES.get('NPTYPECPX'))


class ZPow(MatrixGate, gates.ZPow):

  def __new__(cls, q, theta):
      if BACKEND.get('GATES') == 'custom':
          return U1(q, theta)
      else:
          from qibo.tensorflow import gates
          return gates.U1(q, theta)


class CNOT(TensorflowGate, gates.CNOT):

    def __init__(self, q0, q1):
        TensorflowGate.__init__(self)
        gates.CNOT.__init__(self, q0, q1)
        self.gate_op = op.apply_x

    def construct_unitary(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                         [0, 0, 0, 1], [0, 0, 1, 0]],
                        dtype=DTYPES.get('NPTYPECPX'))


class CZ(TensorflowGate, gates.CZ):

    def __init__(self, q0, q1):
        TensorflowGate.__init__(self)
        gates.CZ.__init__(self, q0, q1)
        self.gate_op = op.apply_z

    def construct_unitary(self) -> np.ndarray:
        return np.diag([1, 1, 1, -1]).astype(DTYPES.get('NPTYPECPX'))


class _CUn_(MatrixGate):
    base = U1

    def __init__(self, q0, q1, **params):
        MatrixGate.__init__(self)
        cbase = "C{}".format(self.base.__name__)
        getattr(gates, cbase).__init__(self, q0, q1, **params)

    def reprepare(self):
        with tf.device(self.device):
            self.matrix = tf.constant(self.base.construct_unitary(self),
                                      dtype=DTYPES.get('DTYPECPX'))

    def construct_unitary(self) -> tf.Tensor:
        return MatrixGate.control_unitary(self.base.construct_unitary(self))


class CRX(_CUn_, gates.CRX):
    base = RX

    def __init__(self, q0, q1, theta):
        _CUn_.__init__(self, q0, q1, theta=theta)


class CRY(_CUn_, gates.CRY):
    base = RY

    def __init__(self, q0, q1, theta):
        _CUn_.__init__(self, q0, q1, theta=theta)


class CRZ(_CUn_, gates.CRZ):
    base = RZ

    def __init__(self, q0, q1, theta):
        _CUn_.__init__(self, q0, q1, theta=theta)


class CU1(_CUn_, gates.CU1):
    base = U1

    def __init__(self, q0, q1, theta):
        _CUn_.__init__(self, q0, q1, theta=theta)
        self.gate_op = op.apply_z_pow

    def reprepare(self):
        U1.reprepare(self)


class CU2(_CUn_, gates.CU2):
    base = U2

    def __init__(self, q0, q1, phi, lam):
        _CUn_.__init__(self, q0, q1, phi=phi, lam=lam)


class CU3(_CUn_, gates.CU3):
    base = U3

    def __init__(self, q0, q1, theta, phi, lam):
        _CUn_.__init__(self, q0, q1, theta=theta, phi=phi, lam=lam)


class CZPow(MatrixGate, gates.CZPow):

  def __new__(cls, q0, q1, theta):
      if BACKEND.get('GATES') == 'custom':
          return CU1(q0, q1, theta)
      else:
          from qibo.tensorflow import gates
          return gates.CU1(q0, q1, theta)


class SWAP(TensorflowGate, gates.SWAP):

    def __init__(self, q0, q1):
        TensorflowGate.__init__(self)
        gates.SWAP.__init__(self, q0, q1)
        self.gate_op = op.apply_swap

    def construct_unitary(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                         [0, 1, 0, 0], [0, 0, 0, 1]],
                        dtype=DTYPES.get('NPTYPECPX'))


class fSim(MatrixGate, gates.fSim):

    def __init__(self, q0, q1, theta, phi):
        MatrixGate.__init__(self)
        gates.fSim.__init__(self, q0, q1, theta, phi)
        self.gate_op = op.apply_fsim

    def reprepare(self):
        theta, phi = self.parameters
        cos, isin = np.cos(theta), -1j * np.sin(theta)
        phase = np.exp(-1j * phi)
        matrix = np.array([cos, isin, isin, cos, phase],
                          dtype=DTYPES.get('NPTYPECPX'))
        with tf.device(self.device):
            self.matrix = tf.constant(matrix, dtype=DTYPES.get('DTYPECPX'))

    def construct_unitary(self) -> np.ndarray:
        theta, phi = self.parameters
        cos, isin = np.cos(theta), -1j * np.sin(theta)
        matrix = np.eye(4, dtype=DTYPES.get('NPTYPECPX'))
        matrix[1, 1], matrix[2, 2] = cos, cos
        matrix[1, 2], matrix[2, 1] = isin, isin
        matrix[3, 3] = np.exp(-1j * phi)
        return matrix


class GeneralizedfSim(MatrixGate, gates.GeneralizedfSim):

    def __init__(self, q0, q1, unitary, phi):
        TensorflowGate.__init__(self)
        gates.GeneralizedfSim.__init__(self, q0, q1, unitary, phi)
        self.gate_op = op.apply_fsim

    def reprepare(self):
        unitary, phi = self.parameters
        matrix = np.zeros(5, dtype=DTYPES.get("NPTYPECPX"))
        matrix[:4] = np.reshape(unitary, (4,))
        matrix[4] = np.exp(-1j * phi)
        with tf.device(self.device):
            self.matrix = tf.constant(matrix, dtype=DTYPES.get('DTYPECPX'))

    def construct_unitary(self) -> np.ndarray:
        unitary, phi = self.parameters
        matrix = np.eye(4, dtype=DTYPES.get('NPTYPECPX'))
        matrix[1:3, 1:3] = np.reshape(unitary, (2, 2))
        matrix[3, 3] = np.exp(-1j * phi)
        return matrix

    def _dagger(self) -> "GenerelizedfSim":
        unitary, phi = self.parameters
        if isinstance(unitary, tf.Tensor):
            ud = tf.math.conj(tf.transpose(unitary))
        else:
            ud = unitary.conj().T
        q0, q1 = self.target_qubits
        return self.__class__(q0, q1, ud, -phi)


class TOFFOLI(TensorflowGate, gates.TOFFOLI):

    def __init__(self, q0, q1, q2):
        TensorflowGate.__init__(self)
        gates.TOFFOLI.__init__(self, q0, q1, q2)
        self.gate_op = op.apply_x

    def construct_unitary(self) -> np.ndarray:
        matrix = np.eye(8, dtype=DTYPES.get('NPTYPECPX'))
        matrix[-2, -2], matrix[-2, -1] = 0, 1
        matrix[-1, -2], matrix[-1, -1] = 1, 0
        return matrix

    @property
    def unitary(self):
        if self._unitary is None:
            self._unitary = self.construct_unitary()
        return self._unitary


class Unitary(MatrixGate, gates.Unitary):

    def __init__(self, unitary, *q, name: Optional[str] = None):
        if not isinstance(unitary, (np.ndarray, tf.Tensor)):
            raise_error(TypeError, "Unknown type {} of unitary matrix."
                                   "".format(type(unitary)))
        MatrixGate.__init__(self)
        gates.Unitary.__init__(self, unitary, *q, name=name)
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

    def construct_unitary(self) -> np.ndarray:
        unitary = self.parameters
        if isinstance(unitary, np.ndarray):
            return unitary.astype(DTYPES.get('NPTYPECPX'))
        if isinstance(unitary, tf.Tensor):
            return tf.identity(tf.cast(unitary, dtype=DTYPES.get('DTYPECPX')))

    def _dagger(self) -> "Unitary":
        unitary = self.parameters
        if isinstance(unitary, tf.Tensor):
            ud = tf.math.conj(tf.transpose(unitary))
        else:
            ud = unitary.conj().T
        return self.__class__(ud, *self.target_qubits, **self.init_kwargs)


class VariationalLayer(TensorflowGate, gates.VariationalLayer):

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

    def __init__(self, qubits: List[int], pairs: List[Tuple[int, int]],
                 one_qubit_gate, two_qubit_gate,
                 params: List[float], params2: Optional[List[float]] = None,
                 name: Optional[str] = None):
        self.module.TensorflowGate.__init__(self)
        gates.VariationalLayer.__init__(self, qubits, pairs,
                                        one_qubit_gate, two_qubit_gate,
                                        params, params2, name=name)

        matrices, additional_matrix = self._calculate_unitaries()
        self.unitaries = []
        for targets, matrix in zip(self.pairs, matrices):
            unitary = Unitary(matrix, *targets)
            unitary.density_matrix = self.density_matrix
            self.unitaries.append(unitary)
        if self.additional_target is not None:
            self.additional_unitary = self.module.Unitary(
                additional_matrix, self.additional_target)
            self.additional_unitary.density_matrix = self.density_matrix
        else:
            self.additional_unitary = None

    def _dagger(self):
        import copy
        varlayer = copy.copy(self)
        varlayer.unitaries = [u.dagger() for u in self.unitaries]
        if self.additional_unitary is not None:
            varlayer.additional_unitary = self.additional_unitary.dagger()
        return varlayer

    def construct_unitary(self):
        raise_error(ValueError, "VariationalLayer gate does not have unitary "
                                 "representation.")

    def reprepare(self):
        matrices, additional_matrix = self._calculate_unitaries()
        for unitary, matrix in zip(self.unitaries, matrices):
            unitary.parameters = matrix
            unitary.reprepare()
        if additional_matrix is not None:
            self.additional_unitary.parameters = additional_matrix
            self.additional_unitary.reprepare()

    def prepare(self):
        self.is_prepared = True

    def state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        for i, unitary in enumerate(self.unitaries):
            state = unitary(state)
        if self.additional_unitary is not None:
            state = self.additional_unitary(state)
        return state

    def density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        return self.state_vector_call(state)


class Flatten(TensorflowGate, gates.Flatten):

    def __init__(self, coefficients):
        TensorflowGate.__init__(self)
        gates.Flatten.__init__(self, coefficients)
        self.swap_reset = []

    def construct_unitary(self):
        raise_error(ValueError, "Flatten gate does not have unitary "
                                 "representation.")

    def state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        shape = tuple(state.shape)
        _state = np.array(self.coefficients).reshape(shape)
        return tf.convert_to_tensor(_state, dtype=DTYPES.get("DTYPECPX"))

    def density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        return self.state_vector_call(state)


class CallbackGate(TensorflowGate, gates.CallbackGate):

    def __init__(self, callback):
        TensorflowGate.__init__(self)
        gates.CallbackGate.__init__(self, callback)
        self.swap_reset = []

    def construct_unitary(self):
        raise_error(ValueError, "Unitary gate does not have unitary "
                                 "representation.")

    def state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        self.callback.append(self.callback(state, self.density_matrix))
        return state

    def density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        return self.state_vector_call(state)


class KrausChannel(TensorflowGate, gates.KrausChannel):

    def __init__(self, ops: Sequence[Tuple[Tuple[int], np.ndarray]]):
        TensorflowGate.__init__(self)
        gates.KrausChannel.__init__(self, ops)
        # create inversion gates to rest to the original state vector
        # because of the in-place updates used in custom operators
        self.inv_gates = tuple()

    @staticmethod
    def _invert(gate):
        """Creates invert gates of each Ak to reset to the original state."""
        matrix = gate.parameters
        if isinstance(matrix, np.ndarray):
            inv_matrix = np.linalg.inv(matrix)
        elif isinstance(matrix, tf.Tensor):
            inv_matrix = np.linalg.inv(matrix)
        return Unitary(inv_matrix, *gate.target_qubits)

    def prepare(self):
        self.is_prepared = True
        inv_gates = []
        for gate in self.gates:
            inv_gate = self._invert(gate)
            # use a ``set`` for this loop because it may be ``inv_gate == gate``
            for g in {gate, inv_gate}:
                if g is not None:
                    g.density_matrix = self.density_matrix
                    g.device = self.device
                    g.nqubits = self.nqubits
                    g.prepare()
            inv_gates.append(inv_gate)
        inv_gates[-1] = None
        self.inv_gates = tuple(inv_gates)

    def construct_unitary(self):
        raise_error(ValueError, "Channels do not have unitary representation.")

    def state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        raise_error(ValueError, "`KrausChannel` cannot be applied to state "
                                "vectors. Please switch to density matrices.")

    def density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        new_state = tf.zeros_like(state)
        for gate, inv_gate in zip(self.gates, self.inv_gates):
            new_state += gate(state)
            if inv_gate is not None:
                inv_gate(state)
        return new_state


class UnitaryChannel(KrausChannel, gates.UnitaryChannel):

    def __init__(self, p: List[float], ops: List["Gate"],
                 seed: Optional[int] = None):
        TensorflowGate.__init__(self)
        gates.UnitaryChannel.__init__(self, p, ops, seed=seed)
        self.inv_gates = tuple()

    @staticmethod
    def _invert(gate):
        return gate.dagger()

    def prepare(self):
        KrausChannel.prepare(self)
        if self.seed is not None:
            np.random.seed(self.seed)

    def state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        for p, gate in zip(self.probs, self.gates):
            if np.random.random() < p:
                state = gate(state)
        return state

    def density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        new_state = (1 - self.psum) * state
        for p, gate, inv_gate in zip(self.probs, self.gates, self.inv_gates):
            state = gate(state)
            new_state += p * state
            if inv_gate is not None:
                state = inv_gate(state) # reset to the original state vector
        return new_state


class PauliNoiseChannel(UnitaryChannel, gates.PauliNoiseChannel):

    def __init__(self, q: int, px: float = 0, py: float = 0, pz: float = 0,
                 seed: Optional[int] = None):
        TensorflowGate.__init__(self)
        gates.PauliNoiseChannel.__init__(self, q, px, py, pz, seed=seed)
        self.inv_gates = tuple()

    @staticmethod
    def _invert(gate):
        # for Pauli gates we can use same gate as inverse for efficiency
        return gate


class ResetChannel(UnitaryChannel, gates.ResetChannel):

    def __init__(self, q: int, p0: float = 0.0, p1: float = 0.0,
                 seed: Optional[int] = None):
        TensorflowGate.__init__(self)
        gates.ResetChannel.__init__(self, q, p0=p0, p1=p1, seed=seed)
        self.inv_gates = tuple()

    @staticmethod
    def _invert(gate):
        if isinstance(gate, gates.Collapse):
            return None
        return gate

    def state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        not_collapsed = True
        if np.random.random() < self.probs[-2]:
            state = self.gates[-2](state)
            not_collapsed = False
        if np.random.random() < self.probs[-1]:
            if not_collapsed:
                state = self.gates[-2](state)
            state = self.gates[-1](state)
        return state


class ThermalRelaxationChannel(TensorflowGate, gates.ThermalRelaxationChannel):

    def __new__(cls, q, t1, t2, time, excited_population=0, seed=None):
        if BACKEND.get('GATES') == "custom":
            cls_a = _ThermalRelaxationChannelA
            cls_b = _ThermalRelaxationChannelB
        else:
            from qibo.tensorflow import gates
            cls_a = gates._ThermalRelaxationChannelA
            cls_b = gates._ThermalRelaxationChannelB
        if t2 > t1:
            cls_s = cls_b
        else:
            cls_s = cls_a
        return cls_s(
            q, t1, t2, time, excited_population=excited_population, seed=seed)


class _ThermalRelaxationChannelA(ResetChannel, gates._ThermalRelaxationChannelA):

    def __init__(self, q, t1, t2, time, excited_population=0, seed=None):
        TensorflowGate.__init__(self)
        gates._ThermalRelaxationChannelA.__init__(
            self, q, t1, t2, time, excited_population=excited_population,
            seed=seed)
        self.inv_gates = tuple()

    def state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        if np.random.random() < self.probs[0]:
            state = self.gates[0](state)
        return ResetChannel.state_vector_call(self, state)


class _ThermalRelaxationChannelB(MatrixGate, gates._ThermalRelaxationChannelB):

    def __init__(self, q, t1, t2, time, excited_population=0, seed=None):
        TensorflowGate.__init__(self)
        gates._ThermalRelaxationChannelB.__init__(
            self, q, t1, t2, time, excited_population=excited_population,
            seed=seed)
        self.gate_op = op.apply_two_qubit_gate

    def prepare(self) -> tf.Tensor:
        super().prepare()
        qubits = sorted(list(self.nqubits - np.array(self.control_qubits) - 1))
        qubits = self.nqubits - np.array(self.target_qubits) - 1
        qubits = np.concatenate([qubits, qubits + self.nqubits], axis=0)
        qubits = sorted(list(qubits))
        self.qubits_tensor = tf.convert_to_tensor(qubits, dtype=tf.int32)
        self.target_qubits_dm = (self.target_qubits +
                                 tuple(np.array(self.target_qubits) + self.nqubits))

    def construct_unitary(self) -> np.ndarray:
        matrix = np.diag([1 - self.preset1, self.exp_t2, self.exp_t2,
                          1 - self.preset0])
        matrix[0, -1] = self.preset1
        matrix[-1, 0] = self.preset0
        return matrix.astype(DTYPES.get('NPTYPECPX'))

    def state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        raise_error(ValueError, "Thermal relaxation cannot be applied to "
                                "state vectors when T1 < T2.")

    def density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        return self.gate_op(state, self.matrix, self.qubits_tensor,
                            2 * self.nqubits, *self.target_qubits_dm)
