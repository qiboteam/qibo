# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import sys
import math
from qibo import K, get_threads, matrices
from qibo.abstractions import gates
from qibo.abstractions.abstract_gates import BaseBackendGate, ParametrizedGate
from qibo.config import raise_error
from typing import Dict, List, Optional, Sequence, Tuple


class BackendGate(BaseBackendGate):
    module = sys.modules[__name__]

    def __new__(cls, *args, **kwargs):
        cgate_only = {"I", "M", "ZPow", "CZPow", "Flatten",
                      "CallbackGate", "PartialTrace"}
        # TODO: Move these to a different file and refactor
        if K.custom_gates or cls.__name__ in cgate_only:
            return super(BackendGate, cls).__new__(cls)
        else:
            from qibo.core import gates
            return getattr(gates, cls.__name__)(*args, **kwargs) # pylint: disable=E0110

    def __init__(self):
        if not K.executing_eagerly():
            raise_error(NotImplementedError,
                        "Custom operator gates should not be used in compiled "
                        "mode.")
        super().__init__()
        if K.op:
            self.gate_op = K.op.apply_gate
        self.qubits_tensor = None
        self.qubits_tensor_dm = None
        self.target_qubits_dm = None

    @staticmethod
    def control_unitary(unitary):
        shape = tuple(unitary.shape)
        if shape != (2, 2):
            raise_error(ValueError, "Cannot use ``control_unitary`` method for "
                                    "input matrix of shape {}.".format(shape))
        zeros = K.zeros((2, 2), dtype='DTYPECPX')
        part1 = K.concatenate([K.eye(2, dtype='DTYPECPX'), zeros], axis=0)
        part2 = K.concatenate([zeros, unitary], axis=0)
        return K.concatenate([part1, part2], axis=1)

    def reprepare(self):
        raise_error(RuntimeError, "Cannot reprepare non-parametrized gate.")

    def prepare(self):
        """Prepares the gate for application to state vectors."""
        self.is_prepared = True
        targets = K.qnp.cast(self.target_qubits, dtype="int32")
        controls = K.qnp.cast(self.control_qubits, dtype="int32")
        qubits = list(self.nqubits - controls - 1)
        qubits.extend(self.nqubits - targets - 1)
        qubits = sorted(qubits)
        with K.device(self.device):
            self.qubits_tensor = K.cast(qubits, dtype="int32")
            if self.density_matrix:
                self.target_qubits_dm = tuple(targets + self.nqubits)
                self.qubits_tensor_dm = self.qubits_tensor + self.nqubits

    def set_nqubits(self, state):
        self.nqubits = int(math.log2(tuple(state.shape)[0]))
        self.prepare()

    def state_vector_call(self, state):
        return self.gate_op(state, self.qubits_tensor, self.nqubits,
                            *self.target_qubits, get_threads())

    def density_matrix_call(self, state):
        state = self.gate_op(state, self.qubits_tensor_dm, 2 * self.nqubits,
                             *self.target_qubits, get_threads())
        state = self.gate_op(state, self.qubits_tensor, 2 * self.nqubits,
                             *self.target_qubits_dm, get_threads())
        return state


class MatrixGate(BackendGate):
    """Gate that uses matrix multiplication to be applied to states."""

    def __init__(self):
        super().__init__()
        self.matrix = None

    def reprepare(self):
        with K.device(self.device):
            self.matrix = K.cast(self.construct_unitary(), dtype='DTYPECPX')

    def prepare(self):
        super().prepare()
        self.reprepare()

    def state_vector_call(self, state):
        return self.gate_op(state, self.matrix, self.qubits_tensor, # pylint: disable=E1121
                            self.nqubits, *self.target_qubits, get_threads())

    def density_matrix_call(self, state):
        state = self.gate_op(state, self.matrix, self.qubits_tensor_dm, # pylint: disable=E1121
                             2 * self.nqubits, *self.target_qubits, get_threads())
        adjmatrix = K.conj(self.matrix)
        state = self.gate_op(state, adjmatrix, self.qubits_tensor,
                             2 * self.nqubits, *self.target_qubits_dm, get_threads())
        return state


class H(MatrixGate, gates.H):

    def __init__(self, q):
        MatrixGate.__init__(self)
        gates.H.__init__(self, q)

    def construct_unitary(self):
        return matrices.H


class X(BackendGate, gates.X):

    def __init__(self, q):
        BackendGate.__init__(self)
        gates.X.__init__(self, q)
        self.gate_op = K.op.apply_x

    def construct_unitary(self):
        return matrices.X


class Y(BackendGate, gates.Y):

    def __init__(self, q):
        BackendGate.__init__(self)
        gates.Y.__init__(self, q)
        self.gate_op = K.op.apply_y

    def construct_unitary(self):
        return matrices.Y

    def density_matrix_call(self, state):
        return -BackendGate.density_matrix_call(self, state)


class Z(BackendGate, gates.Z):

    def __init__(self, q):
        BackendGate.__init__(self)
        gates.Z.__init__(self, q)
        self.gate_op = K.op.apply_z

    def construct_unitary(self):
        return matrices.Z


class I(BackendGate, gates.I):

    def __init__(self, *q):
        BackendGate.__init__(self)
        gates.I.__init__(self, *q)

    def construct_unitary(self):
        return K.qnp.eye(2 ** len(self.target_qubits))

    def state_vector_call(self, state):
        return state

    def density_matrix_call(self, state):
        return state


class Collapse(BackendGate, gates.Collapse):

    def __init__(self, *q: int, result: List[int] = 0):
        BackendGate.__init__(self)
        gates.Collapse.__init__(self, *q, result=result)
        self.result_tensor = None
        self.gate_op = K.op.collapse_state

    def _result_to_list(self, res):
        if isinstance(res, K.tensor_types):
            return list(K.qnp.cast(res, dtype='DTYPEINT'))
        if isinstance(res, int) or isinstance(res, K.numeric_types):
            return len(self.target_qubits) * [res]
        return list(res)

    @gates.Collapse.result.setter
    def result(self, res):
        gates.Collapse.result.fset(self, self._result_to_list(res)) # pylint: disable=no-member
        if self.is_prepared:
            self.reprepare()

    def reprepare(self):
        n = len(self.result)
        result = sum(2 ** (n - i - 1) * r for i, r in enumerate(self.result))
        self.result_tensor = K.cast(result, dtype='DTYPEINT')

    def prepare(self):
        BackendGate.prepare(self)
        self.reprepare()

    def construct_unitary(self):
        raise_error(ValueError, "Collapse gate does not have unitary "
                                "representation.")

    def state_vector_call(self, state):
        return self.gate_op(state, self.qubits_tensor, self.result_tensor,
                            self.nqubits, self.normalize, get_threads())

    def density_matrix_call(self, state):
        state = self.gate_op(state, self.qubits_tensor_dm, self.result_tensor,
                             2 * self.nqubits, False, get_threads())
        state = self.gate_op(state, self.qubits_tensor, self.result_tensor,
                             2 * self.nqubits, False, get_threads())
        return state / K.trace(state)


class M(BackendGate, gates.M):
    from qibo.core import measurements, states

    def __init__(self, *q, register_name: Optional[str] = None,
                 p0: Optional["ProbsType"] = None,
                 p1: Optional["ProbsType"] = None):
        BackendGate.__init__(self)
        gates.M.__init__(self, *q, register_name=register_name, p0=p0, p1=p1)
        self.traceout = None
        self.unmeasured_qubits = None # Tuple
        self.reduced_target_qubits = None # List

    def add(self, gate: gates.M):
        if self.is_prepared:
            raise_error(RuntimeError, "Cannot add qubits to a measurement "
                                      "gate that is prepared.")
        gates.M.add(self, gate)

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
            qubits = set(self.unmeasured_qubits)
            self.traceout = self.einsum_string(qubits, self.nqubits, True)

    def _get_cpu(self): # pragma: no cover
        # case not covered by GitHub workflows because it requires OOM
        if not K.cpu_devices:
            raise_error(RuntimeError, "Cannot find CPU device to use for sampling.")
        return K.cpu_devices[0]

    def construct_unitary(self):
        raise_error(ValueError, "Measurement gate does not have unitary "
                                "representation.")

    def state_vector_call(self, state):
        return self.states.VectorState.from_tensor(state)

    def density_matrix_call(self, state):
        return self.states.MatrixState.from_tensor(state)

    def sample(self, state, nshots):
        probs_dim = K.cast((2 ** len(self.target_qubits),), dtype='DTYPEINT')
        probs = state.probabilities(measurement_gate=self)
        probs = K.transpose(probs, axes=self.reduced_target_qubits)
        probs = K.reshape(probs, probs_dim)
        samples_dec = K.sample_measurements(probs, nshots)
        result = self.measurements.GateResult(
            self.qubits, decimal_samples=samples_dec)
        # optional bitflip noise
        if sum(sum(x.values()) for x in self.bitflip_map) > 0:
            result = result.apply_bitflips(*self.bitflip_map)
        return result

    def __call__(self, state, nshots):
        if isinstance(state, K.tensor_types):
            if not self.is_prepared:
                self.set_nqubits(state)
            state = getattr(self, self._active_call)(state)
        elif isinstance(state, self.states.AbstractState):
            if not self.is_prepared:
                self.set_nqubits(state.tensor)
        else:
            raise_error(TypeError)

        if math.log2(nshots) + len(self.target_qubits) > 31: # pragma: no cover
            # case not covered by GitHub workflows because it requires large example
            # Use CPU to avoid "aborted" error
            device = self._get_cpu()

        try:
            with K.device(self.device):
                result = self.sample(state, nshots)
        except K.oom_error: # pragma: no cover
            # case not covered by GitHub workflows because it requires OOM
            # Force using CPU to perform sampling
            device = self._get_cpu()
            with K.device(device):
                result = self.sample(state, nshots)
        return result


class RX(MatrixGate, gates.RX):

    def __init__(self, q, theta, trainable=True):
        MatrixGate.__init__(self)
        gates.RX.__init__(self, q, theta, trainable)

    def construct_unitary(self):
        theta = self.parameters
        cos, isin = K.qnp.cos(theta / 2.0), -1j * K.qnp.sin(theta / 2.0)
        return K.qnp.cast([[cos, isin], [isin, cos]])


class RY(MatrixGate, gates.RY):

    def __init__(self, q, theta, trainable=True):
        MatrixGate.__init__(self)
        gates.RY.__init__(self, q, theta, trainable)

    def construct_unitary(self):
        theta = self.parameters
        cos, sin = K.qnp.cos(theta / 2.0), K.qnp.sin(theta / 2.0)
        return K.qnp.cast([[cos, -sin], [sin, cos]])


class RZ(MatrixGate, gates.RZ):

    def __init__(self, q, theta, trainable=True):
        MatrixGate.__init__(self)
        gates.RZ.__init__(self, q, theta, trainable)

    def construct_unitary(self):
        phase = K.qnp.exp(1j * self.parameters / 2.0)
        return K.qnp.diag([K.qnp.conj(phase), phase])


class U1(MatrixGate, gates.U1):

    def __init__(self, q, theta, trainable=True):
        MatrixGate.__init__(self)
        gates.U1.__init__(self, q, theta, trainable)
        self.gate_op = K.op.apply_z_pow

    def reprepare(self):
        with K.device(self.device):
            self.matrix = K.cast(K.qnp.exp(1j * self.parameters),
                                 dtype='DTYPECPX')

    def construct_unitary(self):
        return K.qnp.diag([1, K.qnp.exp(1j * self.parameters)])


class U2(MatrixGate, gates.U2):

    def __init__(self, q, phi, lam, trainable=True):
        MatrixGate.__init__(self)
        gates.U2.__init__(self, q, phi, lam, trainable)

    def construct_unitary(self):
        phi, lam = self.parameters
        eplus = K.qnp.exp(1j * (phi + lam) / 2.0)
        eminus = K.qnp.exp(1j * (phi - lam) / 2.0)
        return K.qnp.cast([[eplus.conj(), - eminus.conj()],
                         [eminus, eplus]]) / K.qnp.sqrt(2)


class U3(MatrixGate, gates.U3):

    def __init__(self, q, theta, phi, lam, trainable=True):
        MatrixGate.__init__(self)
        gates.U3.__init__(self, q, theta, phi, lam, trainable)

    def construct_unitary(self):
        theta, phi, lam = self.parameters
        cost = K.qnp.cos(theta / 2)
        sint = K.qnp.sin(theta / 2)
        eplus = K.qnp.exp(1j * (phi + lam) / 2.0)
        eminus = K.qnp.exp(1j * (phi - lam) / 2.0)
        return K.qnp.cast([[eplus.conj() * cost, - eminus.conj() * sint],
                         [eminus * sint, eplus * cost]])


class ZPow(gates.ZPow):

  def __new__(cls, q, theta, trainable=True):
      if K.custom_gates:
          return U1(q, theta, trainable)
      else:
          from qibo.core import gates
          return gates.U1(q, theta, trainable)


class CNOT(BackendGate, gates.CNOT):

    def __init__(self, q0, q1):
        BackendGate.__init__(self)
        gates.CNOT.__init__(self, q0, q1)
        self.gate_op = K.op.apply_x

    def construct_unitary(self):
        return matrices.CNOT


class CZ(BackendGate, gates.CZ):

    def __init__(self, q0, q1):
        BackendGate.__init__(self)
        gates.CZ.__init__(self, q0, q1)
        self.gate_op = K.op.apply_z

    def construct_unitary(self):
        return matrices.CZ


class _CUn_(MatrixGate):
    base = U1

    def __init__(self, q0, q1, **params):
        MatrixGate.__init__(self)
        cbase = "C{}".format(self.base.__name__)
        getattr(gates, cbase).__init__(self, q0, q1, **params)

    def reprepare(self):
        with K.device(self.device):
            self.matrix = K.cast(self.base.construct_unitary(self),
                                 dtype='DTYPECPX')

    def construct_unitary(self):
        return MatrixGate.control_unitary(self.base.construct_unitary(self))


class CRX(_CUn_, gates.CRX):
    base = RX

    def __init__(self, q0, q1, theta, trainable=True):
        _CUn_.__init__(self, q0, q1, theta=theta, trainable=trainable)


class CRY(_CUn_, gates.CRY):
    base = RY

    def __init__(self, q0, q1, theta, trainable=True):
        _CUn_.__init__(self, q0, q1, theta=theta, trainable=trainable)


class CRZ(_CUn_, gates.CRZ):
    base = RZ

    def __init__(self, q0, q1, theta, trainable=True):
        _CUn_.__init__(self, q0, q1, theta=theta, trainable=trainable)


class CU1(_CUn_, gates.CU1):
    base = U1

    def __init__(self, q0, q1, theta, trainable=True):
        _CUn_.__init__(self, q0, q1, theta=theta, trainable=trainable)
        self.gate_op = K.op.apply_z_pow

    def reprepare(self):
        U1.reprepare(self)


class CU2(_CUn_, gates.CU2):
    base = U2

    def __init__(self, q0, q1, phi, lam, trainable=True):
        _CUn_.__init__(self, q0, q1, phi=phi, lam=lam, trainable=trainable)


class CU3(_CUn_, gates.CU3):
    base = U3

    def __init__(self, q0, q1, theta, phi, lam, trainable=True):
        _CUn_.__init__(self, q0, q1, theta=theta, phi=phi, lam=lam,
                       trainable=trainable)


class CZPow(gates.CZPow):

  def __new__(cls, q0, q1, theta, trainable=True):
      if K.custom_gates:
          return CU1(q0, q1, theta, trainable)
      else:
          from qibo.core import gates
          return gates.CU1(q0, q1, theta, trainable)


class SWAP(BackendGate, gates.SWAP):

    def __init__(self, q0, q1):
        BackendGate.__init__(self)
        gates.SWAP.__init__(self, q0, q1)
        self.gate_op = K.op.apply_swap

    def construct_unitary(self):
        return matrices.SWAP


class fSim(MatrixGate, gates.fSim):

    def __init__(self, q0, q1, theta, phi, trainable=True):
        MatrixGate.__init__(self)
        gates.fSim.__init__(self, q0, q1, theta, phi, trainable)
        self.gate_op = K.op.apply_fsim

    def reprepare(self):
        theta, phi = self.parameters
        cos, isin = K.qnp.cos(theta), -1j * K.qnp.sin(theta)
        phase = K.qnp.exp(-1j * phi)
        matrix = K.qnp.cast([cos, isin, isin, cos, phase])
        with K.device(self.device):
            self.matrix = K.cast(matrix)

    def construct_unitary(self):
        theta, phi = self.parameters
        cos, isin = K.qnp.cos(theta), -1j * K.qnp.sin(theta)
        matrix = K.qnp.eye(4)
        matrix[1, 1], matrix[2, 2] = cos, cos
        matrix[1, 2], matrix[2, 1] = isin, isin
        matrix[3, 3] = K.qnp.exp(-1j * phi)
        return matrix


class GeneralizedfSim(MatrixGate, gates.GeneralizedfSim):

    def __init__(self, q0, q1, unitary, phi, trainable=True):
        BackendGate.__init__(self)
        gates.GeneralizedfSim.__init__(self, q0, q1, unitary, phi, trainable)
        self.gate_op = K.op.apply_fsim

    def reprepare(self):
        unitary, phi = self.parameters
        matrix = K.qnp.zeros(5)
        matrix[:4] = K.qnp.reshape(unitary, (4,))
        matrix[4] = K.qnp.exp(-1j * phi)
        with K.device(self.device):
            self.matrix = K.cast(matrix)

    def construct_unitary(self):
        unitary, phi = self.parameters
        matrix = K.qnp.eye(4)
        matrix[1:3, 1:3] = K.qnp.reshape(unitary, (2, 2))
        matrix[3, 3] = K.qnp.exp(-1j * phi)
        return matrix

    def _dagger(self) -> "GenerelizedfSim":
        unitary, phi = self.parameters
        if isinstance(unitary, K.Tensor):
            ud = K.conj(K.transpose(unitary))
        else:
            ud = unitary.conj().T
        q0, q1 = self.target_qubits
        return self.__class__(q0, q1, ud, -phi)


class TOFFOLI(BackendGate, gates.TOFFOLI):

    def __init__(self, q0, q1, q2):
        BackendGate.__init__(self)
        gates.TOFFOLI.__init__(self, q0, q1, q2)
        self.gate_op = K.op.apply_x

    def construct_unitary(self):
        return matrices.TOFFOLI

    @property
    def unitary(self):
        if self._unitary is None:
            self._unitary = self.construct_unitary()
        return self._unitary


class Unitary(MatrixGate, gates.Unitary):

    def __init__(self, unitary, *q, trainable=True, name: Optional[str] = None):
        if not isinstance(unitary, K.tensor_types):
            raise_error(TypeError, "Unknown type {} of unitary matrix."
                                   "".format(type(unitary)))
        MatrixGate.__init__(self)
        gates.Unitary.__init__(self, unitary, *q, trainable=trainable, name=name)
        rank = self.rank
        if rank == 1:
            self.gate_op = K.op.apply_gate
        elif rank == 2:
            self.gate_op = K.op.apply_two_qubit_gate
        else:
            n = len(self.target_qubits)
            raise_error(NotImplementedError, "Unitary gate supports one or two-"
                                             "qubit gates when using custom "
                                             "operators, but {} target qubits "
                                             "were given. Please switch to a "
                                             "different backend to execute "
                                             "this operation.".format(n))

    def construct_unitary(self):
        unitary = self.parameters
        if isinstance(unitary, K.qnp.Tensor):
            return K.qnp.cast(unitary)
        if isinstance(unitary, K.Tensor): # pragma: no cover
            return K.copy(K.cast(unitary))

    def _dagger(self) -> "Unitary":
        unitary = self.parameters
        if isinstance(unitary, K.Tensor):
            ud = K.conj(K.transpose(unitary))
        else:
            ud = unitary.conj().T
        return self.__class__(ud, *self.target_qubits, **self.init_kwargs)

    @ParametrizedGate.parameters.setter
    def parameters(self, x):
        x = K.qnp.cast(x)
        shape = tuple(x.shape)
        true_shape = (2 ** self.rank, 2 ** self.rank)
        if shape == true_shape:
            ParametrizedGate.parameters.fset(self, x) # pylint: disable=no-member
        elif shape == (2 ** (2 * self.rank),):
            ParametrizedGate.parameters.fset(self, x.reshape(true_shape)) # pylint: disable=no-member
        else:
            raise_error(ValueError, "Invalid shape {} of unitary matrix "
                                    "acting on {} target qubits."
                                    "".format(shape, self.rank))


class VariationalLayer(BackendGate, gates.VariationalLayer):

    def _calculate_unitaries(self):
        matrices = K.qnp.stack([K.qnp.kron(
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
            matrices2 = K.qnp.stack([K.qnp.kron(
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
                 trainable: bool = True,
                 name: Optional[str] = None):
        self.module.BackendGate.__init__(self)
        gates.VariationalLayer.__init__(self, qubits, pairs,
                                        one_qubit_gate, two_qubit_gate,
                                        params, params2,
                                        trainable=trainable, name=name)

        matrices, additional_matrix = self._calculate_unitaries()
        self.unitaries = []
        for targets, matrix in zip(self.pairs, matrices):
            unitary = self.module.Unitary(matrix, *targets)
            self.unitaries.append(unitary)
        if self.additional_target is not None:
            self.additional_unitary = self.module.Unitary(
                additional_matrix, self.additional_target)
            self.additional_unitary.density_matrix = self.density_matrix
        else:
            self.additional_unitary = None

    @BaseBackendGate.density_matrix.setter
    def density_matrix(self, x: bool):
        BaseBackendGate.density_matrix.fset(self, x) # pylint: disable=no-member
        for unitary in self.unitaries:
            unitary.density_matrix = x
        if self.additional_unitary is not None:
            self.additional_unitary.density_matrix = x

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

    def state_vector_call(self, state):
        for i, unitary in enumerate(self.unitaries):
            state = unitary(state)
        if self.additional_unitary is not None:
            state = self.additional_unitary(state)
        return state

    def density_matrix_call(self, state):
        return self.state_vector_call(state)


class Flatten(BackendGate, gates.Flatten):

    def __init__(self, coefficients):
        BackendGate.__init__(self)
        gates.Flatten.__init__(self, coefficients)
        self.swap_reset = []

    def construct_unitary(self):
        raise_error(ValueError, "Flatten gate does not have unitary "
                                 "representation.")

    def state_vector_call(self, state):
        shape = tuple(state.shape)
        _state = K.qnp.reshape(K.qnp.cast(self.coefficients), shape)
        return K.cast(_state, dtype="DTYPECPX")

    def density_matrix_call(self, state):
        return self.state_vector_call(state)


class CallbackGate(BackendGate, gates.CallbackGate):

    def __init__(self, callback):
        BackendGate.__init__(self)
        gates.CallbackGate.__init__(self, callback)
        self.swap_reset = []

    @BaseBackendGate.density_matrix.setter
    def density_matrix(self, x):
        BaseBackendGate.density_matrix.fset(self, x) # pylint: disable=no-member
        self.callback.density_matrix = x

    def construct_unitary(self):
        raise_error(ValueError, "Callback gate does not have unitary "
                                "representation.")

    def state_vector_call(self, state):
        self.callback.append(self.callback(state))
        return state

    def density_matrix_call(self, state):
        return self.state_vector_call(state)


class PartialTrace(BackendGate, gates.PartialTrace):

    def __init__(self, *q):
        BackendGate.__init__(self)
        gates.PartialTrace.__init__(self, *q)

        self.traceout_string = None
        self.zero_matrix = None
        self.transpose_order = None
        self.output_shape = None

    def prepare(self):
        BackendGate.prepare(self)
        qubits = set(self.target_qubits)
        self.traceout_string = M.einsum_string(qubits, self.nqubits)
        # Create |00...0><00...0| for qubits that are traced out
        n = len(self.target_qubits)
        row0 = K.cast([1] + (2 ** n - 1) * [0], dtype='DTYPECPX')
        shape = K.cast((2 ** n - 1, 2 ** n), dtype='DTYPEINT')
        rows = K.zeros(shape, dtype='DTYPECPX')
        self.zero_matrix = K.concatenate([row0[K.newaxis], rows], axis=0)
        self.zero_matrix = K.reshape(self.zero_matrix, 2 * n * (2,))
        # Calculate final transpose order
        order1 = tuple(i for i in range(self.nqubits) if i not in qubits)
        order2 = tuple(self.target_qubits)
        order = (order1 + tuple(i + self.nqubits for i in order1) +
                 order2 + tuple(i + self.nqubits for i in order2))
        self.transpose_order = tuple(order.index(i) for i in range(2 * self.nqubits))
        # Output shape
        self.output_shape = K.cast(2 * (2 ** self.nqubits,), dtype='DTYPEINT')

    def construct_unitary(self):
        raise_error(ValueError, "Partial trace gate does not have unitary "
                                "representation.")

    def state_vector_call(self, state):
        raise_error(RuntimeError, "Partial trace gate cannot be used on state "
                                  "vectors. Please switch to density matrix "
                                  "simulation.")

    def density_matrix_call(self, state):
        self.set_nqubits(state)
        state = K.reshape(state, 2 * self.nqubits * (2,))
        substate = K.einsum(self.traceout_string, state)
        state = K.tensordot(substate, self.zero_matrix, axes=0)
        state = K.transpose(state, self.transpose_order)
        return K.reshape(state, self.output_shape)


class KrausChannel(BackendGate, gates.KrausChannel):

    def __init__(self, ops):
        BackendGate.__init__(self)
        gates.KrausChannel.__init__(self, ops)
        # create inversion gates to rest to the original state vector
        # because of the in-place updates used in custom operators
        self.inv_gates = tuple()

    @staticmethod
    def _invert(gate):
        """Creates invert gates of each Ak to reset to the original state."""
        matrix = gate.parameters
        if isinstance(matrix, K.tensor_types):
            inv_matrix = K.qnp.inv(matrix)
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

    def state_vector_call(self, state):
        raise_error(ValueError, "`KrausChannel` cannot be applied to state "
                                "vectors. Please switch to density matrices.")

    def density_matrix_call(self, state):
        new_state = K.zeros_like(state)
        for gate, inv_gate in zip(self.gates, self.inv_gates):
            new_state += gate(state)
            if inv_gate is not None:
                inv_gate(state)
        return new_state


class UnitaryChannel(KrausChannel, gates.UnitaryChannel):

    def __init__(self, p: List[float], ops: List["Gate"],
                 seed: Optional[int] = None):
        BackendGate.__init__(self)
        gates.UnitaryChannel.__init__(self, p, ops, seed=seed)
        self.inv_gates = tuple()

    @staticmethod
    def _invert(gate):
        return gate.dagger()

    def set_seed(self):
        if self.seed is not None:
            K.qnp.random.seed(self.seed)

    def prepare(self):
        KrausChannel.prepare(self)
        self.set_seed()

    def state_vector_call(self, state):
        for p, gate in zip(self.probs, self.gates):
            if K.qnp.random.random() < p:
                state = gate(state)
        return state

    def density_matrix_call(self, state):
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
        BackendGate.__init__(self)
        gates.PauliNoiseChannel.__init__(self, q, px, py, pz, seed=seed)
        self.inv_gates = tuple()

    @staticmethod
    def _invert(gate):
        # for Pauli gates we can use same gate as inverse for efficiency
        return gate


class ResetChannel(UnitaryChannel, gates.ResetChannel):

    def __init__(self, q: int, p0: float = 0.0, p1: float = 0.0,
                 seed: Optional[int] = None):
        BackendGate.__init__(self)
        gates.ResetChannel.__init__(self, q, p0=p0, p1=p1, seed=seed)
        self.inv_gates = tuple()

    @staticmethod
    def _invert(gate):
        if isinstance(gate, gates.Collapse):
            return None
        return gate

    def state_vector_call(self, state):
        not_collapsed = True
        if K.qnp.random.random() < self.probs[-2]:
            state = self.gates[-2](state)
            not_collapsed = False
        if K.qnp.random.random() < self.probs[-1]:
            if not_collapsed:
                state = self.gates[-2](state)
            state = self.gates[-1](state)
        return state


class ThermalRelaxationChannel(gates.ThermalRelaxationChannel):

    def __new__(cls, q, t1, t2, time, excited_population=0, seed=None):
        if K.custom_gates:
            cls_a = _ThermalRelaxationChannelA
            cls_b = _ThermalRelaxationChannelB
        else:
            from qibo.core import gates
            cls_a = gates._ThermalRelaxationChannelA
            cls_b = gates._ThermalRelaxationChannelB
        if t2 > t1:
            cls_s = cls_b
        else:
            cls_s = cls_a
        return cls_s(
            q, t1, t2, time, excited_population=excited_population, seed=seed)

    def calculate_probabilities(self, t1, t2, time, excited_population):
        cls = gates.ThermalRelaxationChannel
        cls.calculate_probabilities(self, t1, t2, time, excited_population)
        p_reset = 1 - K.qnp.exp(-time / t1)
        p0 = p_reset * (1 - excited_population)
        p1 = p_reset * excited_population
        if t1 < t2:
            exp = K.qnp.exp(-time / t2)
        else:
            rate1, rate2 = 1 / t1, 1 / t2
            exp = (1 - p_reset) * (1 - K.qnp.exp(-time * (rate2 - rate1))) / 2
        return (exp, p0, p1)


class _ThermalRelaxationChannelA(ResetChannel, gates._ThermalRelaxationChannelA):

    def calculate_probabilities(self, t1, t2, time, excited_population):
        return ThermalRelaxationChannel.calculate_probabilities(
            self, t1, t2, time, excited_population)

    def __init__(self, q, t1, t2, time, excited_population=0, seed=None):
        BackendGate.__init__(self)
        gates._ThermalRelaxationChannelA.__init__(
            self, q, t1, t2, time, excited_population=excited_population,
            seed=seed)
        self.inv_gates = tuple()

    def state_vector_call(self, state):
        if K.qnp.random.random() < self.probs[0]:
            state = self.gates[0](state)
        return ResetChannel.state_vector_call(self, state)


class _ThermalRelaxationChannelB(MatrixGate, gates._ThermalRelaxationChannelB):

    def calculate_probabilities(self, t1, t2, time, excited_population):
        return ThermalRelaxationChannel.calculate_probabilities(
            self, t1, t2, time, excited_population)

    def __init__(self, q, t1, t2, time, excited_population=0, seed=None):
        BackendGate.__init__(self)
        gates._ThermalRelaxationChannelB.__init__(
            self, q, t1, t2, time, excited_population=excited_population,
            seed=seed)
        self.gate_op = K.op.apply_two_qubit_gate

    def prepare(self):
        super().prepare()
        targets = K.qnp.cast(self.target_qubits, dtype="int32")
        controls = K.qnp.cast(self.control_qubits, dtype="int32")
        qubits = sorted(list(self.nqubits - controls - 1))
        qubits = self.nqubits - targets - 1
        qubits = K.qnp.concatenate([qubits, qubits + self.nqubits], axis=0)
        qubits = sorted(list(qubits))
        self.qubits_tensor = K.cast(qubits, dtype="int32")
        self.target_qubits_dm = (self.target_qubits +
                                 tuple(targets + self.nqubits))

    def construct_unitary(self):
        matrix = K.qnp.diag([1 - self.preset1, self.exp_t2, self.exp_t2,
                           1 - self.preset0])
        matrix[0, -1] = self.preset1
        matrix[-1, 0] = self.preset0
        return matrix

    def state_vector_call(self, state):
        raise_error(ValueError, "Thermal relaxation cannot be applied to "
                                "state vectors when T1 < T2.")

    def density_matrix_call(self, state):
        return self.gate_op(state, self.matrix, self.qubits_tensor,
                            2 * self.nqubits, *self.target_qubits_dm, get_threads())
