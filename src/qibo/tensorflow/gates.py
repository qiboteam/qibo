# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import sys
import numpy as np
import tensorflow as tf
from qibo.base import cache, gates
from qibo.base.abstract_gates import BackendGate
from qibo.tensorflow import cgates
from qibo.config import tfmatrices as matrices
from qibo.config import BACKEND, DTYPES, raise_error
from typing import Dict, List, Optional, Sequence, Tuple


class TensorflowGate(BackendGate):
    module = sys.modules[__name__]

    def __init__(self):
        super().__init__()
        self.calculation_cache = None
        # For `controlled_by` gates (see `cache.ControlCache` for more details)
        self.control_cache = None
        # Gate matrices
        self.matrix = None
        # Einsum backend
        self.einsum = BACKEND.get('EINSUM')

    @staticmethod
    def control_unitary(unitary: tf.Tensor) -> tf.Tensor:
        return cgates.TensorflowGate.control_unitary(unitary)

    def reprepare(self):
        matrix = self.construct_unitary()
        rank = int(np.log2(int(matrix.shape[0])))
        self.matrix = tf.reshape(matrix, 2 * rank * (2,))

    def prepare(self):
        self.is_prepared = True
        self.reprepare()
        if self.is_controlled_by:
            if self.density_matrix:
                # fall back to the 'defaulteinsum' backend when using
                # density matrices with `controlled_by` gates because
                # 'matmuleinsum' is not properly implemented for this case
                from qibo.tensorflow import einsum
                self.einsum = einsum.DefaultEinsum()
            self.control_cache = cache.ControlCache(self)
            nactive = self.nqubits - len(self.control_qubits)
            targets = self.control_cache.targets
            self.calculation_cache = self.einsum.create_cache(
                targets, nactive, ncontrol=len(self.control_qubits))
        else:
            self.calculation_cache = self.einsum.create_cache(
                self.qubits, self.nqubits)
        self.calculation_cache.cast_shapes(
            lambda x: tf.cast(x, dtype=DTYPES.get('DTYPEINT')))

    def set_nqubits(self, state: tf.Tensor):
        self.nqubits = len(tuple(state.shape)) // (1 + self.density_matrix)
        self.prepare()

    def state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        if self.is_controlled_by:
            ncontrol = len(self.control_qubits)
            nactive = self.nqubits - ncontrol
            state = tf.transpose(state, self.control_cache.order(False))
            # Apply `einsum` only to the part of the state where all controls
            # are active. This should be `state[-1]`
            state = tf.reshape(state, (2 ** ncontrol,) + nactive * (2,))
            updates = self.einsum(self.calculation_cache.vector, state[-1],
                                  self.matrix)
            # Concatenate the updated part of the state `updates` with the
            # part of of the state that remained unaffected `state[:-1]`.
            state = tf.concat([state[:-1], updates[tf.newaxis]], axis=0)
            state = tf.reshape(state, self.nqubits * (2,))
            # Put qubit indices back to their proper places
            state = tf.transpose(state, self.control_cache.reverse(False))
        else:
            einsum_str = self.calculation_cache.vector
            state = self.einsum(einsum_str, state, self.matrix)
        return state

    def density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        if self.is_controlled_by:
            ncontrol = len(self.control_qubits)
            nactive = self.nqubits - ncontrol
            state = tf.transpose(state, self.control_cache.order(True))
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
            state = tf.transpose(state, self.control_cache.reverse(True))
        else:
            state = self.einsum(self.calculation_cache.right, state,
                                tf.math.conj(self.matrix))
            state = self.einsum(self.calculation_cache.left, state, self.matrix)
        return state


class H(TensorflowGate, gates.H):

    def __init__(self, q):
        TensorflowGate.__init__(self)
        gates.H.__init__(self, q)

    def construct_unitary(self) -> tf.Tensor:
        return matrices.H


class X(TensorflowGate, gates.X):

    def __init__(self, q):
        TensorflowGate.__init__(self)
        gates.X.__init__(self, q)

    def construct_unitary(self) -> tf.Tensor:
        return matrices.X

    def controlled_by(self, *q):
        """Fall back to CNOT and Toffoli if controls are one or two."""
        # FIXME: This method can probably be removed
        gate = gates.X.controlled_by(self, *q)
        gate.einsum = self.einsum
        return gate


class Y(TensorflowGate, gates.Y):

    def __init__(self, q):
        TensorflowGate.__init__(self)
        gates.Y.__init__(self, q)

    def construct_unitary(self) -> tf.Tensor:
        return matrices.Y


class Z(TensorflowGate, gates.Z):

    def __init__(self, q):
        TensorflowGate.__init__(self)
        gates.Z.__init__(self, q)

    def construct_unitary(self) -> tf.Tensor:
        return matrices.Z


class Collapse(TensorflowGate, gates.Collapse):

    def __init__(self, *q: int, result: List[int] = 0):
        TensorflowGate.__init__(self)
        gates.Collapse.__init__(self, *q, result=result)
        self.order = None
        self.ids = None
        self.density_matrix_result = None

    @gates.Collapse.result.setter
    def result(self, res):
        x = cgates.Collapse._result_to_list(self, res)
        gates.Collapse.result.fset(self, x) # pylint: disable=no-member
        if self.is_prepared:
            self.prepare()

    def prepare(self):
        self.is_prepared = True
        self.order = list(self.sorted_qubits)
        if self.density_matrix:
            self.order.extend((q + self.nqubits for q in self.sorted_qubits))
            self.order.extend((q for q in range(self.nqubits)
                               if q not in self.sorted_qubits))
            self.order.extend((q + self.nqubits for q in range(self.nqubits)
                               if q not in self.sorted_qubits))
            self.sorted_qubits += [q + self.nqubits for q in self.sorted_qubits]
            self.density_matrix_result = 2 * self.result
        else:
            self.order.extend((q for q in range(self.nqubits)
                               if q not in self.sorted_qubits))

    @staticmethod
    def _append_zeros(state: tf.Tensor, qubits: List[int], results: List[int]
                      ) -> tf.Tensor:
        for q, r in zip(qubits, results):
            state = tf.expand_dims(state, axis=q)
            if r:
                state = tf.concat([tf.zeros_like(state), state], axis=q)
            else:
                state = tf.concat([state, tf.zeros_like(state)], axis=q)
        return state

    def construct_unitary(self):
        cgates.Collapse.construct_unitary(self)

    def state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        substate = tf.gather_nd(tf.transpose(state, self.order), self.result)
        norm = tf.reduce_sum(tf.square(tf.abs(substate)))
        state = substate / tf.cast(tf.sqrt(norm), dtype=state.dtype)
        return self._append_zeros(state, self.sorted_qubits, self.result)

    def density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        substate = tf.gather_nd(tf.transpose(state, self.order),
                                self.density_matrix_result)
        n = 2 ** (len(tuple(substate.shape)) // 2)
        norm = tf.linalg.trace(tf.reshape(substate, (n, n)))
        state = substate / norm
        return self._append_zeros(state, self.sorted_qubits,
                                  self.density_matrix_result)


class RX(TensorflowGate, gates.RX):

    def __init__(self, q, theta, trainable=True):
        TensorflowGate.__init__(self)
        gates.RX.__init__(self, q, theta, trainable)

    def construct_unitary(self) -> tf.Tensor:
        t = tf.cast(self.parameters, dtype=DTYPES.get('DTYPECPX'))
        return tf.cos(t / 2.0) * matrices.I - 1j * tf.sin(t / 2.0) * matrices.X


class RY(TensorflowGate, gates.RY):

    def __init__(self, q, theta, trainable=True):
        TensorflowGate.__init__(self)
        gates.RY.__init__(self, q, theta, trainable)

    def construct_unitary(self) -> tf.Tensor:
        t = tf.cast(self.parameters, dtype=DTYPES.get('DTYPECPX'))
        return tf.cos(t / 2.0) * matrices.I - 1j * tf.sin(t / 2.0) * matrices.Y


class RZ(TensorflowGate, gates.RZ):

    def __init__(self, q, theta, trainable=True):
        TensorflowGate.__init__(self)
        gates.RZ.__init__(self, q, theta, trainable)

    def construct_unitary(self) -> tf.Tensor:
        t = tf.cast(self.parameters, dtype=DTYPES.get('DTYPECPX'))
        phase = tf.exp(1j * t / 2.0)[tf.newaxis]
        diag = tf.concat([tf.math.conj(phase), phase], axis=0)
        return tf.linalg.diag(diag)


class U1(TensorflowGate, gates.U1):

    def __init__(self, q, theta, trainable=True):
        TensorflowGate.__init__(self)
        gates.U1.__init__(self, q, theta, trainable)

    def construct_unitary(self) -> tf.Tensor:
        t = tf.cast(self.parameters, dtype=DTYPES.get('DTYPECPX'))
        phase = tf.exp(1j * t)
        return tf.linalg.diag([1, phase])


class U2(TensorflowGate, gates.U2):

    def __init__(self, q, phi, lam, trainable=True):
        TensorflowGate.__init__(self)
        gates.U2.__init__(self, q, phi, lam, trainable)

    def construct_unitary(self):
        return cgates.U2.construct_unitary(self)


class U3(TensorflowGate, gates.U3):

    def __init__(self, q, theta, phi, lam, trainable=True):
        TensorflowGate.__init__(self)
        gates.U3.__init__(self, q, theta, phi, lam, trainable=trainable)

    def construct_unitary(self):
        return cgates.U3.construct_unitary(self)


class CNOT(TensorflowGate, gates.CNOT):

    def __init__(self, q0, q1):
        TensorflowGate.__init__(self)
        gates.CNOT.__init__(self, q0, q1)

    def construct_unitary(self) -> tf.Tensor:
        return tf.reshape(matrices.CNOT, (4, 4))


class CZ(TensorflowGate, gates.CZ):

    def __init__(self, q0, q1):
        TensorflowGate.__init__(self)
        gates.CZ.__init__(self, q0, q1)

    def construct_unitary(self) -> tf.Tensor:
        diag = tf.cast(tf.concat([tf.ones(3), [-1]], axis=0),
                       dtype=DTYPES.get('DTYPECPX'))
        return tf.linalg.diag(diag)


class _CUn_(TensorflowGate):
    base = U1

    def __init__(self, q0, q1, **params):
        cbase = "C{}".format(self.base.__name__)
        TensorflowGate.__init__(self)
        getattr(gates, cbase).__init__(self, q0, q1, **params)

    def construct_unitary(self) -> tf.Tensor:
        return TensorflowGate.control_unitary(self.base.construct_unitary(self))


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


class CU2(_CUn_, gates.CU2):
    base = U2

    def __init__(self, q0, q1, phi, lam, trainable=True):
        _CUn_.__init__(self, q0, q1, phi=phi, lam=lam, trainable=trainable)


class CU3(_CUn_, gates.CU3):
    base = U3

    def __init__(self, q0, q1, theta, phi, lam, trainable=True):
        _CUn_.__init__(self, q0, q1, theta=theta, phi=phi, lam=lam,
                       trainable=trainable)


class SWAP(TensorflowGate, gates.SWAP):

    def __init__(self, q0, q1):
        TensorflowGate.__init__(self)
        gates.SWAP.__init__(self, q0, q1)

    def construct_unitary(self) -> tf.Tensor:
        return tf.cast([[1, 0, 0, 0], [0, 0, 1, 0],
                        [0, 1, 0, 0], [0, 0, 0, 1]],
                       dtype=DTYPES.get('DTYPECPX'))


class fSim(TensorflowGate, gates.fSim):

    def __init__(self, q0, q1, theta, phi, trainable=True):
        TensorflowGate.__init__(self)
        gates.fSim.__init__(self, q0, q1, theta, phi, trainable)

    def construct_unitary(self):
        return cgates.fSim.construct_unitary(self)


class GeneralizedfSim(TensorflowGate, gates.GeneralizedfSim):

    def __init__(self, q0, q1, unitary, phi, trainable=True):
        TensorflowGate.__init__(self)
        gates.GeneralizedfSim.__init__(self, q0, q1, unitary, phi, trainable)

    def construct_unitary(self):
        return cgates.GeneralizedfSim.construct_unitary(self)

    def _dagger(self) -> "GeneralizedfSim":
        return cgates.GeneralizedfSim._dagger(self)


class TOFFOLI(TensorflowGate, gates.TOFFOLI):

    def __init__(self, q0, q1, q2):
        TensorflowGate.__init__(self)
        gates.TOFFOLI.__init__(self, q0, q1, q2)

    def construct_unitary(self) -> tf.Tensor:
        return tf.reshape(matrices.TOFFOLI, (8, 8))

    @property
    def unitary(self):
        if self._unitary is None:
            self._unitary = self.construct_unitary()
        return self._unitary


class Unitary(TensorflowGate, gates.Unitary):

    def __init__(self, unitary, *q, trainable=True, name: Optional[str] = None):
        if not isinstance(unitary, (np.ndarray, tf.Tensor)):
            raise_error(TypeError, "Unknown type {} of unitary matrix."
                                   "".format(type(unitary)))
        TensorflowGate.__init__(self)
        gates.Unitary.__init__(self, unitary, *q, trainable=trainable, name=name)

    def construct_unitary(self):
        unitary = self.parameters
        rank = int(np.log2(int(unitary.shape[0])))
        dtype = DTYPES.get('DTYPECPX')
        if isinstance(unitary, tf.Tensor):
            matrix = tf.identity(tf.cast(unitary, dtype=dtype))
        elif isinstance(unitary, np.ndarray):
            matrix = tf.convert_to_tensor(unitary, dtype=dtype)
        return matrix

    def _dagger(self) -> "Unitary":
        return cgates.Unitary._dagger(self)


class VariationalLayer(TensorflowGate, gates.VariationalLayer):

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

    def __init__(self, qubits: List[int], pairs: List[Tuple[int, int]],
                 one_qubit_gate, two_qubit_gate,
                 params: List[float],
                 params2: Optional[List[float]] = None,
                 trainable: bool = True,
                 name: Optional[str] = None):
        cgates.VariationalLayer.__init__(self, qubits, pairs,
                                         one_qubit_gate, two_qubit_gate,
                                         params, params2,
                                         trainable=trainable, name=name)

    def _dagger(self):
        return cgates.VariationalLayer._dagger(self)

    def construct_unitary(self):
        return cgates.VariationalLayer.construct_unitary(self)

    def reprepare(self):
        cgates.VariationalLayer.reprepare(self)

    def prepare(self):
        cgates.VariationalLayer.prepare(self)

    def state_vector_call(self, state: tf.Tensor) -> tf.Tensor: # pragma: no cover
        # impractical case because VariationalLayer is not called by circuits
        return cgates.VariationalLayer.state_vector_call(self, state)

    def density_matrix_call(self, state: tf.Tensor) -> tf.Tensor: # pragma: no cover
        # impractical case because VariationalLayer is not called by circuits
        return cgates.VariationalLayer.density_matrix_call(self, state)


class KrausChannel(TensorflowGate, gates.KrausChannel):

    def __init__(self, ops: Sequence[Tuple[Tuple[int], np.ndarray]]):
        TensorflowGate.__init__(self)
        gates.KrausChannel.__init__(self, ops)

    def prepare(self):
        self.is_prepared = True
        for gate in self.gates:
            gate.density_matrix = self.density_matrix
            gate.device = self.device
            gate.nqubits = self.nqubits
            gate.prepare()

    def construct_unitary(self):
        cgates.KrausChannel.construct_unitary(self)

    def state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        ccls = getattr(cgates, self.__class__.__name__)
        return ccls.state_vector_call(self, state)

    def density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        new_state = tf.zeros_like(state)
        for gate in self.gates:
            new_state += gate(state)
        return new_state


class UnitaryChannel(KrausChannel, gates.UnitaryChannel):

    def __init__(self, p: List[float], ops: List["Gate"],
                 seed: Optional[int] = None):
        TensorflowGate.__init__(self)
        gates.UnitaryChannel.__init__(self, p, ops, seed=seed)

    def prepare(self):
        KrausChannel.prepare(self)
        if self.seed is not None:
            np.random.seed(self.seed)

    def density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        new_state = (1 - self.psum) * state
        for p, gate in zip(self.probs, self.gates):
            new_state += p * gate(state)
        return new_state


class PauliNoiseChannel(UnitaryChannel, gates.PauliNoiseChannel):

    def __init__(self, q: int, px: float = 0, py: float = 0, pz: float = 0,
                 seed: Optional[int] = None):
        TensorflowGate.__init__(self)
        gates.PauliNoiseChannel.__init__(self, q, px, py, pz, seed=seed)


class ResetChannel(UnitaryChannel, gates.ResetChannel):

    def __init__(self, q: int, p0: float = 0.0, p1: float = 0.0,
                 seed: Optional[int] = None):
        TensorflowGate.__init__(self)
        gates.ResetChannel.__init__(self, q, p0=p0, p1=p1, seed=seed)

    def density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        new_state = (1 - self.psum) * state
        for p, gate in zip(self.probs, self.gates):
            state = gate(state)
            new_state += p * state
        return new_state


class _ThermalRelaxationChannelA(ResetChannel, gates._ThermalRelaxationChannelA):

    def calculate_probabilities(self, t1, t2, time, excited_population):
        cls = cgates.ThermalRelaxationChannel
        return cls.calculate_probabilities(self, t1, t2, time, excited_population)

    def __init__(self, q, t1, t2, time, excited_population=0, seed=None):
        TensorflowGate.__init__(self)
        gates._ThermalRelaxationChannelA.__init__(
            self, q, t1, t2, time, excited_population=excited_population,
            seed=seed)


class _ThermalRelaxationChannelB(TensorflowGate, gates._ThermalRelaxationChannelB):

    def calculate_probabilities(self, t1, t2, time, excited_population):
        cls = cgates.ThermalRelaxationChannel
        return cls.calculate_probabilities(self, t1, t2, time, excited_population)

    def __init__(self, q, t1, t2, time, excited_population=0, seed=None):
        TensorflowGate.__init__(self)
        gates._ThermalRelaxationChannelB.__init__(
            self, q, t1, t2, time, excited_population=excited_population,
            seed=seed)

    def prepare(self):
        self.is_prepared = True
        self.reprepare()
        qubits = self.qubits + tuple(q + self.nqubits for q in self.qubits)
        self.calculation_cache = self.einsum.create_cache(
            qubits, 2 * self.nqubits)
        self.calculation_cache.cast_shapes(
            lambda x: tf.cast(x, dtype=DTYPES.get('DTYPEINT')))

    def construct_unitary(self):
        return cgates._ThermalRelaxationChannelB.construct_unitary(self)

    def state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        raise_error(ValueError, "Thermal relaxation cannot be applied to "
                                "state vectors when T1 < T2.")

    def density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        einsum_str = self.calculation_cache.vector
        return self.einsum(einsum_str, state, self.matrix)
