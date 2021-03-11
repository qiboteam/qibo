# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import math
import sys
from qibo import K
from qibo.abstractions import gates
from qibo.abstractions.abstract_gates import BaseBackendGate, ParametrizedGate
from qibo.core import cgates
from qibo.config import raise_error
from typing import Dict, List, Optional, Tuple


class BackendGate(BaseBackendGate):
    module = sys.modules[__name__]

    def __init__(self):
        super().__init__()
        self.calculation_cache = None
        # For ``controlled_by`` gates
        # (see ``core.einsum.ControlCache`` for more details)
        self.control_cache = None
        # Gate matrices
        self.matrix = None
        # Einsum backend
        from qibo.core import einsum
        self.einsum_module = einsum
        self.einsum = getattr(einsum, K.custom_einsum)()
        self.tensor_shape = None
        self.flat_shape = None

    @staticmethod
    def control_unitary(unitary):
        return cgates.BackendGate.control_unitary(unitary)

    def reprepare(self):
        matrix = self.construct_unitary()
        rank = int(math.log2(int(matrix.shape[0])))
        self.matrix = K.reshape(matrix, 2 * rank * (2,))

    def prepare(self):
        self.is_prepared = True
        if self.well_defined:
            self.reprepare()
        try:
            s = 1 + self.density_matrix
            self.tensor_shape = K.cast(s * self.nqubits * (2,), dtype='DTYPEINT')
            self.flat_shape = K.cast(s * (2 ** self.nqubits,), dtype='DTYPEINT')
            if self.is_controlled_by:
                if self.density_matrix:
                    # fall back to the 'defaulteinsum' backend when using
                    # density matrices with `controlled_by` gates because
                    # 'matmuleinsum' is not properly implemented for this case
                    self.einsum = self.einsum_module.DefaultEinsum()
                self.control_cache = self.einsum_module.ControlCache(self)
                nactive = self.nqubits - len(self.control_qubits)
                targets = self.control_cache.targets
                self.calculation_cache = self.einsum.create_cache(
                    targets, nactive, ncontrol=len(self.control_qubits))
            else:
                self.calculation_cache = self.einsum.create_cache(
                    self.qubits, self.nqubits)
            self.calculation_cache.cast_shapes(
                lambda x: K.cast(x, dtype='DTYPEINT'))
        except (ValueError, OverflowError):
            pass

    def set_nqubits(self, state):
        cgates.BackendGate.set_nqubits(self, state)

    def state_vector_call(self, state):
        state = K.reshape(state, self.tensor_shape)
        if self.is_controlled_by:
            ncontrol = len(self.control_qubits)
            nactive = self.nqubits - ncontrol
            state = K.transpose(state, self.control_cache.order(False))
            # Apply `einsum` only to the part of the state where all controls
            # are active. This should be `state[-1]`
            state = K.reshape(state, (2 ** ncontrol,) + nactive * (2,))
            updates = self.einsum(self.calculation_cache.vector, state[-1],
                                  self.matrix)
            # Concatenate the updated part of the state `updates` with the
            # part of of the state that remained unaffected `state[:-1]`.
            state = K.concatenate([state[:-1], updates[K.newaxis]], axis=0)
            state = K.reshape(state, self.nqubits * (2,))
            # Put qubit indices back to their proper places
            state = K.transpose(state, self.control_cache.reverse(False))
        else:
            einsum_str = self.calculation_cache.vector
            state = self.einsum(einsum_str, state, self.matrix)
        return K.reshape(state, self.flat_shape)

    def density_matrix_call(self, state):
        state = K.reshape(state, self.tensor_shape)
        if self.is_controlled_by:
            ncontrol = len(self.control_qubits)
            nactive = self.nqubits - ncontrol
            n = 2 ** ncontrol
            state = K.transpose(state, self.control_cache.order(True))
            state = K.reshape(state, 2 * (n,) + 2 * nactive * (2,))
            state01 = K.gather(state, indices=range(n - 1), axis=0)
            state01 = K.squeeze(K.gather(state01, indices=[n - 1], axis=1), axis=1)
            state01 = self.einsum(self.calculation_cache.right0,
                                  state01, K.conj(self.matrix))
            state10 = K.gather(state, indices=range(n - 1), axis=1)
            state10 = K.squeeze(K.gather(state10, indices=[n - 1], axis=0), axis=0)
            state10 = self.einsum(self.calculation_cache.left0,
                                  state10, self.matrix)

            state11 = K.squeeze(K.gather(state, indices=[n - 1], axis=0), axis=0)
            state11 = K.squeeze(K.gather(state11, indices=[n - 1], axis=0), axis=0)
            state11 = self.einsum(self.calculation_cache.right, state11,
                                  K.conj(self.matrix))
            state11 = self.einsum(self.calculation_cache.left,
                                  state11, self.matrix)

            state00 = K.gather(state, indices=range(n - 1), axis=0)
            state00 = K.gather(state00, indices=range(n - 1), axis=1)
            state01 = K.concatenate([state00, state01[:, K.newaxis]], axis=1)
            state10 = K.concatenate([state10, state11[K.newaxis]], axis=0)
            state = K.concatenate([state01, state10[K.newaxis]], axis=0)
            state = K.reshape(state, 2 * self.nqubits * (2,))
            state = K.transpose(state, self.control_cache.reverse(True))
        else:
            state = self.einsum(self.calculation_cache.right, state,
                                K.conj(self.matrix))
            state = self.einsum(self.calculation_cache.left, state, self.matrix)
        return K.reshape(state, self.flat_shape)


class H(BackendGate, gates.H):

    def __init__(self, q):
        BackendGate.__init__(self)
        gates.H.__init__(self, q)

    def construct_unitary(self):
        return K.matrices.H


class X(BackendGate, gates.X):

    def __init__(self, q):
        BackendGate.__init__(self)
        gates.X.__init__(self, q)

    def construct_unitary(self):
        return K.matrices.X

    def controlled_by(self, *q):
        """Fall back to CNOT and Toffoli if controls are one or two."""
        # FIXME: This method can probably be removed
        gate = gates.X.controlled_by(self, *q)
        gate.einsum = self.einsum
        return gate


class Y(BackendGate, gates.Y):

    def __init__(self, q):
        BackendGate.__init__(self)
        gates.Y.__init__(self, q)

    def construct_unitary(self):
        return K.matrices.Y


class Z(BackendGate, gates.Z):

    def __init__(self, q):
        BackendGate.__init__(self)
        gates.Z.__init__(self, q)

    def construct_unitary(self):
        return K.matrices.Z


class M(BackendGate, gates.M):
    from qibo.core import measurements, states

    def __init__(self, *q, register_name: Optional[str] = None,
                 collapse: bool = False,
                 p0: Optional["ProbsType"] = None,
                 p1: Optional["ProbsType"] = None):
        BackendGate.__init__(self)
        gates.M.__init__(self, *q, register_name=register_name,
                         collapse=collapse, p0=p0, p1=p1)
        self.unmeasured_qubits = None # Tuple
        self.reduced_target_qubits = None # List

        self.result = self.measurements.MeasurementResult(self.qubits)
        self._result_list = None
        self._result_tensor = None
        self.order = None

    def add(self, gate: gates.M):
        cgates.M.add(self, gate)

    def prepare(self):
        cgates.M.prepare(self)
        try:
            sorted_qubits = sorted(self.target_qubits)
            self.order = list(sorted_qubits)
            s = 1 + self.density_matrix
            self.tensor_shape = K.cast(s * self.nqubits * (2,), dtype='DTYPEINT')
            self.flat_shape = K.cast(s * (2 ** self.nqubits,), dtype='DTYPEINT')
            if self.density_matrix:
                self.order.extend((q + self.nqubits for q in sorted_qubits))
                self.order.extend((q for q in range(self.nqubits)
                                   if q not in sorted_qubits))
                self.order.extend((q + self.nqubits for q in range(self.nqubits)
                                   if q not in sorted_qubits))
            else:
                self.order.extend((q for q in range(self.nqubits)
                                   if q not in sorted_qubits))
        except (ValueError, OverflowError): # pragma: no cover
            pass

    def construct_unitary(self):
        cgates.M.construct_unitary(self)

    @staticmethod
    def _append_zeros(state, qubits: List[int], results: List[int]):
        for q, r in zip(qubits, results):
            state = K.expand_dims(state, axis=q)
            if r:
                state = K.concatenate([K.zeros_like(state), state], axis=q)
            else:
                state = K.concatenate([state, K.zeros_like(state)], axis=q)
        return state

    def state_vector_collapse(self, state, result):
        state = K.reshape(state, self.tensor_shape)
        substate = K.gather_nd(K.transpose(state, self.order), result)
        norm = K.sum(K.square(K.abs(substate)))
        state = substate / K.cast(K.sqrt(norm), dtype=state.dtype)
        state = self._append_zeros(state, sorted(self.target_qubits), result)
        return K.reshape(state, self.flat_shape)

    def density_matrix_collapse(self, state, result):
        density_matrix_result = 2 * result
        sorted_qubits = sorted(self.target_qubits)
        sorted_qubits = sorted_qubits + [q + self.nqubits for q in sorted_qubits]
        state = K.reshape(state, self.tensor_shape)
        substate = K.gather_nd(K.transpose(state, self.order),
                               density_matrix_result)
        n = 2 ** (len(tuple(substate.shape)) // 2)
        norm = K.trace(K.reshape(substate, (n, n)))
        state = substate / norm
        state = self._append_zeros(state, sorted_qubits, density_matrix_result)
        return K.reshape(state, self.flat_shape)

    def result_list(self):
        return cgates.M.result_list(self)

    def set_result(self, probs, nshots):
        return cgates.M.set_result(self, probs, nshots)

    def measure(self, state, nshots):
        return cgates.M.measure(self, state, nshots)

    def state_vector_call(self, state):
        return self.state_vector_collapse(state, self.result.binary[0])

    def density_matrix_call(self, state):
        return self.density_matrix_collapse(state, self.result_list())

    def __call__(self, state, nshots=1):
        return cgates.M.__call__(self, state, nshots)


class RX(BackendGate, gates.RX):

    def __init__(self, q, theta, trainable=True):
        BackendGate.__init__(self)
        gates.RX.__init__(self, q, theta, trainable)

    def construct_unitary(self):
        t = K.cast(self.parameters)
        return K.cos(t / 2.0) * K.matrices.I - 1j * K.sin(t / 2.0) * K.matrices.X


class RY(BackendGate, gates.RY):

    def __init__(self, q, theta, trainable=True):
        BackendGate.__init__(self)
        gates.RY.__init__(self, q, theta, trainable)

    def construct_unitary(self):
        t = K.cast(self.parameters)
        return K.cos(t / 2.0) * K.matrices.I - 1j * K.sin(t / 2.0) * K.matrices.Y


class RZ(BackendGate, gates.RZ):

    def __init__(self, q, theta, trainable=True):
        BackendGate.__init__(self)
        gates.RZ.__init__(self, q, theta, trainable)

    def construct_unitary(self):
        t = K.cast(self.parameters)
        phase = K.exp(1j * t / 2.0)[K.newaxis]
        diag = K.concatenate([K.conj(phase), phase], axis=0)
        return K.diag(diag)


class U1(BackendGate, gates.U1):

    def __init__(self, q, theta, trainable=True):
        BackendGate.__init__(self)
        gates.U1.__init__(self, q, theta, trainable)

    def construct_unitary(self):
        t = K.cast(self.parameters)
        phase = K.exp(1j * t)
        return K.diag([1, phase])


class U2(BackendGate, gates.U2):

    def __init__(self, q, phi, lam, trainable=True):
        BackendGate.__init__(self)
        gates.U2.__init__(self, q, phi, lam, trainable)

    def construct_unitary(self):
        return cgates.U2.construct_unitary(self)


class U3(BackendGate, gates.U3):

    def __init__(self, q, theta, phi, lam, trainable=True):
        BackendGate.__init__(self)
        gates.U3.__init__(self, q, theta, phi, lam, trainable=trainable)

    def construct_unitary(self):
        return cgates.U3.construct_unitary(self)


class CNOT(BackendGate, gates.CNOT):

    def __init__(self, q0, q1):
        BackendGate.__init__(self)
        gates.CNOT.__init__(self, q0, q1)

    def construct_unitary(self):
        return K.matrices.CNOT


class CZ(BackendGate, gates.CZ):

    def __init__(self, q0, q1):
        BackendGate.__init__(self)
        gates.CZ.__init__(self, q0, q1)

    def construct_unitary(self):
        return K.matrices.CZ


class _CUn_(BackendGate):
    base = U1

    def __init__(self, q0, q1, **params):
        cbase = "C{}".format(self.base.__name__)
        BackendGate.__init__(self)
        getattr(gates, cbase).__init__(self, q0, q1, **params)

    def construct_unitary(self):
        return BackendGate.control_unitary(self.base.construct_unitary(self))


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


class SWAP(BackendGate, gates.SWAP):

    def __init__(self, q0, q1):
        BackendGate.__init__(self)
        gates.SWAP.__init__(self, q0, q1)

    def construct_unitary(self):
        return K.cast([[1, 0, 0, 0], [0, 0, 1, 0],
                       [0, 1, 0, 0], [0, 0, 0, 1]])


class fSim(BackendGate, gates.fSim):

    def __init__(self, q0, q1, theta, phi, trainable=True):
        BackendGate.__init__(self)
        gates.fSim.__init__(self, q0, q1, theta, phi, trainable)

    def construct_unitary(self):
        return cgates.fSim.construct_unitary(self)


class GeneralizedfSim(BackendGate, gates.GeneralizedfSim):

    def __init__(self, q0, q1, unitary, phi, trainable=True):
        BackendGate.__init__(self)
        gates.GeneralizedfSim.__init__(self, q0, q1, unitary, phi, trainable)

    def construct_unitary(self):
        return cgates.GeneralizedfSim.construct_unitary(self)

    def _dagger(self) -> "GeneralizedfSim":
        return cgates.GeneralizedfSim._dagger(self)


class TOFFOLI(BackendGate, gates.TOFFOLI):

    def __init__(self, q0, q1, q2):
        BackendGate.__init__(self)
        gates.TOFFOLI.__init__(self, q0, q1, q2)

    def construct_unitary(self):
        return K.matrices.TOFFOLI

    @property
    def unitary(self):
        if self._unitary is None:
            self._unitary = self.construct_unitary()
        return self._unitary


class Unitary(BackendGate, gates.Unitary):

    def __init__(self, unitary, *q, trainable=True, name: Optional[str] = None):
        if not isinstance(unitary, K.tensor_types):
            raise_error(TypeError, "Unknown type {} of unitary matrix."
                                   "".format(type(unitary)))
        BackendGate.__init__(self)
        gates.Unitary.__init__(self, unitary, *q, trainable=trainable, name=name)

    def construct_unitary(self):
        unitary = self.parameters
        rank = int(math.log2(int(unitary.shape[0])))
        matrix = K.copy(K.cast(unitary))
        return matrix

    def _dagger(self) -> "Unitary":
        return cgates.Unitary._dagger(self)

    @ParametrizedGate.parameters.setter
    def parameters(self, x):
        cgates.Unitary.parameters.fset(self, x) # pylint: disable=no-member


class VariationalLayer(BackendGate, gates.VariationalLayer):

    @staticmethod
    def _tfkron(m1, m2):
        m = K.transpose(K.tensordot(m1, m2, axes=0), [0, 2, 1, 3])
        return K.reshape(m, (4, 4))

    def _calculate_unitaries(self):
        matrices = K.stack([self._tfkron(
            self.one_qubit_gate(q1, theta=self.params[q1]).unitary,
            self.one_qubit_gate(q2, theta=self.params[q2]).unitary)
                             for q1, q2 in self.pairs], axis=0)
        entangling_matrix = self.two_qubit_gate(0, 1).unitary
        matrices = K.matmul(entangling_matrix, matrices)

        additional_matrix = None
        q = self.additional_target
        if q is not None:
            additional_matrix = self.one_qubit_gate(
                q, theta=self.params[q]).unitary

        if self.params2:
            matrices2 = K.stack([self._tfkron(
                self.one_qubit_gate(q1, theta=self.params2[q1]).unitary,
                self.one_qubit_gate(q2, theta=self.params2[q2]).unitary)
                                for q1, q2 in self.pairs], axis=0)
            matrices = K.matmul(matrices2, matrices)

            q = self.additional_target
            if q is not None:
                additional_matrix = K.matmul(
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

    @BaseBackendGate.density_matrix.setter
    def density_matrix(self, x: bool):
        cgates.VariationalLayer.density_matrix.fset(self, x) # pylint: disable=no-member

    def _dagger(self):
        return cgates.VariationalLayer._dagger(self)

    def construct_unitary(self):
        return cgates.VariationalLayer.construct_unitary(self)

    def reprepare(self):
        cgates.VariationalLayer.reprepare(self)

    def prepare(self):
        cgates.VariationalLayer.prepare(self)

    def state_vector_call(self, state):
        return cgates.VariationalLayer.state_vector_call(self, state)

    def density_matrix_call(self, state):
        return cgates.VariationalLayer.density_matrix_call(self, state)


class PartialTrace(BackendGate, gates.PartialTrace):

    def __init__(self, *q):
        BackendGate.__init__(self)
        gates.PartialTrace.__init__(self, *q)

        self.traceout_string = None
        self.zero_matrix = None
        self.transpose_order = None
        self.output_shape = None

    def prepare(self):
        cgates.PartialTrace.prepare(self)

    def construct_unitary(self):
        cgates.PartialTrace.construct_unitary(self)

    def state_vector_partial_trace(self, state):
        return cgates.PartialTrace.state_vector_partial_trace(self, state)

    def density_matrix_partial_trace(self, state):
        return cgates.PartialTrace.density_matrix_partial_trace(self, state)

    def state_vector_call(self, state):
        return cgates.PartialTrace.state_vector_call(self, state)

    def density_matrix_call(self, state):
        return cgates.PartialTrace.density_matrix_call(self, state)


class KrausChannel(BackendGate, gates.KrausChannel):

    def __init__(self, ops):
        BackendGate.__init__(self)
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

    def state_vector_call(self, state):
        ccls = getattr(cgates, self.__class__.__name__)
        return ccls.state_vector_call(self, state)

    def density_matrix_call(self, state):
        new_state = K.zeros_like(state)
        for gate in self.gates:
            new_state += gate(state)
        return new_state


class UnitaryChannel(KrausChannel, gates.UnitaryChannel):

    def __init__(self, p: List[float], ops: List["Gate"],
                 seed: Optional[int] = None):
        BackendGate.__init__(self)
        gates.UnitaryChannel.__init__(self, p, ops, seed=seed)

    def prepare(self):
        KrausChannel.prepare(self)
        cgates.UnitaryChannel.set_seed(self)

    def density_matrix_call(self, state):
        new_state = (1 - self.psum) * state
        for p, gate in zip(self.probs, self.gates):
            new_state += p * gate(state)
        return new_state


class PauliNoiseChannel(UnitaryChannel, gates.PauliNoiseChannel):

    def __init__(self, q: int, px: float = 0, py: float = 0, pz: float = 0,
                 seed: Optional[int] = None):
        BackendGate.__init__(self)
        gates.PauliNoiseChannel.__init__(self, q, px, py, pz, seed=seed)


class ResetChannel(UnitaryChannel, gates.ResetChannel):

    def __init__(self, q: int, p0: float = 0.0, p1: float = 0.0,
                 seed: Optional[int] = None):
        BackendGate.__init__(self)
        gates.ResetChannel.__init__(self, q, p0=p0, p1=p1, seed=seed)

    def density_matrix_call(self, state):
        new_state = (1 - self.psum) * state
        for p, gate in zip(self.probs, self.gates):
            if isinstance(gate, M):
                state = gate.density_matrix_collapse(state, [0])
            else:
                state = gate(state)
            new_state += p * state
        return new_state


class _ThermalRelaxationChannelA(ResetChannel, gates._ThermalRelaxationChannelA):

    def calculate_probabilities(self, t1, t2, time, excited_population):
        cls = cgates.ThermalRelaxationChannel
        return cls.calculate_probabilities(self, t1, t2, time, excited_population)

    def __init__(self, q, t1, t2, time, excited_population=0, seed=None):
        BackendGate.__init__(self)
        gates._ThermalRelaxationChannelA.__init__(
            self, q, t1, t2, time, excited_population=excited_population,
            seed=seed)


class _ThermalRelaxationChannelB(BackendGate, gates._ThermalRelaxationChannelB):

    def calculate_probabilities(self, t1, t2, time, excited_population):
        cls = cgates.ThermalRelaxationChannel
        return cls.calculate_probabilities(self, t1, t2, time, excited_population)

    def __init__(self, q, t1, t2, time, excited_population=0, seed=None):
        BackendGate.__init__(self)
        gates._ThermalRelaxationChannelB.__init__(
            self, q, t1, t2, time, excited_population=excited_population,
            seed=seed)

    def prepare(self):
        self.is_prepared = True
        try:
            self.tensor_shape = K.cast(2 * self.nqubits * (2,), dtype='DTYPEINT')
            self.flat_shape = K.cast(2 * (2 ** self.nqubits,), dtype='DTYPEINT')
            self.reprepare()
            qubits = self.qubits + tuple(q + self.nqubits for q in self.qubits)
            self.calculation_cache = self.einsum.create_cache(
                qubits, 2 * self.nqubits)
            self.calculation_cache.cast_shapes(
                lambda x: K.cast(x, dtype='DTYPEINT'))
        except (ValueError, OverflowError): # pragma: no cover
            pass

    def construct_unitary(self):
        return cgates._ThermalRelaxationChannelB.construct_unitary(self)

    def state_vector_call(self, state):
        raise_error(ValueError, "Thermal relaxation cannot be applied to "
                                "state vectors when T1 < T2.")

    def density_matrix_call(self, state):
        einsum_str = self.calculation_cache.vector
        state = K.reshape(state, self.tensor_shape)
        state = self.einsum(einsum_str, state, self.matrix)
        return K.reshape(state, self.flat_shape)
