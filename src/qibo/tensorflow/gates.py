# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import gates as base_gates
from qibo.base import cache
from qibo.config import tfmatrices as matrices
from qibo.config import BACKEND, DTYPES, raise_error
from typing import Dict, List, Optional, Sequence, Tuple


class TensorflowGate(base_gates.Gate):
    """The base Tensorflow gate.

    **Properties:**
        matrix: The matrix that represents the gate to be applied.
            This is (2, 2) for 1-qubit gates and (4, 4) for 2-qubit gates.
        qubits: List with the qubits that the gate is applied to.
    """

    import sys
    module = sys.modules[__name__]

    def __init__(self):
        self.calculation_cache = None
        # For `controlled_by` gates (see `cache.ControlCache` for more details)
        self.control_cache = None
        # Gate matrices
        self.matrix = None
        # Einsum backend
        self.einsum = BACKEND.get('EINSUM')

    def _prepare(self):
        matrix = self.construct_unitary()
        rank = int(np.log2(int(matrix.shape[0])))
        self.matrix = tf.reshape(matrix, 2 * rank * (2,))

    def __matmul__(self, other: "TensorflowGate") -> "TensorflowGate":
        gate = base_gates.Gate.__matmul__(self, other)
        if gate is None:
            gate = Unitary(tf.matmul(self.unitary, other.unitary), *self.qubits)
        return gate

    @staticmethod
    def control_unitary(unitary: tf.Tensor) -> tf.Tensor:
        from qibo.tensorflow import cgates
        return cgates.TensorflowGate.control_unitary(unitary)

    @base_gates.Gate.nqubits.setter
    def nqubits(self, n: int):
        """Sets the number of qubit that this gate acts on.

        This is called automatically by the `Circuit.add` method if the gate
        is used on a `Circuit`. If the gate is called on a state then `nqubits`
        is set during the first `__call__`.
        When `nqubits` is set we also calculate the einsum string so that it
        is calculated only once per gate.
        """
        base_gates.Gate.nqubits.fset(self, n) # pylint: disable=no-member
        if self.is_controlled_by:
            if self.density_matrix:
                from qibo.tensorflow import einsum
                self.einsum = einsum.DefaultEinsum()
            self.control_cache = cache.ControlCache(self)
            nactive = n - len(self.control_qubits)
            targets = self.control_cache.targets
            self.calculation_cache = self.einsum.create_cache(targets, nactive, ncontrol=len(self.control_qubits))
        else:
            self.calculation_cache = self.einsum.create_cache(self.qubits, n)
        self.calculation_cache.cast_shapes(lambda x: tf.cast(x, dtype=DTYPES.get('DTYPEINT')))

    def _set_nqubits(self, state: tf.Tensor):
        """Sets ``gate.nqubits`` from state, if not already set."""
        if self._nqubits is None:
            self.nqubits = len(tuple(state.shape)) // (1 + self.density_matrix)

    def _state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        """Implements the gate on a state vector."""
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

    def _density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        """Implements the gate on a density matrix."""
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

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        """Implements the gate on a given state."""
        self._set_nqubits(state)
        return getattr(self, self._active_call)(state)


class H(TensorflowGate, base_gates.H):

    def __init__(self, q):
        base_gates.H.__init__(self, q)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        return matrices.H


class X(TensorflowGate, base_gates.X):

    def __init__(self, q):
        base_gates.X.__init__(self, q)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
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

    def construct_unitary(self) -> tf.Tensor:
        return matrices.Y


class Z(TensorflowGate, base_gates.Z):

    def __init__(self, q):
        base_gates.Z.__init__(self, q)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        return matrices.Z


class Collapse(TensorflowGate, base_gates.Collapse):

    def __init__(self, *q: int, result: List[int] = 0):
        base_gates.Collapse.__init__(self, *q, result=result)
        TensorflowGate.__init__(self)
        self.order = None
        self.ids = None
        self.dm_result = None

    @staticmethod
    def _result_to_list(res):
        from qibo.tensorflow.cgates import Collapse
        return Collapse._result_to_list(res)

    def _prepare(self):
        self.order = list(self.sorted_qubits)
        if self.density_matrix:
            self.order.extend((q + self.nqubits for q in self.sorted_qubits))
            self.order.extend((q for q in range(self.nqubits)
                               if q not in self.sorted_qubits))
            self.order.extend((q + self.nqubits for q in range(self.nqubits)
                               if q not in self.sorted_qubits))
            self.sorted_qubits += [q + self.nqubits for q in self.sorted_qubits]
            self.dm_result = 2 * self.result
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

    def _state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        substate = tf.gather_nd(tf.transpose(state, self.order), self.result)
        norm = tf.reduce_sum(tf.square(tf.abs(substate)))
        state = substate / tf.cast(tf.sqrt(norm), dtype=state.dtype)
        return self._append_zeros(state, self.sorted_qubits, self.result)

    def _density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        substate = tf.gather_nd(tf.transpose(state, self.order), self.dm_result)
        n = 2 ** (len(tuple(substate.shape)) // 2)
        norm = tf.linalg.trace(tf.reshape(substate, (n, n)))
        state = substate / norm
        return self._append_zeros(state, self.sorted_qubits, self.dm_result)


class RX(TensorflowGate, base_gates.RX):

    def __init__(self, q, theta):
        base_gates.RX.__init__(self, q, theta)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        t = tf.cast(self.parameter, dtype=DTYPES.get('DTYPECPX'))
        return tf.cos(t / 2.0) * matrices.I - 1j * tf.sin(t / 2.0) * matrices.X


class RY(TensorflowGate, base_gates.RY):

    def __init__(self, q, theta):
        base_gates.RY.__init__(self, q, theta)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        t = tf.cast(self.parameter, dtype=DTYPES.get('DTYPECPX'))
        return tf.cos(t / 2.0) * matrices.I - 1j * tf.sin(t / 2.0) * matrices.Y


class RZ(TensorflowGate, base_gates.RZ):

    def __init__(self, q, theta):
        base_gates.RZ.__init__(self, q, theta)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        t = tf.cast(self.parameter, dtype=DTYPES.get('DTYPECPX'))
        phase = tf.exp(1j * t / 2.0)[tf.newaxis]
        diag = tf.concat([tf.math.conj(phase), phase], axis=0)
        return tf.linalg.diag(diag)


class U1(TensorflowGate, base_gates.U1):

    def __init__(self, q, theta):
        base_gates.U1.__init__(self, q, theta)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        t = tf.cast(self.parameter, dtype=DTYPES.get('DTYPECPX'))
        phase = tf.exp(1j * t)
        return tf.linalg.diag([1, phase])


class U2(TensorflowGate, base_gates.U2):

    def __init__(self, q, phi, lam):
        base_gates.U2.__init__(self, q, phi, lam)
        TensorflowGate.__init__(self)

    def construct_unitary(self):
        from qibo.tensorflow import cgates
        return cgates.U2.construct_unitary(self)


class U3(TensorflowGate, base_gates.U3):

    def __init__(self, q, theta, phi, lam):
        base_gates.U3.__init__(self, q, theta, phi, lam)
        TensorflowGate.__init__(self)

    def construct_unitary(self):
        from qibo.tensorflow import cgates
        return cgates.U3.construct_unitary(self)


class CNOT(TensorflowGate, base_gates.CNOT):

    def __init__(self, q0, q1):
        base_gates.CNOT.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        return tf.reshape(matrices.CNOT, (4, 4))


class CZ(TensorflowGate, base_gates.CZ):

    def __init__(self, q0, q1):
        base_gates.CZ.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        diag = tf.cast(tf.concat([tf.ones(3), [-1]], axis=0),
                       dtype=DTYPES.get('DTYPECPX'))
        return tf.linalg.diag(diag)


class _CUn_(TensorflowGate):
    base = U1

    def __init__(self, q0, q1, **params):
        cbase = "C{}".format(self.base.__name__)
        getattr(base_gates, cbase).__init__(self, q0, q1, **params)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        return TensorflowGate.control_unitary(self.base.construct_unitary(self))


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


class CU2(_CUn_, base_gates.CU2):
    base = U2

    def __init__(self, q0, q1, phi, lam):
        _CUn_.__init__(self, q0, q1, phi=phi, lam=lam)


class CU3(_CUn_, base_gates.CU3):
    base = U3

    def __init__(self, q0, q1, theta, phi, lam):
        _CUn_.__init__(self, q0, q1, theta=theta, phi=phi, lam=lam)


class SWAP(TensorflowGate, base_gates.SWAP):

    def __init__(self, q0, q1):
        base_gates.SWAP.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        return tf.cast([[1, 0, 0, 0], [0, 0, 1, 0],
                        [0, 1, 0, 0], [0, 0, 0, 1]],
                       dtype=DTYPES.get('DTYPECPX'))


class fSim(TensorflowGate, base_gates.fSim):

    def __init__(self, q0, q1, theta, phi):
        base_gates.fSim.__init__(self, q0, q1, theta, phi)
        TensorflowGate.__init__(self)

    def construct_unitary(self):
        from qibo.tensorflow import cgates
        return cgates.fSim.construct_unitary(self)


class GeneralizedfSim(TensorflowGate, base_gates.GeneralizedfSim):

    def __init__(self, q0, q1, unitary, phi):
        base_gates.GeneralizedfSim.__init__(self, q0, q1, unitary, phi)
        TensorflowGate.__init__(self)

    def construct_unitary(self):
        from qibo.tensorflow import cgates
        return cgates.GeneralizedfSim.construct_unitary(self)

    def _dagger(self) -> "GeneralizedfSim":
        from qibo.tensorflow import cgates
        return cgates.GeneralizedfSim._dagger(self)


class TOFFOLI(TensorflowGate, base_gates.TOFFOLI):

    def __init__(self, q0, q1, q2):
        base_gates.TOFFOLI.__init__(self, q0, q1, q2)
        TensorflowGate.__init__(self)

    def construct_unitary(self) -> tf.Tensor:
        return tf.reshape(matrices.TOFFOLI, (8, 8))


class Unitary(TensorflowGate, base_gates.Unitary):

    def __init__(self, unitary, *q, name: Optional[str] = None):
        if not isinstance(unitary, (np.ndarray, tf.Tensor)):
            raise_error(TypeError, "Unknown type {} of unitary matrix."
                                   "".format(type(unitary)))
        base_gates.Unitary.__init__(self, unitary, *q, name=name)
        TensorflowGate.__init__(self)
        self._unitary = self.construct_unitary()

    def construct_unitary(self):
        unitary = self.parameter
        rank = int(np.log2(int(unitary.shape[0])))
        dtype = DTYPES.get('DTYPECPX')
        if isinstance(unitary, tf.Tensor):
            matrix = tf.identity(tf.cast(unitary, dtype=dtype))
        elif isinstance(unitary, np.ndarray):
            matrix = tf.convert_to_tensor(unitary, dtype=dtype)
        return matrix

    def _dagger(self) -> "Unitary":
        from qibo.tensorflow import cgates
        return cgates.Unitary._dagger(self)


class VariationalLayer(TensorflowGate, base_gates.VariationalLayer):

    from qibo.tensorflow import cgates

    def __init__(self, qubits: List[int], pairs: List[Tuple[int, int]],
                 one_qubit_gate, two_qubit_gate,
                 params: List[float],
                 params2: Optional[List[float]] = None,
                 name: Optional[str] = None):
        base_gates.VariationalLayer.__init__(self, qubits, pairs,
                                             one_qubit_gate, two_qubit_gate,
                                             params, params2,
                                             name=name)
        TensorflowGate.__init__(self)

    def _unitary_constructor(self, matrix, *targets):
        gate = Unitary(matrix, *targets)
        gate.density_matrix = self.density_matrix
        return gate

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

    def _prepare(self):
        self.cgates.VariationalLayer._prepare(self)

    def __call__(self, state: tf.Tensor) -> tf.Tensor: # pragma: no cover
        # impractical case because VariationalLayer is not called by circuits
        return self.cgates.VariationalLayer.__call__(self, state)


class TensorflowChannel(TensorflowGate):
    """Base Tensorflow channels.

    All channels should inherit this class.
    """

    def __init__(self):
        super(TensorflowChannel, self).__init__()

    def _prepare(self):
        from qibo.tensorflow import cgates
        cgates.TensorflowChannel._prepare(self)

    def _state_vector_call(self, state: tf.Tensor) -> tf.Tensor:
        raise_error(ValueError, "Channels cannot be used on state vectors.")

    def _density_matrix_call(self, state: tf.Tensor) -> tf.Tensor: # pragma: no cover
        # abstract method
        raise_error(NotImplementedError)


class NoiseChannel(TensorflowChannel, base_gates.NoiseChannel):

    def __init__(self, q: int, px: float = 0, py: float = 0, pz: float = 0):
        base_gates.NoiseChannel.__init__(self, q, px, py, pz)
        TensorflowChannel.__init__(self)

    def _density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        new_state = tf.zeros_like(state)
        for p, gate in self.gates:
            new_state += p * gate(state)
        return (1 - self.total_p) * state + new_state


class GeneralChannel(TensorflowChannel, base_gates.GeneralChannel):

    def __init__(self, A: Sequence[Tuple[Tuple[int], np.ndarray]]):
        base_gates.GeneralChannel.__init__(self, A)
        TensorflowChannel.__init__(self)

    def _density_matrix_call(self, state: tf.Tensor) -> tf.Tensor:
        new_state = tf.zeros_like(state)
        for gate in self.gates:
            new_state += gate(state)
        return new_state
