# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import sys
import math
from qibo import K
from qibo.abstractions import gates as abstract_gates
from qibo.abstractions.abstract_gates import BaseBackendGate, ParametrizedGate
from qibo.config import raise_error
from typing import Dict, List, Optional, Sequence, Tuple


class BackendGate(BaseBackendGate):
    module = sys.modules[__name__]

    def __init__(self):
        if K.name == "custom":
            if not K.executing_eagerly():
                raise_error(NotImplementedError,
                            "Custom operator gates should not be used in "
                            "compiled mode.")
            self.gate_op = K.op.apply_gate
        else:
            self.gate_op = None
        super().__init__()

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

    @property
    def cache(self):
        if self._cache is None:
            self._cache = K.create_gate_cache(self)
        return self._cache

    @property
    def matrix(self):
        if self._matrix is None:
            self._matrix = self.calculate_matrix()
        return self._matrix

    def calculate_matrix(self):
        matrix = self.construct_unitary()
        if K.name != "custom":
            rank = int(math.log2(int(matrix.shape[0])))
            matrix = K.reshape(matrix, 2 * rank * (2,))
        return matrix

    def set_nqubits(self, state):
        if self._nqubits is None:
            self.nqubits = int(math.log2(tuple(state.shape)[0]))

    def state_vector_call(self, state):
        return K.state_vector_call(self, state)

    def density_matrix_call(self, state):
        return K.density_matrix_call(self, state)


class MatrixGate(BackendGate):
    """Gate that uses matrix multiplication to be applied to states."""

    def state_vector_call(self, state):
        return K.state_vector_matrix_call(self, state)

    def density_matrix_call(self, state):
        return K.density_matrix_matrix_call(self, state)


class H(MatrixGate, abstract_gates.H):

    def __init__(self, q):
        MatrixGate.__init__(self)
        abstract_gates.H.__init__(self, q)

    def construct_unitary(self):
        return K.matrices.H


class X(BackendGate, abstract_gates.X):

    def __init__(self, q):
        BackendGate.__init__(self)
        abstract_gates.X.__init__(self, q)
        if self.gate_op:
            self.gate_op = K.op.apply_x

    def construct_unitary(self):
        return K.matrices.X


class Y(BackendGate, abstract_gates.Y):

    def __init__(self, q):
        BackendGate.__init__(self)
        abstract_gates.Y.__init__(self, q)
        if self.gate_op:
            self.gate_op = K.op.apply_y
            self.density_matrix_call = lambda state: self._custom_density_matrix_call(state)

    def construct_unitary(self):
        return K.matrices.Y

    def _custom_density_matrix_call(self, state):
        state = self.gate_op(state, self.cache.qubits_tensor + self.nqubits,
                             2 * self.nqubits, *self.target_qubits,
                             K.get_threads())
        matrix = K.conj(K.matrices.Y)
        state = K.op.apply_gate(state, matrix, self.cache.qubits_tensor,
                                2 * self.nqubits, *self.cache.target_qubits_dm,
                                K.get_threads())
        return state


class Z(BackendGate, abstract_gates.Z):

    def __init__(self, q):
        BackendGate.__init__(self)
        abstract_gates.Z.__init__(self, q)
        if self.gate_op:
            self.gate_op = K.op.apply_z

    def construct_unitary(self):
        return K.matrices.Z


class I(BackendGate, abstract_gates.I):

    def __init__(self, *q):
        BackendGate.__init__(self)
        abstract_gates.I.__init__(self, *q)

    def construct_unitary(self):
        return K.eye(2 ** len(self.target_qubits))

    def state_vector_call(self, state):
        return state

    def density_matrix_call(self, state):
        return state


class M(BackendGate, abstract_gates.M):
    from qibo.core import measurements, states

    def __init__(self, *q, register_name: Optional[str] = None,
                 collapse: bool = False,
                 p0: Optional["ProbsType"] = None,
                 p1: Optional["ProbsType"] = None):
        BackendGate.__init__(self)
        abstract_gates.M.__init__(self, *q, register_name=register_name,
                                  collapse=collapse, p0=p0, p1=p1)
        self.result = None
        self._result_list = None
        self._result_tensor = None
        if collapse:
            self.result = self.measurements.MeasurementResult(self.qubits)
        if self.gate_op:
            self.gate_op = K.op.collapse_state
        self.order = None

    @property
    def cache(self):
        if self._cache is None:
            cache = K.create_gate_cache(self)
            target_qubits = set(self.target_qubits)
            unmeasured_qubits = []
            reduced_target_qubits = dict()
            for i in range(self.nqubits):
                if i in target_qubits:
                    reduced_target_qubits[i] = i - len(unmeasured_qubits)
                else:
                    unmeasured_qubits.append(i)
            cache.unmeasured_qubits = tuple(unmeasured_qubits)
            cache.reduced_target_qubits = list(
                reduced_target_qubits[i] for i in self.target_qubits)

            if K.name != "custom":
                sorted_qubits = sorted(self.target_qubits)
                cache.order = list(sorted_qubits)
                s = 1 + self.density_matrix
                cache.tensor_shape = K.cast(s * self.nqubits * (2,), dtype='DTYPEINT')
                cache.flat_shape = K.cast(s * (2 ** self.nqubits,), dtype='DTYPEINT')
                if self.density_matrix:
                    cache.order.extend((q + self.nqubits for q in sorted_qubits))
                    cache.order.extend((q for q in range(self.nqubits)
                                        if q not in sorted_qubits))
                    cache.order.extend((q + self.nqubits for q in range(self.nqubits)
                                        if q not in sorted_qubits))
                else:
                    cache.order.extend((q for q in range(self.nqubits)
                                        if q not in sorted_qubits))
            self._cache = cache
        return self._cache

    def construct_unitary(self):
        raise_error(ValueError, "Measurement gate does not have unitary "
                                "representation.")

    def symbol(self):
        if self._symbol is None:
            from qibo.core.measurements import MeasurementSymbol
            self._symbol = MeasurementSymbol(self.result)
        return self._symbol

    def result_list(self):
        if self._result_list is None:
            pairs = zip(self.target_qubits, self.result.binary[-1])
            resdict = {q: r for q, r in pairs}
            self._result_list = [resdict[q] for q in sorted(self.target_qubits)]
        return self._result_list

    def result_tensor(self):
        if self._result_tensor is None:
            n = len(self.result_list())
            result = sum(2 ** (n - i - 1) * r
                         for i, r in enumerate(self.result_list()))
            self._result_tensor = K.cast(result, dtype='DTYPEINT')
        return self._result_tensor

    def measure(self, state, nshots):
        if isinstance(state, K.tensor_types):
            self.set_nqubits(state)
            if self.density_matrix:
                state = self.states.MatrixState.from_tensor(state)
            else:
                state = self.states.VectorState.from_tensor(state)
        elif isinstance(state, self.states.AbstractState):
            self.set_nqubits(state.tensor)
        else:
            raise_error(TypeError, "Measurement gate called on state of type "
                                   "{} that is not supported."
                                   "".format(type(state)))

        def calculate_probs():
            probs_dim = K.cast((2 ** len(self.target_qubits),), dtype='DTYPEINT')
            probs = state.probabilities(measurement_gate=self)
            probs = K.transpose(probs, axes=self.cache.reduced_target_qubits)
            probs = K.reshape(probs, probs_dim)
            return probs

        probs = K.cpu_fallback(calculate_probs)
        if self.collapse:
            self._result_list = None
            self._result_tensor = None
            self.result.add_shot(probs)
            # optional bitflip noise
            if sum(sum(x.values()) for x in self.bitflip_map) > 0:
                noisy_result = self.result.apply_bitflips(*self.bitflip_map)
                self.result.binary = noisy_result.binary
            return self.result

        result = self.measurements.MeasurementResult(self.qubits, probs, nshots)
        if sum(sum(x.values()) for x in self.bitflip_map) > 0:
            result = result.apply_bitflips(*self.bitflip_map)
        return result

    def state_vector_call(self, state):
        if K.name == "custom":
            result = self.result_tensor()
        else:
            result = self.result.binary[-1]
        return K.state_vector_collapse(self, state, result)

    def density_matrix_call(self, state):
        if K.name == "custom":
            result = self.result_tensor()
        else:
            result = self.result_list()
        return K.density_matrix_collapse(self, state, result)

    def __call__(self, state, nshots=1):
        self.result = self.measure(state, nshots)
        if self.collapse:
            if nshots > 1:
                raise_error(ValueError, "Cannot perform measurement collapse "
                                        "for more than one shots.")
            return getattr(self, self._active_call)(state)
        return self.result


class RX(MatrixGate, abstract_gates.RX):

    def __init__(self, q, theta, trainable=True):
        MatrixGate.__init__(self)
        abstract_gates.RX.__init__(self, q, theta, trainable)

    def construct_unitary(self):
        theta = self.parameters
        if isinstance(theta, K.native_types): # pragma: no cover
            p = K
            theta = K.cast(theta)
        else:
            p = K.qnp
        cos, isin = p.cos(theta / 2.0) + 0j, -1j * p.sin(theta / 2.0)
        return K.cast([[cos, isin], [isin, cos]])


class RY(MatrixGate, abstract_gates.RY):

    def __init__(self, q, theta, trainable=True):
        MatrixGate.__init__(self)
        abstract_gates.RY.__init__(self, q, theta, trainable)

    def construct_unitary(self):
        theta = self.parameters
        if isinstance(theta, K.native_types):
            p = K
            theta = K.cast(theta)
        else:
            p = K.qnp
        cos, sin = p.cos(theta / 2.0), p.sin(theta / 2.0)
        return K.cast([[cos, -sin], [sin, cos]])


class RZ(MatrixGate, abstract_gates.RZ):

    def __init__(self, q, theta, trainable=True):
        MatrixGate.__init__(self)
        abstract_gates.RZ.__init__(self, q, theta, trainable)

    def construct_unitary(self):
        if isinstance(self.parameters, K.native_types): # pragma: no cover
            p = K
            theta = K.cast(self.parameters)
        else:
            p = K.qnp
            theta = self.parameters
        phase = p.exp(1j * theta / 2.0)
        return K.cast(p.diag([p.conj(phase), phase]))


class U1(MatrixGate, abstract_gates.U1):

    def __init__(self, q, theta, trainable=True):
        MatrixGate.__init__(self)
        abstract_gates.U1.__init__(self, q, theta, trainable)
        if self.gate_op:
            self.gate_op = K.op.apply_z_pow

    def calculate_matrix(self):
        if K.name == "custom":
            return K.cast(K.qnp.exp(1j * self.parameters))
        else:
            return MatrixGate.calculate_matrix(self)

    def construct_unitary(self):
        if isinstance(self.parameters, K.native_types): # pragma: no cover
            p = K
            theta = K.cast(self.parameters)
        else:
            p = K.qnp
            theta = self.parameters
        return p.diag([1, p.exp(1j * theta)])


class U2(MatrixGate, abstract_gates.U2):

    def __init__(self, q, phi, lam, trainable=True):
        MatrixGate.__init__(self)
        abstract_gates.U2.__init__(self, q, phi, lam, trainable)

    def construct_unitary(self):
        phi, lam = self.parameters
        if isinstance(phi, K.native_types) or isinstance(lam, K.native_types): # pragma: no cover
            p = K
        else:
            p = K.qnp
        eplus = p.exp(1j * (phi + lam) / 2.0)
        eminus = p.exp(1j * (phi - lam) / 2.0)
        return K.cast([[p.conj(eplus), - p.conj(eminus)],
                       [eminus, eplus]]) / p.sqrt(2)


class U3(MatrixGate, abstract_gates.U3):

    def __init__(self, q, theta, phi, lam, trainable=True):
        MatrixGate.__init__(self)
        abstract_gates.U3.__init__(self, q, theta, phi, lam, trainable)

    def construct_unitary(self):
        theta, phi, lam = self.parameters
        if isinstance(theta, K.native_types) or isinstance(phi, K.native_types) or isinstance(lam, K.native_types): # pragma: no cover
            p = K
        else:
            p = K.qnp
        cost, sint = p.cos(theta / 2), p.sin(theta / 2)
        eplus, eminus = p.exp(1j * (phi + lam) / 2.0), p.exp(1j * (phi - lam) / 2.0)
        return K.cast([[p.conj(eplus) * cost, - p.conj(eminus) * sint],
                       [eminus * sint, eplus * cost]])


class ZPow(abstract_gates.ZPow):

    def __new__(cls, q, theta, trainable=True):
        return U1(q, theta, trainable)


class CNOT(BackendGate, abstract_gates.CNOT):

    def __init__(self, q0, q1):
        BackendGate.__init__(self)
        abstract_gates.CNOT.__init__(self, q0, q1)
        if self.gate_op:
            self.gate_op = K.op.apply_x

    def construct_unitary(self):
        return K.matrices.CNOT


class CZ(BackendGate, abstract_gates.CZ):

    def __init__(self, q0, q1):
        BackendGate.__init__(self)
        abstract_gates.CZ.__init__(self, q0, q1)
        if self.gate_op:
            self.gate_op = K.op.apply_z

    def construct_unitary(self):
        return K.matrices.CZ


class _CUn_(MatrixGate):
    base = U1

    def __init__(self, q0, q1, **params):
        MatrixGate.__init__(self)
        cbase = "C{}".format(self.base.__name__)
        getattr(abstract_gates, cbase).__init__(self, q0, q1, **params)

    def calculate_matrix(self):
        matrix = self.base.construct_unitary(self)
        if K.name != "custom":
            matrix = K.reshape(self.control_unitary(matrix), 4 * (2,))
        return matrix

    def construct_unitary(self):
        return MatrixGate.control_unitary(self.base.construct_unitary(self))


class CRX(_CUn_, abstract_gates.CRX):
    base = RX

    def __init__(self, q0, q1, theta, trainable=True):
        _CUn_.__init__(self, q0, q1, theta=theta, trainable=trainable)


class CRY(_CUn_, abstract_gates.CRY):
    base = RY

    def __init__(self, q0, q1, theta, trainable=True):
        _CUn_.__init__(self, q0, q1, theta=theta, trainable=trainable)


class CRZ(_CUn_, abstract_gates.CRZ):
    base = RZ

    def __init__(self, q0, q1, theta, trainable=True):
        _CUn_.__init__(self, q0, q1, theta=theta, trainable=trainable)


class CU1(_CUn_, abstract_gates.CU1):
    base = U1

    def __init__(self, q0, q1, theta, trainable=True):
        _CUn_.__init__(self, q0, q1, theta=theta, trainable=trainable)
        if self.gate_op:
            self.gate_op = K.op.apply_z_pow

    def calculate_matrix(self):
        return U1.calculate_matrix(self)


class CU2(_CUn_, abstract_gates.CU2):
    base = U2

    def __init__(self, q0, q1, phi, lam, trainable=True):
        _CUn_.__init__(self, q0, q1, phi=phi, lam=lam, trainable=trainable)


class CU3(_CUn_, abstract_gates.CU3):
    base = U3

    def __init__(self, q0, q1, theta, phi, lam, trainable=True):
        _CUn_.__init__(self, q0, q1, theta=theta, phi=phi, lam=lam,
                       trainable=trainable)


class CZPow(abstract_gates.CZPow):

    def __new__(cls, q0, q1, theta, trainable=True):
        return CU1(q0, q1, theta, trainable)


class SWAP(BackendGate, abstract_gates.SWAP):

    def __init__(self, q0, q1):
        BackendGate.__init__(self)
        abstract_gates.SWAP.__init__(self, q0, q1)
        if self.gate_op:
            self.gate_op = K.op.apply_swap

    def construct_unitary(self):
        return K.matrices.SWAP


class fSim(MatrixGate, abstract_gates.fSim):

    def __init__(self, q0, q1, theta, phi, trainable=True):
        MatrixGate.__init__(self)
        abstract_gates.fSim.__init__(self, q0, q1, theta, phi, trainable)
        if self.gate_op:
            self.gate_op = K.op.apply_fsim

    def calculate_matrix(self):
        if K.name == "custom":
            theta, phi = self.parameters
            cos, isin = K.qnp.cos(theta) + 0j, -1j * K.qnp.sin(theta)
            phase = K.qnp.exp(-1j * phi)
            return K.cast([cos, isin, isin, cos, phase])
        return MatrixGate.calculate_matrix(self)

    def construct_unitary(self):
        theta, phi = self.parameters
        if isinstance(theta, K.native_types) or isinstance(phi, K.native_types): # pragma: no cover
            p = K
        else:
            p = K.qnp
        cos, isin = p.cos(theta), -1j * p.sin(theta)
        matrix = p.eye(4)
        matrix[1, 1], matrix[2, 2] = cos, cos
        matrix[1, 2], matrix[2, 1] = isin, isin
        matrix[3, 3] = p.exp(-1j * phi)
        return K.cast(matrix)


class GeneralizedfSim(MatrixGate, abstract_gates.GeneralizedfSim):

    def __init__(self, q0, q1, unitary, phi, trainable=True):
        BackendGate.__init__(self)
        abstract_gates.GeneralizedfSim.__init__(self, q0, q1, unitary, phi, trainable)
        if self.gate_op:
            self.gate_op = K.op.apply_fsim

    def calculate_matrix(self):
        if K.name == "custom":
            unitary, phi = self.parameters
            matrix = K.qnp.zeros(5)
            matrix[:4] = K.qnp.reshape(unitary, (4,))
            matrix[4] = K.qnp.exp(-1j * phi)
            return K.cast(matrix)
        return MatrixGate.calculate_matrix(self)

    def construct_unitary(self):
        unitary, phi = self.parameters
        if isinstance(unitary, K.native_types) or isinstance(phi, K.native_types): # pragma: no cover
            p = K
        else:
            p = K.qnp
        matrix = p.eye(4)
        matrix[1:3, 1:3] = p.reshape(unitary, (2, 2))
        matrix[3, 3] = p.exp(-1j * phi)
        return K.cast(matrix)

    def _dagger(self) -> "GenerelizedfSim":
        unitary, phi = self.parameters
        if isinstance(unitary, K.native_types):
            ud = K.conj(K.transpose(unitary))
        else:
            ud = unitary.conj().T
        q0, q1 = self.target_qubits
        return self.__class__(q0, q1, ud, -phi)


class TOFFOLI(BackendGate, abstract_gates.TOFFOLI):

    def __init__(self, q0, q1, q2):
        BackendGate.__init__(self)
        abstract_gates.TOFFOLI.__init__(self, q0, q1, q2)
        if self.gate_op:
            self.gate_op = K.op.apply_x

    def construct_unitary(self):
        return K.matrices.TOFFOLI

    @property
    def unitary(self):
        if self._unitary is None:
            self._unitary = self.construct_unitary()
        return self._unitary


class Unitary(MatrixGate, abstract_gates.Unitary):

    def __init__(self, unitary, *q, trainable=True, name: Optional[str] = None):
        if not isinstance(unitary, K.tensor_types):
            raise_error(TypeError, "Unknown type {} of unitary matrix."
                                   "".format(type(unitary)))
        MatrixGate.__init__(self)
        abstract_gates.Unitary.__init__(self, unitary, *q, trainable=trainable, name=name)
        rank = self.rank
        if self.gate_op:
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
        if len(shape) > 2 and shape[0] == 1:
            shape = shape[1:]
            x = K.squeeze(x, axis=0)
        true_shape = (2 ** self.rank, 2 ** self.rank)
        if shape == true_shape:
            ParametrizedGate.parameters.fset(self, x) # pylint: disable=no-member
        elif shape == (2 ** (2 * self.rank),):
            ParametrizedGate.parameters.fset(self, x.reshape(true_shape)) # pylint: disable=no-member
        else:
            raise_error(ValueError, "Invalid shape {} of unitary matrix "
                                    "acting on {} target qubits."
                                    "".format(shape, self.rank))


class VariationalLayer(BackendGate, abstract_gates.VariationalLayer):

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
        BackendGate.__init__(self)
        abstract_gates.VariationalLayer.__init__(self, qubits, pairs,
                                                 one_qubit_gate, two_qubit_gate,
                                                 params, params2,
                                                 trainable=trainable, name=name)

        matrices, additional_matrix = self._calculate_unitaries()
        self.unitaries = [Unitary(matrix, *targets)
                          for targets, matrix in zip(self.pairs, matrices)]
        if self.additional_target is not None:
            self.additional_unitary = Unitary(
                additional_matrix, self.additional_target)
            self.additional_unitary.density_matrix = self.density_matrix

    @BaseBackendGate.density_matrix.setter
    def density_matrix(self, x: bool):
        BaseBackendGate.density_matrix.fset(self, x) # pylint: disable=no-member
        for unitary in self.unitaries:
            unitary.density_matrix = x
        if self.additional_unitary is not None:
            self.additional_unitary.density_matrix = x

    @ParametrizedGate.parameters.setter
    def parameters(self, x):
        abstract_gates.VariationalLayer.parameters.fset(self, x) # pylint: disable=no-member
        if self.unitaries:
            matrices, additional_matrix = self._calculate_unitaries()
            for gate, matrix in zip(self.unitaries, matrices):
                gate.parameters = matrix
        if self.additional_unitary is not None:
            self.additional_unitary.parameters = additional_matrix

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

    def state_vector_call(self, state):
        for i, unitary in enumerate(self.unitaries):
            state = unitary(state)
        if self.additional_unitary is not None:
            state = self.additional_unitary(state)
        return state

    def density_matrix_call(self, state):
        return self.state_vector_call(state)


class Flatten(BackendGate, abstract_gates.Flatten):

    def __init__(self, coefficients):
        BackendGate.__init__(self)
        abstract_gates.Flatten.__init__(self, coefficients)
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


class CallbackGate(BackendGate, abstract_gates.CallbackGate):

    def __init__(self, callback):
        BackendGate.__init__(self)
        abstract_gates.CallbackGate.__init__(self, callback)
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


class PartialTrace(BackendGate, abstract_gates.PartialTrace):

    def __init__(self, *q):
        BackendGate.__init__(self)
        abstract_gates.PartialTrace.__init__(self, *q)

    class GateCache:
        pass

    @property
    def cache(self):
        if self._cache is None:
            cache = self.GateCache()
            qubits = set(self.target_qubits)
            # Create |00...0><00...0| for qubits that are traced out
            n = len(self.target_qubits)
            row0 = K.cast([1] + (2 ** n - 1) * [0], dtype='DTYPECPX')
            shape = K.cast((2 ** n - 1, 2 ** n), dtype='DTYPEINT')
            rows = K.zeros(shape, dtype='DTYPECPX')
            cache.zero_matrix = K.concatenate([row0[K.newaxis], rows], axis=0)
            cache.zero_matrix = K.reshape(cache.zero_matrix, 2 * n * (2,))
            # Calculate initial transpose order
            order = tuple(sorted(self.target_qubits))
            order += tuple(i for i in range(self.nqubits) if i not in qubits)
            order += tuple(i + self.nqubits for i in order)
            cache.einsum_order = order
            # Calculate final transpose order
            order1 = tuple(i for i in range(self.nqubits) if i not in qubits)
            order2 = tuple(self.target_qubits)
            order = (order1 + tuple(i + self.nqubits for i in order1) +
                     order2 + tuple(i + self.nqubits for i in order2))
            cache.final_order = tuple(order.index(i) for i in range(2 * self.nqubits))
            # Shapes
            cache.einsum_shape = K.cast(2 * (2 ** n, 2 ** (self.nqubits - n)), dtype='DTYPEINT')
            cache.output_shape = K.cast(2 * (2 ** self.nqubits,), dtype='DTYPEINT')
            cache.reduced_shape = K.cast(2 * (2 ** (self.nqubits - n),), dtype='DTYPEINT')
            self._cache = cache
        return self._cache

    def construct_unitary(self):
        raise_error(ValueError, "Partial trace gate does not have unitary "
                                "representation.")

    def state_vector_partial_trace(self, state):
        self.set_nqubits(state)
        state = K.reshape(state, self.nqubits * (2,))
        axes = 2 * [list(self.target_qubits)]
        rho = K.tensordot(state, K.conj(state), axes=axes)
        return K.reshape(rho, self.cache.reduced_shape)

    def density_matrix_partial_trace(self, state):
        self.set_nqubits(state)
        state = K.reshape(state, 2 * self.nqubits * (2,))
        state = K.transpose(state, self.cache.einsum_order)
        state = K.reshape(state, self.cache.einsum_shape)
        return K.einsum("abac->bc", state)

    def state_vector_call(self, state):
        raise_error(RuntimeError, "Partial trace gate cannot be used on state "
                                  "vectors. Please switch to density matrix "
                                  "simulation.")

    def density_matrix_call(self, state):
        substate = self.density_matrix_partial_trace(state)
        n = self.nqubits - len(self.target_qubits)
        substate = K.reshape(substate, 2 * n * (2,))
        state = K.tensordot(substate, self.cache.zero_matrix, axes=0)
        state = K.transpose(state, self.cache.final_order)
        return K.reshape(state, self.cache.output_shape)


class KrausChannel(BackendGate, abstract_gates.KrausChannel):

    def __init__(self, ops):
        BackendGate.__init__(self)
        abstract_gates.KrausChannel.__init__(self, ops)

    def calculate_inverse_gates(self):
        inv_gates = []
        for gate in self.gates[:-1]:
            matrix = gate.parameters
            if isinstance(matrix, K.tensor_types):
                inv_matrix = K.qnp.inv(matrix)
            inv_gates.append(Unitary(inv_matrix, *gate.target_qubits))
        inv_gates.append(None)
        return tuple(inv_gates)

    def construct_unitary(self):
        raise_error(ValueError, "Channels do not have unitary representation.")

    def state_vector_call(self, state):
        raise_error(ValueError, "`KrausChannel` cannot be applied to state "
                                "vectors. Please switch to density matrices.")

    def density_matrix_call(self, state):
        new_state = K.zeros_like(state)
        for gate, inv_gate in zip(self.gates, self.inverse_gates):
            new_state += gate(state)
            if inv_gate is not None:
                inv_gate(state)
        return new_state


class UnitaryChannel(KrausChannel, abstract_gates.UnitaryChannel):

    def __init__(self, p: List[float], ops: List["Gate"],
                 seed: Optional[int] = None):
        BackendGate.__init__(self)
        abstract_gates.UnitaryChannel.__init__(self, p, ops, seed=seed)
        self.set_seed()

    def calculate_inverse_gates(self):
        inv_gates = tuple(gate.dagger() for gate in self.gates[:-1])
        return inv_gates + (None,)

    def set_seed(self):
        if self.seed is not None:
            K.qnp.random.seed(self.seed)

    def state_vector_call(self, state):
        for p, gate in zip(self.probs, self.gates):
            if K.qnp.random.random() < p:
                state = gate(state)
        return state

    def density_matrix_call(self, state):
        new_state = (1 - self.psum) * state
        for p, gate, inv_gate in zip(self.probs, self.gates, self.inverse_gates):
            state = gate(state)
            new_state += p * state
            if inv_gate is not None:
                state = inv_gate(state) # reset to the original state vector
        return new_state


class PauliNoiseChannel(UnitaryChannel, abstract_gates.PauliNoiseChannel):

    def __init__(self, q: int, px: float = 0, py: float = 0, pz: float = 0,
                 seed: Optional[int] = None):
        BackendGate.__init__(self)
        abstract_gates.PauliNoiseChannel.__init__(self, q, px, py, pz, seed=seed)
        self.set_seed()

    def calculate_inverse_gates(self):
        return tuple(self.gates[:-1]) + (None,)


class ResetChannel(UnitaryChannel, abstract_gates.ResetChannel):

    def __init__(self, q: int, p0: float = 0.0, p1: float = 0.0,
                 seed: Optional[int] = None):
        BackendGate.__init__(self)
        abstract_gates.ResetChannel.__init__(self, q, p0=p0, p1=p1, seed=seed)
        self.set_seed()

    def calculate_inverse_gates(self):
        inv_gates = tuple(gate.dagger() if not isinstance(gate, abstract_gates.M) else None
                          for gate in self.gates[:-1])
        return inv_gates + (None,)

    def state_vector_call(self, state):
        not_collapsed = True
        if K.qnp.random.random() < self.probs[-2]:
            state = K.state_vector_collapse(self.gates[-2], state, [0])
            not_collapsed = False
        if K.qnp.random.random() < self.probs[-1]:
            if not_collapsed:
                state = K.state_vector_collapse(self.gates[-2], state, [0])
            state = self.gates[-1](state)
        return state

    def density_matrix_call(self, state):
        new_state = (1 - self.psum) * state
        for p, gate, inv_gate in zip(self.probs, self.gates, self.inverse_gates):
            if isinstance(gate, M):
                state = K.density_matrix_collapse(gate, state, [0])
            else:
                state = gate(state)
            new_state += p * state
            if inv_gate is not None:
                state = inv_gate(state) # reset to the original state vector
        return new_state


class ThermalRelaxationChannel(abstract_gates.ThermalRelaxationChannel):

    def __new__(cls, q, t1, t2, time, excited_population=0, seed=None):
        if t2 > t1:
            cls_s = _ThermalRelaxationChannelB
        else:
            cls_s = _ThermalRelaxationChannelA
        return cls_s(
            q, t1, t2, time, excited_population=excited_population, seed=seed)

    def calculate_probabilities(self, t1, t2, time, excited_population):
        cls = abstract_gates.ThermalRelaxationChannel
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


class _ThermalRelaxationChannelA(ResetChannel, abstract_gates._ThermalRelaxationChannelA):

    def calculate_probabilities(self, t1, t2, time, excited_population):
        return ThermalRelaxationChannel.calculate_probabilities(
            self, t1, t2, time, excited_population)

    def __init__(self, q, t1, t2, time, excited_population=0, seed=None):
        BackendGate.__init__(self)
        abstract_gates._ThermalRelaxationChannelA.__init__(
            self, q, t1, t2, time, excited_population=excited_population,
            seed=seed)
        self.set_seed()

    def state_vector_call(self, state):
        if K.qnp.random.random() < self.probs[0]:
            state = self.gates[0](state)
        return ResetChannel.state_vector_call(self, state)


class _ThermalRelaxationChannelB(MatrixGate, abstract_gates._ThermalRelaxationChannelB):

    def calculate_probabilities(self, t1, t2, time, excited_population):
        return ThermalRelaxationChannel.calculate_probabilities(
            self, t1, t2, time, excited_population)

    def __init__(self, q, t1, t2, time, excited_population=0, seed=None):
        BackendGate.__init__(self)
        abstract_gates._ThermalRelaxationChannelB.__init__(
            self, q, t1, t2, time, excited_population=excited_population,
            seed=seed)
        self._qubits_tensor = None
        if self.gate_op:
            self.gate_op = K.op.apply_two_qubit_gate

    @property
    def cache(self):
        if self._cache is None:
            cache = K.create_gate_cache(self)

            qubits = sorted(self.nqubits - q - 1 for q in self.target_qubits)
            cache.qubits_tensor = qubits + [q + self.nqubits for q in qubits]
            cache.target_qubits_dm = self.qubits + tuple(q + self.nqubits for q in self.qubits)

            if K.name != "custom":
                cache.calculation_cache = K.create_einsum_cache(
                    cache.target_qubits_dm, 2 * self.nqubits)

            self._cache = cache

        return self._cache

    def construct_unitary(self):
        matrix = K.qnp.diag([1 - self.preset1, self.exp_t2, self.exp_t2,
                             1 - self.preset0])
        matrix[0, -1] = self.preset1
        matrix[-1, 0] = self.preset0
        return K.cast(matrix)

    def state_vector_call(self, state):
        raise_error(ValueError, "Thermal relaxation cannot be applied to "
                                "state vectors when T1 < T2.")

    def density_matrix_call(self, state):
        if K.name == "custom":
            return self.gate_op(state, self.matrix, self.cache.qubits_tensor,
                                2 * self.nqubits, *self.cache.target_qubits_dm,
                                K.get_threads())
        return K.state_vector_call(self, state)
