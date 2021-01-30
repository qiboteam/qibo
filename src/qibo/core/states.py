import math
from qibo import K
from qibo.config import raise_error
from qibo.abstractions.states import AbstractState


class VectorState(AbstractState):

    @property
    def shape(self):
        return (self.nstates,)

    @AbstractState.tensor.setter
    def tensor(self, x):
        if not isinstance(x, K.tensor_types):
            raise_error(TypeError, "Initial state type {} is not recognized."
                                    "".format(type(x)))
        AbstractState.tensor.fset(self, x) # pylint: disable=no-member
        if x.shape != self.shape:
            raise_error(ValueError, "Invalid tensor shape {} for state of {} "
                                    "qubits.".format(x.shape, self.nqubits))
        self._tensor = K.cast(x)

    def __array__(self):
        return K.qnp.cast(self.tensor)

    def numpy(self):
        return self.__array__()

    @classmethod
    def zstate(cls, nqubits):
        state = cls(nqubits)
        is_matrix = isinstance(state, MatrixState)
        state.tensor = K.initial_state(nqubits, is_matrix)
        return state

    @classmethod
    def xstate(cls, nqubits):
        state = cls(nqubits)
        shape = K.cast(self.nstates, dtype='DTYPEINT')
        state.tensor = K.ones(shape) / K.cast(K.sqrt(shape))
        return state

    @classmethod
    def random(cls, nqubits):
        raise_error(NotImplementedError)

    def to_density_matrix(self):
        matrix = K.outer(self.tensor, K.conj(self.tensor))
        return MatrixState.from_tensor(matrix, nqubits=self.nqubits)


class MatrixState(VectorState):

    @property
    def shape(self):
        return (self.nstates, self.nstates)

    @classmethod
    def xstate(cls, nqubits):
        state = VectorState.xstate(nqubits)
        return state.to_density_matrix()

    @classmethod
    def random(cls, nqubits):
        raise_error(NotImplementedError)

    def to_density_matrix(self):
        raise_error(RuntimeError, "State is already a density matrix.")
