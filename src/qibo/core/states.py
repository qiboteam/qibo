import math
from qibo import K
from qibo.abstractions import states
from qibo.config import raise_error


class State(states.AbstractState):

    @states.AbstractState.vector.setter
    def vector(self, x):
        if not isinstance(x, K.tensor_types):
            raise_error(TypeError, "Initial state type {} is not recognized."
                                    "".format(type(x)))
        states.AbstractState.vector.fset(self, x) # pylint: disable=no-member
        self._vector = K.cast(self._vector)

    @states.AbstractState.matrix.setter
    def matrix(self, x):
        if not isinstance(x, K.tensor_types):
            raise_error(TypeError, "Initial state type {} is not recognized."
                                    "".format(type(x)))
        states.AbstractState.matrix.fset(self, x) # pylint: disable=no-member
        self._matrix = K.cast(self._matrix)

    def to_matrix(self):
        self.matrix = K.outer(self.vector, K.conj(self.vector))
        return self

    @classmethod
    def from_vector(cls, x, nqubits=None):
        state = cls()
        state.vector = x
        return state

    @classmethod
    def from_matrix(cls, x, nqubits=None):
        state = cls()
        state.matrix = x
        return state

    @classmethod
    def default(cls, nqubits, is_matrix=False):
        state = cls(nqubits)
        if is_matrix:
            state.matrix = K.initial_state(nqubits, is_matrix)
        else:
            state.vector = K.initial_state(nqubits, is_matrix)
        return state

    @classmethod
    def ones(cls, nqubits, is_matrix=False):
        state = cls(nqubits)
        shape = K.cast(self.nstates, dtype='DTYPEINT')
        state.vector = K.ones(shape) / K.cast(K.sqrt(shape))
        if is_matrix:
            state.to_matrix()
        return state

    @classmethod
    def random(cls, nqubits, is_matrix=False):
        raise_error(NotImplementedError)

    @property
    def shape(self):
        return tuple(self.tensor.shape)

    def __array__(self):
        return K.qnp.cast(self.tensor)

    def numpy(self):
        return self.__array__()
