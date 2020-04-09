import numpy as np
import tensorflow as tf
from qibo.config import DTYPE, DTYPECPX
from typing import List, Optional, Union


class Callback:
    """Base callback class.

    All Tensorflow callbacks should inherit this class and implement its
    `__call__` method.

    Results of a callback can be accessed by indexing the corresponding object.

    Args:
        steps (int): Every how many gates to perform the callback calculation.
            Defaults at 1 for which the calculation is done after every gate.
    """

    def __init__(self, steps: int = 1):
        self.steps = steps
        self._results = []
        self._nqubits = None

    @property
    def nqubits(self) -> int:
        """Total number of qubits in the circuit that the callback was added in."""
        return self._nqubits

    @nqubits.setter
    def nqubits(self, n: int):
        self._nqubits = n

    def __getitem__(self, k) -> tf.Tensor:
        if isinstance(k, int):
            if k > len(self._results):
                raise IndexError("Attempting to access callbacks {} run but "
                                 "the callback has been used in {} executions."
                                 "".format(k, len(self._results)))
            return self._results[k]
        if isinstance(k, slice) or isinstance(k, list) or isisntance(k, tuple):
            return tf.stack(self._results[k])
        raise IndexError("Unrecognized type for index {}.".format(k))

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def append(self, result: tf.Tensor):
        self._results.append(result)


class EntanglementEntropy(Callback):
    """Von Neumann entanglement entropy callback.

    Args:
        partition (list): List with qubit ids that defines the first subsystem
            for the entropy calculation.
            If `partition` is not given then the first subsystem is the first
            half of the qubits.
        steps (int): Every how many gates to perform the entropy calculation.
            Defaults at 1 for which the calculation is done after every gate.

    Example:
        ::

            from qibo import models, gates, callbacks
            # create entropy callback where qubit 0 is the first subsystem
            entropy = callbacks.EntanglementEntropy([0])
            # initialize circuit with 2 qubits and add gates
            c = models.Circuit(2)
            c.add(gates.H(0))
            c.add(gates.CNOT(0, 1))
            # execute the circuit using the callback
            final_state = c(callback=entropy)
            print(entropy[0])
            # Should print [0, 0, np.log(2)] which is the entanglement entropy
            # after every gate in the calculation.
    """
    _log2 = tf.cast(tf.math.log(2.0), dtype=DTYPE)

    def __init__(self, partition: Optional[List[int]] = None, steps: int = 1):
        super(EntanglementEntropy, self).__init__(steps)
        self.partition = partition
        self.rho_dim = None

    @property
    def nqubits(self) -> int:
        return self._nqubits

    @nqubits.setter
    def nqubits(self, n: int):
        self._nqubits = n
        if self.partition is None:
            self.partition = list(range(n // 2 + n % 2))

        if len(self.partition) < n // 2:
            # Revert parition so that we diagonalize a smaller matrix
            self.partition = [i for i in range(n)
                              if i not in set(self.partition)]
        self.rho_dim = 2 ** (n - len(self.partition))

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        # Cast state in the proper state if it is given as a numpy array
        shape = tuple(state.shape)
        if len(shape) == 1:
            if not isinstance(state, np.ndarray):
                raise TypeError("If the state is passed as a vector then it should "
                                "be a numpy array but is {}.".format(type(state)))
            self.nqubits = int(np.log2(shape[0]))
        if self.nqubits is None:
            self.nqubits = len(shape)
        if isinstance(state, np.ndarray):
            state = tf.convert_to_tensor(state.reshape(self.nqubits * (2,)),
                                         dtype=DTYPECPX)

        # Construct density matrix
        rho = tf.tensordot(state, tf.math.conj(state),
                           axes=[self.partition, self.partition])
        rho = tf.reshape(rho, (self.rho_dim, self.rho_dim))
        # Diagonalize
        eigvals = tf.linalg.eigvalsh(rho)
        eigvals2 = tf.square(tf.abs(eigvals))
        # Calculate entropy (treating zero eigenvalues)
        regularizer = tf.where(eigvals2 == 0,
                               tf.ones_like(eigvals2),
                               tf.zeros_like(eigvals2))
        entropy = - tf.reduce_sum(eigvals2 * tf.math.log(eigvals2 + regularizer))
        return entropy / self._log2
