import tensorflow as tf
from typing import List, Optional, Union


class Callback:
    """Base callback class.

    All Tensorflow callbacks should inherit this class and implement its
    `__call__` method.

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

    @property
    def results(self) -> Union[tf.Tensor, List[tf.Tensor]]:
        """Callback's results after a circuit execution.

        If the callback was used in a single circuit execution
            A single rank-1 `tf.Tensor` that holds all the results.
        If the callback was used in more than one circuit executions
            A list of rank-1 `tf.Tensor`s where each holds the results of each
            execution.
        """
        if len(self._results) > 1:
            return self._results
        elif len(self._results) == 1:
            return self._results[0]
        raise ValueError("Callback does not have results available before "
                         "using in a circuit execution.")

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
            final_state = c(callbacks=entropy)
            print(entropy.results.numpy())
            # Should print [0, 0, np.log(2)] which is the entanglement entropy
            # after every gate in the calculation.
    """

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
        if self.nqubits is None:
            self.nqubits = len(tuple(state.shape))

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
        return entropy
