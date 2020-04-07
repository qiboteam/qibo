import tensorflow as tf
from typing import List, Union


class Callback:
    """Base callback class.

    All Tensorflow callbacks should inherit this class.

    Args:
        steps: Every how many gates to perform the callback calculation.
            Defaults at 1 for which calculation is done after every gate.
    """

    def __init__(self, steps: int = 1):
        self.steps = steps
        self._results = []
        self._nqubits = None

    @property
    def nqubits(self) -> int:
        return self._nqubits

    @nqubits.setter
    def nqubits(self, n: int):
        self._nqubits = n

    @property
    def results(self) -> Union[tf.Tensor, List[tf.Tensor]]:
        if len(self._results) > 1:
            return self._results
        return self._results[0]

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def append(self, result: tf.Tensor):
        self._results.append(result)


class EntanglementEntropy(Callback):

    def __init__(self, partition: List[int], steps: int = 1):
        super(EntanglementEntropy, self).__init__(steps)
        self.partition = partition
        self.rho_dim = None

    @property
    def nqubits(self) -> int:
        return self._nqubits

    @nqubits.setter
    def nqubits(self, n: int):
        self._nqubits = n
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
