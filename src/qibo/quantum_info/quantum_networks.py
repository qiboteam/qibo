"""Module defining the QuantumNetwork class and adjacent functions."""
import warnings
from functools import reduce
from operator import mul
from re import match
from typing import List, Optional, Tuple, Union

import numpy as np

from qibo.backends import GlobalBackend
from qibo.config import raise_error


class QuantumNetwork:
    def __init__(
        self,
        matrix,
        partition: Union[List[int], Tuple[int]],
        system_output: Optional[Tuple[bool]] = None,
        is_pure: bool = False,
        backend=None,
    ):
        self._run_checks(partition, system_output, is_pure)

        self._matrix = matrix
        self.partition = partition
        self.system_output = system_output
        self._is_pure = is_pure
        self._backend = backend
        self.dims = reduce(mul, self.partition)

        self._set_tensor_and_parameters()

    def matrix(self, backend=None):
        """Returns the Choi operator of the quantum network in matrix form.

        Args:
            backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to be used
                to return the Choi operator. If ``None``, defaults to the backend defined
                when initializing the :class:`qibo.quantum_info.quantum_networks.QuantumNetwork`
                object. Defaults to ``None``.

        Returns:
            ndarray: Choi matrix of the quantum network.
        """
        if backend is None:
            backend = self._backend

        return backend.cast(self._matrix, dtype=self._matrix.dtype)

    def is_pure(self):
        """Returns bool indicading if the Choi operator of the network is pure."""
        return self._is_pure

    def is_hermitian(
        self, order: Optional[Union[int, str]] = None, precision_tol: float = 1e-8
    ):
        """Returns bool indicating if the Choi operator :math:`\\mathcal{E}` of the network is Hermitian.

        Hermicity is calculated as distance between :math:`\\mathcal{E}` and
        :math:`\\mathcal{E}^{\\dagger}` with respect to a given norm.
        Default is the ``Hilbert-Schmidt`` norm (also known as ``Frobenius`` norm).

        For specifications on the other  possible values of the
        parameter ``order`` for the ``tensorflow`` backend, please refer to
        `tensorflow.norm <https://www.tensorflow.org/api_docs/python/tf/norm>`_.
        For all other backends, please refer to
        `numpy.linalg.norm <https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html>`_.

        Args:
            order (str or int, optional): order of the norm. Defaults to ``None``.
            precision_tol (float, optional): threshold :math:`\\epsilon` that defines if
                Choi operator of the network is :math:`\\epsilon`-close to Hermicity in
                the norm given by ``order``. Defaults to :math:`10^{-8}`.

        Returns:
            bool: Hermiticity condition.
        """
        if precision_tol < 0.0:
            raise_error(
                ValueError,
                f"``precision_tol`` must be non-negative float, but it is {precision_tol}",
            )

        reshaped = np.reshape(self._matrix, (self.dims, self.dims))
        return bool(
            self._backend.calculate_norm_density_matrix(
                np.transpose(np.conj(reshaped)) - reshaped, order=order
            )
            <= precision_tol
        )

    def is_unital(
        self, order: Optional[Union[int, str]] = None, precision_tol: float = 1e-8
    ):
        """Returns bool indicating if the Choi operator :math:`\\mathcal{E}` of the network is unital.

        Unitality is calculated as distance between the partial trace of :math:`\\mathcal{E}`
        and the Identity operator :math:`I`, with respect to a given norm.
        Default is the ``Hilbert-Schmidt`` norm (also known as ``Frobenius`` norm).

        For specifications on the other  possible values of the
        parameter ``order`` for the ``tensorflow`` backend, please refer to
        `tensorflow.norm <https://www.tensorflow.org/api_docs/python/tf/norm>`_.
        For all other backends, please refer to
        `numpy.linalg.norm <https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html>`_.

        Args:
            order (str or int, optional): order of the norm. Defaults to ``None``.
            precision_tol (float, optional): threshold :math:`\\epsilon` that defines
                if Choi operator of the network is :math:`\\epsilon`-close to unitality
                in the norm given by ``order``. Defaults to :math:`10^{-8}`.

        Returns:
            bool: Unitality condition.
        """
        if precision_tol < 0.0:
            raise_error(
                ValueError,
                f"``precision_tol`` must be non-negative float, but it is {precision_tol}",
            )

        self._matrix = self._full()

        partial_trace = np.einsum("jkjl -> kl", self._matrix)
        identity = self._backend.cast(
            np.eye(partial_trace.shape[0]), dtype=partial_trace.dtype
        )

        return bool(
            self._backend.calculate_norm_density_matrix(
                partial_trace - identity,
                order=order,
            )
            <= precision_tol
        )

    def is_causal(
        self, order: Optional[Union[int, str]] = None, precision_tol: float = 1e-8
    ):
        """Returns bool indicating if the Choi operator :math:`\\mathcal{E}` of the network satisfies the causal order condition.

        Causality is calculated as distance between partial trace of :math:`\\mathcal{E}`
        and the Identity operator :math:`I`, with respect to a given norm.
        Default is the ``Hilbert-Schmidt`` norm (also known as ``Frobenius`` norm).

        For specifications on the other  possible values of the
        parameter ``order`` for the ``tensorflow`` backend, please refer to
        `tensorflow.norm <https://www.tensorflow.org/api_docs/python/tf/norm>`_.
        For all other backends, please refer to
        `numpy.linalg.norm <https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html>`_.

        Args:
            order (str or int, optional): order of the norm. Defaults to ``None``.
            precision_tol (float, optional): threshold :math:`\\epsilon` that defines
                if Choi operator of the network is :math:`\\epsilon`-close to causality
                in the norm given by ``order``. Defaults to :math:`10^{-8}`.

        Returns:
            bool: Causal order condition.
        """
        if precision_tol < 0.0:
            raise_error(
                ValueError,
                f"``precision_tol`` must be non-negative float, but it is {precision_tol}",
            )

        self._matrix = self._full()

        partial_trace = np.einsum("jklk -> jl", self._matrix)
        identity = self._backend.cast(
            np.eye(partial_trace.shape[0]), dtype=partial_trace.dtype
        )

        return bool(
            self._backend.calculate_norm_density_matrix(
                partial_trace - identity,
                order=order,
            )
            <= precision_tol
        )

    def is_positive_semidefinite(self, precision_tol: float = 0.0):
        """Returns bool indicating if Choi operator :math:`\\mathcal{E}` of the networn is positive-semidefinite.

        Args:
            precision_tol (float, optional): threshold value used to check if eigenvalues of
                the Choi operator :math:`\\mathcal{E}` are such that
                :math:`\\textup{eigenvalues}(\\mathcal{E}) >= \\textup{precision_tol}`.
                Note that this parameter can be set to negative values.
                Defaults to :math:`0.0`.

        Returns:
            bool: Positive-semidefinite condition.
        """
        reshaped = np.reshape(self._matrix, (self.dims, self.dims))
        if self.is_hermitian():
            eigenvalues = np.linalg.eigvalsh(reshaped)
        else:
            if self._backend.__class__.__name__ in ["CupyBackend", "CuQuantumBackend"]:
                reshaped = np.array(reshaped.tolist(), dtype=reshaped.dtype)
            eigenvalues = np.linalg.eigvals(reshaped)

        return all(eigenvalue >= precision_tol for eigenvalue in eigenvalues)

    def is_channel(
        self,
        order: Optional[Union[int, str]] = None,
        precision_tol_causal: float = 1e-8,
        precision_tol_psd: float = 0.0,
    ):
        """Returns bool indicating if Choi operator :math:`\\mathcal{E}` is a channel.

        Args:
            order (int or str, optional): order of the norm used to calculate causality.
                Defaults to ``None``.
            precision_tol_causal (float, optional): threshold :math:`\\epsilon` that defines if
                Choi operator of the network is :math:`\\epsilon`-close to causality in the norm
                given by ``order``. Defaults to :math:`10^{-8}`.
            precision_tol_psd (float, optional): threshold value used to check if eigenvalues of
                the Choi operator :math:`\\mathcal{E}` are such that
                :math:`\\textup{eigenvalues}(\\mathcal{E}) >= \\textup{precision_tol_psd}`.
                Note that this parameter can be set to negative values.
                Defaults to :math:`0.0`.

        Returns:
            bool: Channel condition.
        """
        return self.is_causal(
            order, precision_tol_causal
        ) and self.is_positive_semidefinite(precision_tol_psd)

    def apply(self, state):
        """Apply the Choi operator :math:`\\mathcal{E}` to ``state`` :math:`\\varrho`.

        It is assumed that ``state`` :math:`\\varrho` is a density matrix.

        Args:
            state (ndarray): density matrix of a ``state``.

        Returns:
            ndarray: Resulting state :math:`\\mathcal{E}(\\varrho)`.
        """
        if self.is_pure():
            return np.einsum(
                "jk,lm,jl -> km", self._matrix, np.conj(self._matrix), state
            )

        return np.einsum("jklm,kl -> jl", self._matrix, state)

    def link_product(self, second_network, subscripts: Optional[str] = None):
        if not isinstance(second_network, QuantumNetwork):
            raise_error(
                TypeError,
                "It is not possible to implement link product of a "
                + "``QuantumNetwork`` with a non-``QuantumNetwork``.",
            )

        channel_subscripts = match(r"i\s*j\s*,\s*j\s*k\s* -> \s*i\s*k", subscripts)
        inv_subscripts = match(r"i\s*j\s*,\s*k\s*i\s* -> \s*k\s*j", subscripts)

        first_matrix = self._full()
        second_matrix = second_network._full()  # pylint: disable=W0212

        if subscripts is None or channel_subscripts is not None:
            cexpr = "ijab,jkbc->ikac"
            return QuantumNetwork(
                np.einsum(cexpr, first_matrix, second_matrix),
                [self.partition[0], second_network.partition[1]],
            )

        if inv_subscripts is not None:
            cexpr = "ijab,jkbc->ikac"
            return QuantumNetwork(
                np.einsum(cexpr, first_matrix, second_matrix),
                [second_network.partition[0], self.partition[1]],
            )

        raise_error(NotImplementedError, "Not implemented.")

    def __add__(self, second_network):
        """Add two Quantum Networks by adding their Choi operators.

        This operation always returns a non-pure Quantum Network.

        Args:
            second_network (:class:`qibo.quantum_info.quantum_networks.QuantumNetwork`): Quantum
                network to be added to the original network.

        Returns:
            (:class:`qibo.quantum_info.quantum_networks.QuantumNetwork`): Quantum network resulting
                from the summation of two Choi operators.
        """
        if not isinstance(second_network, QuantumNetwork):
            raise_error(
                TypeError,
                "It is not possible to add a object of type ``QuantumNetwork`` "
                + f"and and object of type ``{type(second_network)}``.",
            )

        if self._matrix.shape != second_network.matrix().shape:
            raise_error(
                ValueError,
                "The Choi operators must have the same shape, "
                + f"but {self._matrix.shape} != {second_network.shape}.",
            )

        if self.system_output != second_network.system_output:
            raise_error(ValueError, "The networks must have the same output system.")

        first_matrix = self._full()
        second_matrix = second_network._full()

        return QuantumNetwork(
            first_matrix + second_matrix,
            self.partition,
            self.system_output,
            is_pure=False,
            backend=self._backend,
        )

    def __mul__(self, number: Union[float, int]):
        """Returns quantum network with its Choi operator multiplied by a scalar.

        Args:
            number (float or int): scalar to multiply the Choi operator of the network with.

        Returns:
            :class:`qibo.quantum_info.quantum_networks.QuantumNetwork`: Quantum network with its
                Choi operator multiplied by ``number``.
        """
        if not isinstance(number, (float, int)):
            raise_error(
                TypeError,
                "It is not possible to multiply a ``QuantumNetwork`` by a non-scalar.",
            )

        matrix = self._full()

        return QuantumNetwork(number * matrix, self.partition, self.system_output)

    def __truediv__(self, number: Union[float, int]):
        """Returns quantum network with its Choi operator divided by a scalar.

        Args:
            number (float or int): scalar to divide the Choi operator of the network with.

        Returns:
            :class:`qibo.quantum_info.quantum_networks.QuantumNetwork`: Quantum network with its
                Choi operator divided by ``number``.
        """
        if not isinstance(number, (float, int)):
            raise_error(
                TypeError,
                "It is not possible to divide a ``QuantumNetwork`` by a non-scalar.",
            )

        matrix = self._full()

        return QuantumNetwork(matrix / number, self.partition, self.system_output)

    def __matmul__(self, second_network):
        """Defines matrix multiplication between two ``QuantumNetwork`` objects.

        If ``len(self.partition) == 2`` and ``len(second_network.partition) == 2``,
        this method is overwritten by
        :meth:`qibo.quantum_info.quantum_networks.QuantumNetwork.link_product`.

        Args:
            second_network (:class:`qibo.quantum_info.quantum_networks.QuantumNetwork`):

        Returns:
            :class:`qibo.quantum_info.quantum_networks.QuantumNetwork`: Quantum network resulting
                from the link
        """
        if not isinstance(second_network, QuantumNetwork):
            raise_error(
                TypeError,
                "It is not possible to implement matrix multiplication of a "
                + "``QuantumNetwork`` by a non-``QuantumNetwork``.",
            )

        if len(self.partition) == 2 and len(second_network.partition) == 2:
            return self.link_product(second_network)

    def __str__(self):
        """Method to define how to print relevant information of the quantum network."""
        string_in = ", ".join(
            [
                str(self.partition[k])
                for k in range(len(self.partition))
                if not self.system_output[k]
            ]
        )

        string_out = ", ".join(
            [
                str(self.partition[k])
                for k in range(len(self.partition))
                if self.system_output[k]
            ]
        )

        return f"J[{string_in} -> {string_out}]"

    def _run_checks(self, partition, system_output, is_pure):
        """Checks if all inputs are correct in type and value."""
        if not isinstance(partition, (list, tuple)):
            raise_error(
                TypeError,
                "``partition`` must be type ``tuple`` or ``list``, "
                + f"but it is type ``{type(partition)}``.",
            )

        if any(not isinstance(party, int) for party in partition):
            raise_error(
                ValueError,
                "``partition`` must be a ``tuple`` or ``list`` of positive integers, "
                + "but contains non-integers.",
            )

        if any(party <= 0 for party in partition):
            raise_error(
                ValueError,
                "``partition`` must be a ``tuple`` or ``list`` of positive integers, "
                + "but contains non-positive integers.",
            )

        if system_output is not None and len(system_output) != len(partition):
            raise_error(
                ValueError,
                "``len(system_output)`` must be the same as ``len(partition)``, "
                + f"but {len(system_output)} != {len(partition)}.",
            )

        if not isinstance(is_pure, bool):
            raise_error(
                TypeError,
                f"``is_pure`` must be type ``bool``, but it is type ``{type(is_pure)}``.",
            )

    def _set_tensor_and_parameters(self):
        """Sets tensor based on inputs."""
        if self._backend is None:
            self._backend = GlobalBackend()

        if not self.is_hermitian():
            warnings.warn("Input matrix is not Hermitian.")

        if isinstance(self.partition, list):
            self.partition = tuple(self.partition)

        if self._is_pure:
            self._matrix = np.reshape(self._matrix, self.partition)
        else:
            matrix_partition = self.partition * 2
            self._matrix = np.reshape(self._matrix, matrix_partition)

        if self.system_output is None:
            self.system_output = (False,) if len(self.partition) == 1 else (False, True)
        else:
            self.system_output = tuple(self.system_output)

    def _full(self):
        if self.is_pure():
            matrix = np.einsum("jk,lm -> kjml", self._matrix, np.conj(self._matrix))
            self._is_pure = False

            return matrix

        return self._matrix
