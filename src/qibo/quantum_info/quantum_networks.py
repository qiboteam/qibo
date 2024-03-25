"""Module defining the `QuantumNetwork` class and adjacent functions."""

import re
from functools import reduce
from operator import mul
from typing import List, Optional, Tuple, Union

import numpy as np

from qibo.backends import _check_backend
from qibo.config import raise_error


class QuantumNetwork:
    """This class stores the Choi operator of the quantum network as a tensor,
    which is an unique representation of the quantum network.

    A minimum quantum network is a quantum channel, which is a quantum network of the form
    :math:`J[n \\to m]`, where :math:`n` is the dimension of the input system ,
    and :math:`m` is the dimension of the output system.
    A quantum state is a quantum network of the form :math:`J[1 \\to n]`,
    such that the input system is trivial.
    An observable is a quantum network of the form :math:`J[n \\to 1]`,
    such that the output system is trivial.

    A quantum network may contain multiple input and output systems.
    For example, a "quantum comb" is a quantum network of the form :math:`J[n', n \\to m, m']`,
    which convert a quantum channel of the form :math:`J[n \\to m]`
    to a quantum channel of the form :math:`J[n' \\to m']`.

    Args:
        matrix (ndarray): input Choi operator.
        partition (List[int] or Tuple[int]): partition of ``matrix``.
        system_output (List[bool] or Tuple[bool], optional): mask on the output system of the
            Choi operator. If ``None``, defaults to
            ``(False,True,False,True,...)``, where ``len(system_output)=len(partition)``.
            Defaults to ``None``.
        pure (bool, optional): ``True`` when ``matrix`` is a "pure" representation (e.g. a pure
            state, a unitary operator, etc.), ``False`` otherwise. Defaults to ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to be used in
            calculations. If ``None``, defaults to :class:`qibo.backends.GlobalBackend`.
            Defaults to ``None``.
    """

    def __init__(
        self,
        matrix,
        partition: Union[List[int], Tuple[int]],
        system_output: Optional[Union[List[bool], Tuple[bool]]] = None,
        pure: bool = False,
        backend=None,
    ):
        self._run_checks(partition, system_output, pure)

        self._matrix = matrix
        self.partition = partition
        self.system_output = system_output
        self._pure = pure
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
        if backend is None:  # pragma: no cover
            backend = self._backend

        return backend.cast(self._matrix, dtype=self._matrix.dtype)

    def is_pure(self):
        """Returns bool indicading if the Choi operator of the network is pure."""
        return self._pure

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

        if order is None and self._backend.__class__.__name__ == "TensorflowBackend":
            order = "euclidean"

        self._matrix = self._full()
        self._pure = False

        reshaped = self._backend.cast(
            np.reshape(self._matrix, (self.dims, self.dims)), dtype=self._matrix.dtype
        )
        reshaped = self._backend.cast(
            np.transpose(np.conj(reshaped)) - reshaped, dtype=reshaped.dtype
        )
        norm = self._backend.calculate_norm_density_matrix(reshaped, order=order)

        return float(norm) <= precision_tol

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

        if order is None and self._backend.__class__.__name__ == "TensorflowBackend":
            order = "euclidean"

        self._matrix = self._full()
        self._pure = False

        partial_trace = self._einsum("jkjl -> kl", self._matrix)
        identity = self._backend.cast(
            np.eye(partial_trace.shape[0]), dtype=partial_trace.dtype
        )

        norm = self._backend.calculate_norm_density_matrix(
            partial_trace - identity,
            order=order,
        )

        return float(norm) <= precision_tol

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

        if order is None and self._backend.__class__.__name__ == "TensorflowBackend":
            order = "euclidean"

        self._matrix = self._full()
        self._pure = False

        partial_trace = self._einsum("jklk -> jl", self._matrix)
        identity = self._backend.cast(
            np.eye(partial_trace.shape[0]), dtype=partial_trace.dtype
        )

        norm = self._backend.calculate_norm_density_matrix(
            partial_trace - identity,
            order=order,
        )

        return float(norm) <= precision_tol

    def is_positive_semidefinite(self, precision_tol: float = 1e-8):
        """Returns bool indicating if Choi operator :math:`\\mathcal{E}` of the network is positive-semidefinite.

        Args:
            precision_tol (float, optional): threshold value used to check if eigenvalues of
                the Choi operator :math:`\\mathcal{E}` are such that
                :math:`\\textup{eigenvalues}(\\mathcal{E}) >= - \\textup{precision_tol}`.
                Note that this parameter can be set to negative values.
                Defaults to :math:`0.0`.

        Returns:
            bool: Positive-semidefinite condition.
        """
        self._matrix = self._full()
        self._pure = False

        reshaped = np.reshape(self._matrix, (self.dims, self.dims))

        if self.is_hermitian():
            eigenvalues = np.linalg.eigvalsh(reshaped)
        else:
            if self._backend.__class__.__name__ in [
                "CupyBackend",
                "CuQuantumBackend",
            ]:  # pragma: no cover
                reshaped = np.array(reshaped.tolist(), dtype=reshaped.dtype)
            eigenvalues = np.linalg.eigvals(reshaped)

        return all(eigenvalue >= -precision_tol for eigenvalue in eigenvalues)

    def is_channel(
        self,
        order: Optional[Union[int, str]] = None,
        precision_tol_causal: float = 1e-8,
        precision_tol_psd: float = 1e-8,
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
        matrix = self._backend.cast(self._matrix, copy=True)

        if self.is_pure():
            return self._einsum("kj,ml,jl -> km", matrix, np.conj(matrix), state)

        return self._einsum("jklm,km -> jl", matrix, state)

    def link_product(self, second_network, subscripts: str = "ij,jk -> ik"):
        """Link product between two quantum networks.

        The link product is not commutative. Here, we assume that
        :math:`A.\\textup{link_product}(B)` means "applying :math:`B` to :math:`A`".
        However, the ``link_product`` is associative, so we override the `@` operation
        in order to simplify notation.

        Args:
            second_network (:class:`qibo.quantum_info.quantum_networks.QuantumNetwork`): Quantum
                network to be applied to the original network.
            subscripts (str, optional): Specifies the subscript for summation using
                the Einstein summation convention. For more details, please refer to
                `numpy.einsum <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>`_.

        Returns:
            :class:`qibo.quantum_info.quantum_networks.QuantumNetwork`: Quantum network resulting
                from the link product between two quantum networks.
        """
        if not isinstance(second_network, QuantumNetwork):
            raise_error(
                TypeError,
                "It is not possible to implement link product of a "
                + "``QuantumNetwork`` with a non-``QuantumNetwork``.",
            )

        if not isinstance(subscripts, str):
            raise_error(
                TypeError,
                f"subscripts must be type str, but it is type {type(subscripts)}.",
            )

        subscripts = subscripts.replace(" ", "")
        pattern_two, pattern_four = self._check_subscript_pattern(subscripts)
        channel_subscripts = pattern_two and subscripts[1] == subscripts[3]
        inv_subscripts = pattern_two and subscripts[0] == subscripts[4]
        super_subscripts = (
            pattern_four
            and subscripts[1] == subscripts[5]
            and subscripts[2] == subscripts[6]
        )

        if not channel_subscripts and not inv_subscripts and not super_subscripts:
            raise_error(
                NotImplementedError,
                "Subscripts do not match any implemented pattern.",
            )

        first_matrix = self._full()
        second_matrix = second_network._full()  # pylint: disable=W0212

        if super_subscripts:
            cexpr = "jklmnopq,klop->jmnq"
            return QuantumNetwork(
                self._einsum(cexpr, first_matrix, second_matrix),
                [self.partition[0] + self.partition[-1]],
            )

        cexpr = "jkab,klbc->jlac"

        if inv_subscripts:
            return QuantumNetwork(
                self._einsum(cexpr, second_matrix, first_matrix),
                [second_network.partition[0], self.partition[1]],
            )

        return QuantumNetwork(
            self._einsum(cexpr, first_matrix, second_matrix),
            [self.partition[0], second_network.partition[1]],
        )

    def copy(self):
        """Returns a copy of the :class:`qibo.quantum_info.quantum_networks.QuantumNetwork` object."""
        return self.__class__(
            np.copy(self._matrix),
            partition=self.partition,
            system_output=self.system_output,
            pure=self._pure,
            backend=self._backend,
        )

    def to_full(self, backend=None):
        """Convert the internal representation to the full Choi operator of the network.

        Returns:
            (:class:`qibo.quantum_info.quantum_networks.QuantumNetwork`): The full representation
            of the Quantum network.
        """
        if backend is None:  # pragma: no cover
            backend = self._backend

        if self.is_pure():
            self._matrix = self._full()
            self._pure = False

        return self.matrix(backend)

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

        if self._full().shape != second_network._full().shape:
            raise_error(
                ValueError,
                f"The Choi operators must have the same shape, but {self._matrix.shape} != "
                + f"{second_network.matrix(second_network._backend).shape}.",
            )

        if self.system_output != second_network.system_output:
            raise_error(ValueError, "The networks must have the same output system.")

        new_first_matrix = self._full()
        new_second_matrix = second_network._full()

        return QuantumNetwork(
            new_first_matrix + new_second_matrix,
            self.partition,
            self.system_output,
            pure=False,
            backend=self._backend,
        )

    def __mul__(self, number: Union[float, int]):
        """Returns quantum network with its Choi operator multiplied by a scalar.

        If the quantum network is pure and ``number > 0.0``, the method returns a pure quantum
        network with its Choi operator multiplied by the square root of ``number``.
        This is equivalent to multiplying `self.to_full()` by the ``number``.
        Otherwise, this method will return a full quantum network.

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

        if self.is_pure() and number > 0.0:
            return QuantumNetwork(
                np.sqrt(number) * self.matrix(backend=self._backend),
                partition=self.partition,
                system_output=self.system_output,
                pure=True,
                backend=self._backend,
            )

        matrix = self._full()

        return QuantumNetwork(
            number * matrix,
            partition=self.partition,
            system_output=self.system_output,
            pure=False,
            backend=self._backend,
        )

    def __rmul__(self, number: Union[float, int]):
        """"""
        return self.__mul__(number)

    def __truediv__(self, number: Union[float, int]):
        """Returns quantum network with its Choi operator divided by a scalar.

        If the quantum network is pure and ``number > 0.0``, the method returns a pure quantum
        network with its Choi operator divided by the square root of ``number``.
        This is equivalent to dividing `self.to_full()` by the ``number``.
        Otherwise, this method will return a full quantum network.

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

        number = np.sqrt(number) if self.is_pure() and number > 0.0 else number

        return QuantumNetwork(
            self.matrix(backend=self._backend) / number,
            partition=self.partition,
            system_output=self.system_output,
            pure=self.is_pure(),
            backend=self._backend,
        )

    def __matmul__(self, second_network):
        """Defines matrix multiplication between two ``QuantumNetwork`` objects.

        If ``self.partition == second_network.partition in [2, 4]``, this method is overwritten by
        :meth:`qibo.quantum_info.quantum_networks.QuantumNetwork.link_product`.

        Args:
            second_network (:class:`qibo.quantum_info.quantum_networks.QuantumNetwork`):

        Returns:
            :class:`qibo.quantum_info.quantum_networks.QuantumNetwork`: Quantum network resulting
                from the link product operation.
        """
        if not isinstance(second_network, QuantumNetwork):
            raise_error(
                TypeError,
                "It is not possible to implement matrix multiplication of a "
                + "``QuantumNetwork`` by a non-``QuantumNetwork``.",
            )

        if len(self.partition) == 2:  # `self` is a channel
            if len(second_network.partition) != 2:
                raise_error(
                    ValueError,
                    f"`QuantumNetwork {second_network} is assumed to be a channel, but it is not. "
                    + "Use `link_product` method to specify the subscript.",
                )
            if self.partition[1] != second_network.partition[0]:
                raise_error(
                    ValueError,
                    "partitions of the networks do not match: "
                    + f"{self.partition[1]} != {second_network.partition[0]}.",
                )

            subscripts = "jk,kl -> jl"

        elif len(self.partition) == 4:  # `self` is a super-channel
            if len(second_network.partition) != 2:
                raise_error(
                    ValueError,
                    f"`QuantumNetwork {second_network} is assumed to be a channel, but it is not. "
                    + "Use `link_product` method to specify the subscript.",
                )
            if self.partition[1] != second_network.partition[0]:
                raise_error(
                    ValueError,
                    "Systems of the channel do not match the super-channel: "
                    + f"{self.partition[1], self.partition[2]} != "
                    + f"{second_network.partition[0],second_network.partition[1]}.",
                )

            subscripts = "jklm,kl -> jm"
        else:
            raise_error(
                NotImplementedError,
                "`partitions` do not match any implemented pattern``. "
                + "Use `link_product` method to specify the subscript.",
            )

        return self.link_product(second_network, subscripts=subscripts)

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

    def _run_checks(self, partition, system_output, pure):
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

        if not isinstance(pure, bool):
            raise_error(
                TypeError,
                f"``pure`` must be type ``bool``, but it is type ``{type(pure)}``.",
            )

    def _set_tensor_and_parameters(self):
        """Sets tensor based on inputs."""
        self._backend = _check_backend(self._backend)

        self._einsum = self._backend.np.einsum

        if isinstance(self.partition, list):
            self.partition = tuple(self.partition)

        try:
            if self._pure:
                self._matrix = np.reshape(self._matrix, self.partition)
            else:
                matrix_partition = self.partition * 2
                self._matrix = np.reshape(self._matrix, matrix_partition)
        except:
            raise_error(
                ValueError,
                "``partition`` does not match the shape of the input matrix. "
                + f"Cannot reshape matrix of size {self._matrix.shape} to partition {self.partition}",
            )

        if self.system_output is None:
            self.system_output = [
                True,
            ] * len(self.partition)
            for k in range(len(self.partition) // 2):
                self.system_output[k * 2] = False
            self.system_output = tuple(self.system_output)
        else:
            self.system_output = tuple(self.system_output)

    def _full(self):
        """Reshapes input matrix based on purity."""
        matrix = self._backend.cast(self._matrix, copy=True)

        if self.is_pure():
            matrix = self._einsum("jk,lm -> kjml", matrix, np.conj(matrix))

        return matrix

    def _check_subscript_pattern(self, subscripts: str):
        """Checks if input subscript match any implemented pattern."""
        braket = "[a-z]"
        pattern_two = re.compile(braket * 2 + "," + braket * 2 + "->" + braket * 2)
        pattern_four = re.compile(braket * 4 + "," + braket * 2 + "->" + braket * 2)

        return bool(re.match(pattern_two, subscripts)), bool(
            re.match(pattern_four, subscripts)
        )
