"""Module defining the QuantumNetwork class and adjacent functions."""

from functools import reduce
from operator import mul
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
        self._run_checks(matrix, partition, system_output, is_pure)

        self.matrix = matrix
        self.partition = partition
        self.system_output = system_output
        self._is_pure = is_pure
        self._backend = backend
        self.np = None
        self.dims = reduce(mul, self.partition)

        self._set_tensor_and_parameters()

        # if not self.is_hermitian:
        #     raise Warning("The input matrix is not Hermitian.")

    def _run_checks(self, matrix, partition, system_output, is_pure):
        """Checks if all inputs are correct in type and value."""
        if not isinstance(partition, (list, tuple)):
            raise_error(
                TypeError,
                "``partition`` must be type ``tuple`` or ``list``, "
                + f"but it is type ``{type(partition)}``.",
            )

        if any([not isinstance(party, int) for party in partition]):
            raise_error(
                ValueError,
                "``partition`` must be a ``tuple`` or ``list`` of positive integers, "
                + "but contains non-integers.",
            )

        if any([party <= 0 for party in partition]):
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

        self.np = self._backend.np

        if isinstance(self.partition, list):
            self.partition = tuple(self.partition)

        if self._is_pure:
            self.matrix = np.reshape(self.matrix, self.partition)
        else:
            matrix_partition = self.partition * 2
            self.matrix = np.reshape(self.matrix, matrix_partition)

        if self.system_output is None:
            self.system_output = (False,) if len(self.partition) == 1 else (False, True)
        else:
            self.system_output = tuple(self.system_output)

    @property
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
            precision_tol (float, optional): threshold :math:`\\epsilon` that defines if Choi operator
                of the network is :math:`\\epsilon`-close to Hermicity in the norm given by ``order``.
                Defaults to :math:`10^{-8}`.

        Returns:
            bool: Hermiticity condition.
        """
        if precision_tol < 0.0:
            raise_error(
                ValueError,
                f"``precision_tol`` must be non-negative float, but it is {precision_tol}",
            )

        reshaped = np.reshape(self.matrix, (self.dims, self.dims))
        return (
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
            precision_tol (float, optional): threshold :math:`\\epsilon` that defines if Choi operator
                of the network is :math:`\\epsilon`-close to unitality in the norm given by ``order``.
                Defaults to :math:`10^{-8}`.

        Returns:
            bool: Unitality condition.
        """
        if precision_tol < 0.0:
            raise_error(
                ValueError,
                f"``precision_tol`` must be non-negative float, but it is {precision_tol}",
            )

        partial_trace = np.einsum("jklm -> km", self.matrix)
        return (
            self._backend.calculate_norm_density_matrix(
                partial_trace
                - self._backend.identity_density_matrix(
                    self.matrix.shape[1], normalize=False
                ),
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
            precision_tol (float, optional): threshold :math:`\\epsilon` that defines if Choi operator
                of the network is :math:`\\epsilon`-close to causality in the norm given by ``order``.
                Defaults to :math:`10^{-8}`.

        Returns:
            bool: Causal order condition.
        """
        if precision_tol < 0.0:
            raise_error(
                ValueError,
                f"``precision_tol`` must be non-negative float, but it is {precision_tol}",
            )

        partial_trace = np.einsum("jklm -> km", self.matrix)
        return (
            self._backend.calculate_norm_density_matrix(
                partial_trace
                - self._backend.identity_density_matrix(
                    self.matrix.shape[1], normalize=False
                ),
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
        reshaped = np.reshape(self.matrix, (self.dims, self.dims))
        if self.is_hermitian:
            eigenvalues = np.linalg.eigvalsh(reshaped)
        else:
            if self._backend.__class__.__name__ in ["CupyBackend", "CuQuantumBackend"]:
                reshaped = np.array(reshaped.tolist(), dtype=reshaped.dtype)
            eigenvalues = np.linalg.eigvals(reshaped)

        return all([eigenvalue >= precision_tol for eigenvalue in eigenvalues])

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
        if self.is_pure:
            return np.einsum("jk,lm,jl -> km", self.matrix, np.conj(self.matrix), state)

        return np.einsum("jklm,kl -> jl", self.matrix, state)
