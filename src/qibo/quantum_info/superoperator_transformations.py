"""Module with the most commom superoperator transformations."""

import math
import warnings
from functools import reduce
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from qibo.backends import Backend, _check_backend
from qibo.config import PRECISION_TOL, raise_error
from qibo.gates.abstract import Gate
from qibo.gates.gates import Unitary
from qibo.gates.special import FusedGate
from qibo.quantum_info.linalg_operations import singular_value_decomposition
from qibo.quantum_info.utils import _get_single_paulis, _pauli_basis_normalization


def vectorization(state, order: str = "row", backend: Optional[Backend] = None):
    """Returns state :math:`\\rho` in its Liouville representation :math:`|\\rho)`.

    If ``order="row"``, then:

    .. math::
        |\\rho) = \\sum_{k, l} \\, \\rho_{kl} \\, \\ket{k} \\otimes \\ket{l} \\, .

    If ``order="column"``, then:

    .. math::
        |\\rho) = \\sum_{k, l} \\, \\rho_{kl} \\, \\ket{l} \\otimes \\ket{k} \\, .

    If ``state`` is a 3-dimensional tensor, it is interpreted as a batch of states.

    Args:
        state (ndarray): statevector, density matrix, an array of statevectors,
            or an array of density matrices.
        order (str, optional): If ``"row"``, vectorization is performed
            row-wise. If ``"column"``, vectorization is performed
            column-wise. If ``"system"``, a block-vectorization is
            performed. Defaults to ``"row"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Liouville representation of ``state``.
    """
    if (
        (len(state.shape) > 3)
        or (len(state) == 0)
        or (len(state.shape) == 2 and state.shape[0] != state.shape[1])
    ):
        raise_error(
            TypeError,
            f"Object must have dims either (k,), (k, k), (N, 1, k) or (N, k, k), but have dims {state.shape}.",
        )

    if order not in ["row", "column", "system"]:
        raise_error(
            ValueError,
            f"order must be either 'row' or 'column' or 'system', but it is {order}.",
        )

    backend = _check_backend(backend)

    dims = state.shape[-1]

    if len(state.shape) == 1:
        state = backend.outer(state, backend.conj(state))
    elif len(state.shape) == 3 and state.shape[1] == 1:
        state = backend.einsum("aij,akl->aijkl", state, backend.conj(state)).reshape(
            state.shape[0], dims, dims
        )

    if order == "row":
        state = backend.qinfo._vectorization_row(state, dims)
    elif order == "column":
        state = backend.qinfo._vectorization_column(state, dims)
    else:
        state = backend.qinfo._vectorization_system(state)

    state = backend.squeeze(
        state, axis=tuple(i for i, ax in enumerate(state.shape) if ax == 1)
    )

    return state


def unvectorization(state, order: str = "row", backend: Optional[Backend] = None):
    """Returns state :math:`\\rho` from its Liouville
    representation :math:`|\\rho)`. This operation is
    the inverse function of :func:`vectorization`, i.e.

    .. math::
        \\begin{align}
            \\rho &= \\text{unvectorization}(|\\rho)) \\nonumber \\\\
            &= \\text{unvectorization}(\\text{vectorization}(\\rho)) \\nonumber
        \\end{align}

    If ``state`` has shape (N, k), it is interpreted as a batch of states.

    Args:
        state: quantum state or batch of states in the Liouville representation.
        order (str, optional): If ``"row"``, unvectorization is performed
            row-wise. If ``"column"``, unvectorization is performed
            column-wise. If ``"system"``, system-wise vectorization is
            performed. Defaults to ``"row"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Density matrix of ``state``.
    """

    if len(state.shape) not in (1, 2):
        raise_error(
            TypeError,
            f"Object must have dims (k,) or (N, k), but have dims {state.shape}.",
        )

    if order not in ["row", "column", "system"]:
        raise_error(
            ValueError,
            f"order must be either 'row' or 'column' or 'system', but it is {order}.",
        )
    backend = _check_backend(backend)
    state = backend.cast(state, dtype=state.dtype)

    dim = int(math.sqrt(state.shape[-1]))
    if len(state.shape) == 1:
        state = state.reshape(1, -1)
    func = getattr(backend.qinfo, f"_unvectorization_{order}")
    state = func(state, dim)
    if state.shape[0] == 1:
        state = backend.squeeze(state, 0)
    return state


def to_choi(channel, order: str = "row", backend: Optional[Backend] = None):
    """Converts quantum ``channel`` :math:`U` to its Choi representation :math:`\\Lambda`.

    .. math::
        \\Lambda = | U ) ( U | \\, ,

    where :math:`| \\cdot )` is the :func:`qibo.quantum_info.vectorization`
    operation.

    Args:
        channel (ndarray): quantum channel.
        order (str, optional): If ``"row"``, vectorization is performed
            row-wise. If ``"column"``, vectorization is performed
            column-wise. If ``"system"``, a block-vectorization is
            performed. Default is ``"row"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: quantum channel in its Choi representation.
    """
    backend = _check_backend(backend)

    func_order = getattr(backend.qinfo, f"_to_choi_{order}")
    return func_order(channel)


def to_liouville(channel, order: str = "row", backend: Optional[Backend] = None):
    """Converts quantum ``channel`` :math:`U` to its Liouville representation
    :math:`\\mathcal{E}`. It uses the Choi representation as an
    intermediate step.

    Args:
        channel (ndarray): quantum channel.
        order (str, optional): If ``"row"``, vectorization is performed
            row-wise. If ``"column"``, vectorization is performed
            column-wise. If ``"system"``, a block-vectorization is
            performed. Default is ``"row"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: quantum channel in its Liouville representation.
    """
    backend = _check_backend(backend)

    func_order = getattr(backend.qinfo, f"_to_liouville_{order}")
    return func_order(channel)


def to_pauli_liouville(
    channel,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    method: Optional[str] = None,
    backend: Optional[Backend] = None,
):
    r"""Converts quantum ``channel`` :math:`U` to its Pauli-Liouville
    representation :math:`\\mathcal{E}`. It uses the Liouville representation
    as an intermediate step.

    If ``method`` is ``None`` or ``"fht"``, the conversion from the
    computational basis to the Pauli basis is performed with a Fast
    Walsh-Hadamard transform. For an operator
    :math:`A = (a_{p, q})_{p, q = 0}^{2^{n} - 1}`, this computes the
    coefficients :math:`\alpha_{r, s}` of

    .. math::
        A = \sum_{r, s = 0}^{2^{n} - 1} \alpha_{r, s} P_{r, s},

    where

    .. math::
        P_{r, s} = \bigotimes_{j = 0}^{n - 1}
            i^{r_j \wedge s_j} X^{r_j} Z^{s_j},

    using

    .. math::
        \alpha_{r, s} = \frac{(-i)^{|r \wedge s|}}{2^n}
            \sum_{q = 0}^{2^{n} - 1} a_{q \oplus r, q}
            (H^{\otimes n})_{q, s}.

    Args:
        channel (ndarray): quantum channel.
        normalize (bool, optional): If ``True`` superoperator is returned
            in the normalized Pauli basis. If ``False``, it is returned
            in the unnormalized Pauli basis. Defaults to ``False``.
        order (str, optional): If ``"row"``, vectorization is performed
            row-wise. If ``"column"``, vectorization is performed
            column-wise. If ``"system"``, a block-vectorization is
            performed. Default is ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements. Default is "IXYZ".
        method (str, optional): If ``None`` or ``"fht"``, uses the Fast
            Walsh-Hadamard transform. If ``"standard"``, uses the dense
            Pauli basis-change matrix. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
           to be used in the execution. If ``None``, it uses
           the current backend. Defaults to ``None``.

    Returns:
        ndarray: quantum channel in its Pauli-Liouville representation.
    """
    method = _check_pauli_transform_method(method)
    backend = _check_backend(backend)

    if method == "fht":
        return _to_pauli_liouville_fht(
            channel,
            normalize=normalize,
            order=order,
            pauli_order=pauli_order,
            backend=backend,
        )

    normalization = (
        _pauli_basis_normalization(int(math.log2(channel.shape[0])))
        if normalize
        else 1.0
    )
    func_order = getattr(backend.qinfo, f"_to_pauli_liouville_{order}")
    return func_order(
        channel, *_get_single_paulis(pauli_order, backend), normalization=normalization
    )


def to_chi(
    channel,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
):
    """Converts quantum ``channel`` :math:`U` to its :math:`\\chi`-representation.

    Args:
        channel (ndarray): quantum channel.
        normalize (bool, optional): If ``True`` superoperator is returned
            in the normalized Pauli basis. If ``False``, it is returned
            in the unnormalized Pauli basis. Defaults to ``False``.
        order (str, optional): If ``"row"``, vectorization is performed
            row-wise. If ``"column"``, vectorization is performed
            column-wise. If ``"system"``, a block-vectorization is
            performed. Default is ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements. Default is "IXYZ".
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: quantum channel in its :math:`\\chi`-representation.
    """
    channel = to_choi(channel, order=order, backend=backend)
    channel = liouville_to_pauli(
        channel,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    return channel


def to_stinespring(
    channel,
    partition: Optional[Union[List[int], Tuple[int, ...]]] = None,
    nqubits: Optional[int] = None,
    initial_state_env=None,
    backend: Optional[Backend] = None,
):
    """Convert quantum ``channel`` :math:`U` to its Stinespring representation :math:`U_{0}`.

    It uses the Kraus representation as an intermediate step.

    Args:
        channel (ndarray): quantum channel.
        nqubits (int, optional): total number of qubits in the system that is
            interacting with the environment. Must be equal or greater than
            the number of qubits ``channel`` acts on. If ``None``,
            defaults to the number of qubits in ``channel``.
            Defauts to ``None``.
        initial_state_env (ndarray, optional): statevector representing the
            initial state of the enviroment. If ``None``, it assumes the
            environment in its ground state. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ndarray: Quantum channel in its Stinespring representation :math:`U_{0}`.
    """
    backend = _check_backend(backend)

    if partition is None:
        nqubits_channel = int(math.log2(channel.shape[-1]))
        partition = tuple(range(nqubits_channel))

    channel = kraus_to_stinespring(
        [(partition, channel)],
        nqubits=nqubits,
        initial_state_env=initial_state_env,
        backend=backend,
    )

    return channel


def choi_to_liouville(
    choi_super_op, order: str = "row", backend: Optional[Backend] = None
):
    """Converts Choi representation :math:`\\Lambda` of quantum channel
    to its Liouville representation :math:`\\mathcal{E}`.


    If ``order="row"``, then:

    .. math::
        \\Lambda_{\\alpha\\beta, \\, \\gamma\\delta} \\mapsto
            \\Lambda_{\\alpha\\gamma, \\, \\beta\\delta} \\equiv \\mathcal{E}

    If ``order="column"``, then:

    .. math::
        \\Lambda_{\\alpha\\beta, \\, \\gamma\\delta} \\mapsto
            \\Lambda_{\\delta\\beta, \\, \\gamma\\alpha} \\equiv \\mathcal{E}

    Args:
        choi_super_op: Choi representation of quantum channel.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Liouville representation of quantum channel.
    """

    return _reshuffling(choi_super_op, order=order, backend=backend)


def choi_to_pauli(
    choi_super_op,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
):
    """Converts Choi representation :math:`\\Lambda` of a quantum channel
    to its Pauli-Liouville representation.

    Args:
        choi_super_op (ndarray): superoperator in the Choi representation.
        normalize (bool, optional): If ``True`` superoperator is returned
            in the normalized Pauli basis. If ``False``, it is returned
            in the unnormalized Pauli basis. Defaults to ``False``.
        order (str, optional): If ``"row"``, it assumes ``choi_super_op`` is in
            row-vectorization. If ``"column"``, it assumes column-vectorization.
            Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements. Defaults to "IXYZ".
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: superoperator in the Pauli-Liouville representation.
    """
    super_op = choi_to_liouville(choi_super_op, order, backend=backend)
    super_op = liouville_to_pauli(
        super_op, normalize, order, pauli_order, backend=backend
    )

    return super_op


def choi_to_kraus(
    choi_super_op,
    precision_tol: Optional[float] = None,
    order: str = "row",
    validate_cp: bool = True,
    backend: Optional[Backend] = None,
):
    """Converts Choi representation :math:`\\Lambda` of a quantum channel :math:`\\mathcal{E}`
    into Kraus operators :math:`\\{ K_{\\alpha} \\}_{\\alpha}`.

    If :math:`\\mathcal{E}` is a completely positive (CP) map, then

    .. math::
        \\Lambda = \\sum_{\\alpha} \\, \\lambda_{\\alpha}^{2} \\,
            |\\tilde{K}_{\\alpha})(\\tilde{K}_{\\alpha}| \\, .

    where :math:`|\\cdot)` is the :func:`qibo.quantum_info.vectorization` operation.

    This is the spectral decomposition of :math:`\\Lambda`, Hence, the set
    :math:`\\{\\lambda_{\\alpha}, \\, \\tilde{K}_{\\alpha}\\}_{\\alpha}`
    is found by diagonalization of :math:`\\Lambda`. The Kraus operators
    :math:`\\{K_{\\alpha}\\}_{\\alpha}` are defined as

    .. math::
        K_{\\alpha} = \\lambda_{\\alpha} \\,
            \\text{unvectorization}(|\\tilde{K}_{\\alpha})) \\, .

    If :math:`\\mathcal{E}` is not CP, then spectral composition is replaced by
    a singular value decomposition (SVD), i.e.

    .. math::
        \\Lambda = U \\, S \\, V^{\\dagger} \\, ,

    where :math:`U` is a :math:`d^{2} \\times d^{2}` unitary matrix, :math:`S` is a
    :math:`d^{2} \\times d^{2}` positive diagonal matrix containing the singular values
    of :math:`\\Lambda`, and :math:`V` is a :math:`d^{2} \\times d^{2}` unitary matrix.
    The Kraus coefficients are replaced by the square root of the singular values, and
    :math:`U` (:math:`V`) determine the left-generalized (right-generalized) Kraus
    operators.

    Args:
        choi_super_op: Choi representation of a quantum channel.
        precision_tol (float, optional): Precision tolerance for eigenvalues
            found in the spectral decomposition problem. Any eigenvalue
            :math:`\\lambda <` ``precision_tol`` is set to 0 (zero).
            If ``None``, ``precision_tol`` defaults to
            ``qibo.config.PRECISION_TOL=1e-8``. Defaults to ``None``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        validate_cp (bool, optional): If ``True``, checks if ``choi_super_op``
            is a completely positive map. If ``False``, it assumes that
            ``choi_super_op`` is completely positive (and Hermitian).
            Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        tuple(ndarray, ndarray): The set
        :math:`\\{K_{\\alpha}, \\, \\lambda_{\\alpha} \\}_{\\alpha}`
        of Kraus operators representing the quantum channel and their respective coefficients.
        If map is non-CP, then function returns the set
        :math:`\\{ \\{K_{L}, \\, K_{R}\\}_{\\alpha}, \\, \\lambda_{\\alpha} \\}_{\\alpha}`,
        with the left- and right-generalized Kraus operators as well as the square root of
        their corresponding singular values.
    """
    if precision_tol is None:  # pragma: no cover
        precision_tol = PRECISION_TOL

    backend = _check_backend(backend)
    choi_super_op = backend.cast(choi_super_op, dtype=choi_super_op.dtype)

    if validate_cp:
        norm = float(
            backend.matrix_norm(choi_super_op - backend.conj(choi_super_op).T, order=2)
        )
        if norm > PRECISION_TOL:
            non_cp = True
        else:
            # using eigh because, in this case, choi_super_op is
            # *already confirmed* to be Hermitian
            eigenvalues, eigenvectors = backend.eigenvectors(choi_super_op)
            eigenvectors = eigenvectors.T

            non_cp = bool(any(backend.real(eigenvalues) < -PRECISION_TOL))
    else:
        non_cp = False
        # using eigh because, in this case, choi_super_op is
        # *assumed* to be Hermitian
        eigenvalues, eigenvectors = backend.eigenvectors(choi_super_op)
        eigenvectors = eigenvectors.T

    if non_cp:
        warnings.warn("Input choi_super_op is a non-completely positive map.")

        # using singular value decomposition because choi_super_op is non-CP
        func_order = getattr(backend.qinfo, f"_choi_to_kraus_{order}")
        kraus_ops, coefficients = func_order(choi_super_op)
    else:
        # when choi_super_op is CP
        func_order = getattr(backend.qinfo, f"_choi_to_kraus_cp_{order}")
        kraus_ops, coefficients = func_order(eigenvalues, eigenvectors, precision_tol)

    return kraus_ops, coefficients


def choi_to_chi(
    choi_super_op,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
):
    """Converts Choi representation :math:`\\Lambda` of quantum channel
    to  its :math:`\\chi`-matrix representation.

    .. math::
        \\chi = \\text{liouville_to_pauli}(\\Lambda)

    Args:
        choi_super_op: Choi representation of a quantum channel.
        normalize (bool, optional): If ``True`` assumes the normalized
            Pauli basis. If ``False``, it assumes unnormalized
            Pauli basis. Defaults to ``False``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4
            single-qubit Pauli elements. Defaults to "IXYZ".
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.
    Returns:
        ndarray: Chi-matrix representation of the quantum channel.
    """
    process_matrix = liouville_to_pauli(
        choi_super_op,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    return process_matrix


def choi_to_stinespring(
    choi_super_op,
    precision_tol: Optional[float] = None,
    order: str = "row",
    validate_cp: bool = True,
    nqubits: Optional[int] = None,
    initial_state_env=None,
    backend: Optional[Backend] = None,
):
    """Converts Choi representation :math:`\\Lambda` of quantum channel
    to its Stinespring representation :math:`U_{0}`.
    It uses the Kraus representation as an intermediate step.

    Args:
        choi_super_op: Choi representation of a quantum channel.
        precision_tol (float, optional): Precision tolerance for eigenvalues
            found in the spectral decomposition problem. Any eigenvalue
            :math:`\\lambda <` ``precision_tol`` is set to 0 (zero).
            If ``None``, ``precision_tol`` defaults to
            ``qibo.config.PRECISION_TOL=1e-8``. Defaults to ``None``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        validate_cp (bool, optional): If ``True``, checks if ``choi_super_op``
            is a completely positive map. If ``False``, it assumes that
            ``choi_super_op`` is completely positive (and Hermitian).
            Defaults to ``True``.
        nqubits (int, optional): total number of qubits in the system that is
            interacting with the environment. Must be equal or greater than
            the number of qubits ``kraus_ops`` acts on. If ``None``,
            defaults to the number of qubits in ``kraus_ops``.
            Defauts to ``None``.
        initial_state_env (ndarray, optional): statevector representing the
            initial state of the enviroment. If ``None``, it assumes the
            environment in its ground state. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Choi representation of quantum channel.
    """
    kraus_ops, _ = choi_to_kraus(
        choi_super_op,
        precision_tol=precision_tol,
        order=order,
        validate_cp=validate_cp,
        backend=backend,
    )

    if validate_cp is True and len(kraus_ops.shape) != 3:
        raise_error(
            NotImplementedError,
            "Stinespring representation not implemented for non-completely positive maps.",
        )

    if nqubits is None:
        nqubits = int(math.log2(kraus_ops[0].shape[0]))

    nqubits_list = [tuple(range(nqubits))] * len(kraus_ops)

    kraus_ops = list(zip(nqubits_list, kraus_ops))

    stinespring = kraus_to_stinespring(
        kraus_ops, nqubits=nqubits, initial_state_env=initial_state_env, backend=backend
    )

    return stinespring


def kraus_to_choi(kraus_ops, order: str = "row", backend: Optional[Backend] = None):
    """Converts Kraus representation :math:`\\{K_{\\alpha}\\}_{\\alpha}`
    of quantum channel to its Choi representation :math:`\\Lambda`.

    .. math::
        \\Lambda = \\sum_{\\alpha} \\, |K_{\\alpha})( K_{\\alpha}|

    where :math:`|K_{\\alpha})` is the vectorization of the Kraus operator
    :math:`K_{\\alpha}`.
    For a definition of vectorization, see :func:`qibo.quantum_info.vectorization`.

    Args:
        kraus_ops (list): List of Kraus operators as pairs ``(qubits, Ak)``
            where ``qubits`` refers the qubit ids that :math:`A_k`  acts on
            and :math:`A_k` is the corresponding matrix as a ``ArrayLike``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Choi representation of the Kraus channel.
    """
    backend = _check_backend(backend)

    gates, target_qubits = _set_gate_and_target_qubits(kraus_ops)
    nqubits = 1 + max(target_qubits)

    kraus_ops = []
    for gate in gates:
        kraus_op = FusedGate(*range(nqubits))
        kraus_op.append(gate)
        kraus_ops.append(kraus_op.matrix(backend)[None, :])
        del kraus_op
    kraus_ops = backend.vstack(kraus_ops)
    func_order = getattr(backend.qinfo, f"_kraus_to_choi_{order}")
    return func_order(kraus_ops)


def kraus_to_liouville(
    kraus_ops, order: str = "row", backend: Optional[Backend] = None
):
    """Converts from Kraus representation :math:`\\{K_{\\alpha}\\}_{\\alpha}`
    of quantum channel to its Liouville representation :math:`\\mathcal{E}`.
    It uses the Choi representation as an intermediate step.

    .. math::
        \\mathcal{E} = \\text{choi_to_liouville}(\\text{kraus_to_choi}
            (\\{K_{\\alpha}\\}_{\\alpha}))

    Args:
        kraus_ops (list): List of Kraus operators as pairs ``(qubits, Ak)``
            where ``qubits`` refers the qubit ids that :math:`A_k` acts on
            and :math:`A_k` is the corresponding matrix as a ``ArrayLike``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Liouville representation of quantum channel.
    """
    super_op = kraus_to_choi(kraus_ops, order=order, backend=backend)
    super_op = choi_to_liouville(super_op, order=order, backend=backend)

    return super_op


def kraus_to_pauli(
    kraus_ops,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
):
    """Converts Kraus representation :math:`\\{K_{\\alpha}\\}_{\\alpha}`
    of a quantum channel to its Pauli-Liouville representation.

    Args:
        kraus_ops (list): List of Kraus operators as pairs ``(qubits, Ak)``
            where ``qubits`` refers the qubit ids that :math:`A_k` acts on
            and :math:`A_k` is the corresponding matrix as a ``ArrayLike``.
        normalize (bool, optional): If ``True`` superoperator is returned
            in the normalized Pauli basis. If ``False``, it is returned
            in the unnormalized Pauli basis. Defaults to ``False``.
        order (str, optional): If ``"row"``, intermediate step for Choi
            representation is done in row-vectorization. If ``"column"``,
            step is done in column-vectorization. If ``"system"``,
            block-vectorization is performed. Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4
            single-qubit Pauli elements. Defaults to "IXYZ".
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: superoperator in the Pauli-Liouville representation.
    """
    super_op = kraus_to_choi(kraus_ops, order, backend=backend)
    super_op = choi_to_pauli(super_op, normalize, order, pauli_order, backend=backend)

    return super_op


def kraus_to_chi(
    kraus_ops,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
):
    """Convert Kraus representation :math:`\\{K_{\\alpha}\\}_{\\alpha}`
    of quantum channel to  its :math:`\\chi`-matrix representation.

    .. math::
        \\chi = \\sum_{\\alpha} \\, |c_{\\alpha})( c_{\\alpha}|,

    where :math:`|c_{\\alpha}) \\cong |K_{\\alpha})` in Pauli-Liouville basis,
    and :math:`| \\cdot )` is the :func:`qibo.quantum_info.vectorization`
    operation.

    Args:
        kraus_ops (list): List of Kraus operators as pairs ``(qubits, Ak)``
            where ``qubits`` refers the qubit ids that :math:`A_k`  acts on
            and :math:`A_k` is the corresponding matrix as a ``ArrayLike``.
        normalize (bool, optional): If ``True`` assumes the normalized
            Pauli basis. If ``False``, it assumes unnormalized
            Pauli basis. Defaults to ``False``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements in the Pauli basis. Defaults to "IXYZ".
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Chi-matrix representation of the Kraus channel.
    """
    from qibo.quantum_info.basis import comp_basis_to_pauli  # pylint: disable=C0415

    backend = _check_backend(backend)

    gates, target_qubits = _set_gate_and_target_qubits(kraus_ops)
    nqubits = 1 + max(target_qubits)
    dim = 2**nqubits

    comp_to_pauli = comp_basis_to_pauli(
        int(nqubits),
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    super_op = backend.zeros((dim**2, dim**2), dtype=backend.complex128)
    for gate in gates:
        kraus_op = FusedGate(*range(nqubits))
        kraus_op.append(gate)
        kraus_op = kraus_op.matrix(backend)
        kraus_op = vectorization(kraus_op, order=order, backend=backend)
        kraus_op = comp_to_pauli @ kraus_op
        super_op = super_op + backend.outer(kraus_op, backend.conj(kraus_op))
        del kraus_op

    return super_op


def kraus_to_stinespring(
    kraus_ops,
    nqubits: Optional[int] = None,
    initial_state_env: Optional[ArrayLike] = None,
    backend: Optional[Backend] = None,
):
    """Converts Kraus representation :math:`\\{K_{\\alpha}\\}_{\\alpha}`
    of quantum channel to its Stinespring representation :math:`U_{0}`, i.e.

    .. math::
        U_{0} = \\sum_{\\alpha} \\, K_{\\alpha} \\otimes \\ketbra{\\alpha}{v_{0}} \\, ,

    where :math:`\\ket{v_{0}}` is the initial state of the environment
    (``initial_state_env``), :math:`D` is the dimension of the environment's
    Hilbert space, and
    :math:`\\{\\ket{\\alpha} \\, : \\, \\alpha = 0, 1, \\cdots, D - 1 \\}`
    is an orthonormal basis for the environment's space.

    Args:
        kraus_ops (list): List of Kraus operators as pairs ``(qubits, Ak)``
            where ``qubits`` refers the qubit ids that :math:`A_k`  acts on
            and :math:`A_k` is the corresponding matrix as a ``ArrayLike``.
        nqubits (int, optional): total number of qubits in the system that is
            interacting with the environment. Must be equal or greater than
            the number of qubits ``kraus_ops`` acts on. If ``None``,
            defaults to the number of qubits in ``kraus_ops``.
            Defauts to ``None``.
        initial_state_env (ndarray, optional): statevector representing the
            initial state of the enviroment. If ``None``, it assumes the
            environment in its ground state. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Stinespring representation (restricted unitary) of the Kraus channel.
    """
    backend = _check_backend(backend)

    if initial_state_env is not None:
        if len(initial_state_env) != len(kraus_ops):
            raise_error(
                ValueError,
                "dim of initial_state_env must be equal to the number of Kraus operators.",
            )

        if len(initial_state_env.shape) != 1:
            raise_error(ValueError, "initial_state_env must be a statevector.")

    gates, target_qubits = _set_gate_and_target_qubits(kraus_ops)

    if nqubits is None:
        nqubits = 1 + max(target_qubits)

    dim_env = len(kraus_ops)

    if initial_state_env is None:
        initial_state_env = backend.zeros(dim_env, dtype=backend.complex128)
        initial_state_env[0] = 1.0

    initial_state_env = backend.conj(initial_state_env)

    kraus_ops = []
    for gate in gates:
        kraus_op = FusedGate(*range(nqubits))
        kraus_op.append(gate)
        kraus_ops.append(kraus_op.matrix(backend)[None, :])
        del kraus_op
    kraus_ops = backend.vstack(kraus_ops)
    return backend.qinfo._kraus_to_stinespring(kraus_ops, initial_state_env, dim_env)


def liouville_to_choi(super_op, order: str = "row", backend: Optional[Backend] = None):
    """Converts Liouville representation of quantum channel :math:`\\mathcal{E}`
    to its Choi representation :math:`\\Lambda`. Indexing :math:`\\mathcal{E}` as
    :math:`\\mathcal{E}_{\\alpha\\beta, \\, \\gamma\\delta} \\,\\,`, then

    If ``order="row"``:

    .. math::
        \\Lambda = \\sum_{k, l} \\, \\ketbra{k}{l} \\otimes \\mathcal{E}(\\ketbra{k}{l})
            \\equiv \\mathcal{E}_{\\alpha\\gamma, \\, \\beta\\delta}

    If ``order="column"``, then:

    .. math::
            \\Lambda = \\sum_{k, l} \\, \\mathcal{E}(\\ketbra{k}{l}) \\otimes \\ketbra{k}{l}
                \\equiv \\mathcal{E}_{\\delta\\beta, \\, \\gamma\\alpha}

    Args:
        super_op: Liouville representation of quantum channel.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Choi representation of quantum channel.
    """

    return _reshuffling(super_op, order=order, backend=backend)


def liouville_to_pauli(
    super_op,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    method: Optional[str] = None,
    backend: Optional[Backend] = None,
):
    r"""Converts Liouville representation :math:`\\mathcal{E}` of a
    quantum channel to its Pauli-Liouville representation.

    If ``method`` is ``None`` or ``"fht"``, this function applies the
    Fast Walsh-Hadamard transform-based Pauli decomposition to the
    Liouville operator without explicitly constructing the dense Pauli
    basis-change matrix. Setting ``method="standard"`` recovers the
    dense matrix multiplication implementation.

    Args:
        super_op (ndarray): superoperator in the Liouville representation._
        normalize (bool, optional): If ``True`` superoperator is returned
            in the normalized Pauli basis. If ``False``, it is returned
            in the unnormalized Pauli basis. Defaults to ``False``.
        order (str, optional): If ``"row"``, it assumes ``super_op`` is in
            row-vectorization. If ``"column"``, it assumes column-vectorization.
            If ``"system"``, it assumes block-vectorization. Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements in the basis. Defaults to "IXYZ".
        method (str, optional): If ``None`` or ``"fht"``, uses the Fast
            Walsh-Hadamard transform. If ``"standard"``, uses the dense
            Pauli basis-change matrix. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
           to be used in the execution. If ``None``, it uses
           the current backend. Defaults to ``None``.

    Returns:
        ndarray: superoperator in the Pauli-Liouville representation.
    """
    from qibo.quantum_info.basis import comp_basis_to_pauli  # pylint: disable=C0415

    method = _check_pauli_transform_method(method)
    backend = _check_backend(backend)

    dim, nqubits = _check_pauli_superoperator_shape(super_op, "super_op")

    if method == "fht":
        return _liouville_to_pauli_fht(
            super_op,
            nqubits=nqubits,
            dim=dim,
            normalize=normalize,
            order=order,
            pauli_order=pauli_order,
            backend=backend,
        )

    comp_to_pauli = comp_basis_to_pauli(
        nqubits,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    return comp_to_pauli @ super_op @ backend.conj(comp_to_pauli.T)


def liouville_to_kraus(
    super_op,
    precision_tol: Optional[float] = None,
    order: str = "row",
    backend: Optional[Backend] = None,
):
    """Converts Liouville representation :math:`\\mathcal{E}` of a quantum
    channel to its Kraus representation :math:`\\{K_{\\alpha}\\}_{\\alpha}`.
    It uses the Choi representation as an intermediate step.

    .. math::
        \\{K_{\\alpha}, \\, \\lambda_{\\alpha}\\}_{\\alpha} =
            \\text{choi_to_kraus}(\\text{liouville_to_choi}(\\mathcal{E}))

    Args:
        super_op (ndarray): Liouville representation of quantum channel.
        precision_tol (float, optional): Precision tolerance for eigenvalues
            found in the spectral decomposition problem. Any eigenvalue
            :math:`\\lambda < \\text{precision_tol}` is set to 0 (zero).
            If ``None``, ``precision_tol`` defaults to
            ``qibo.config.PRECISION_TOL=1e-8``. Defaults to None.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        (ndarray, ndarray): Kraus operators of quantum channel and their respective coefficients.
    """
    choi_super_op = liouville_to_choi(super_op, order=order, backend=backend)
    kraus_ops, coefficients = choi_to_kraus(
        choi_super_op, precision_tol, order=order, backend=backend
    )

    return kraus_ops, coefficients


def liouville_to_chi(
    super_op,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
):
    """Converts Liouville representation of quantum channel :math:`\\mathcal{E}`
    to its :math:`\\chi`-matrix representation.
    It uses the Choi representation as an intermediate step.

    .. math::
        \\chi = \\text{liouville_to_pauli}(\\text{liouville_to_choi}(\\mathcal{E}))

    Args:
        super_op: Liouville representation of quantum channel.
        normalize (bool, optional): If ``True`` assumes the normalized
            Pauli basis. If ``False``, it assumes unnormalized
            Pauli basis. Defaults to ``False``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements in the basis. Defaults to "IXYZ".
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Chi-matrix representation of quantum channel.
    """

    choi_super_op = liouville_to_choi(super_op, order=order, backend=backend)
    process_matrix = liouville_to_pauli(
        choi_super_op,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    return process_matrix


def liouville_to_stinespring(
    super_op,
    order: str = "row",
    precision_tol: Optional[float] = None,
    validate_cp: bool = True,
    nqubits: Optional[int] = None,
    initial_state_env=None,
    backend: Optional[Backend] = None,
):
    """Converts Liouville representation :math:`\\mathcal{E}` of quantum channel
    to its Stinespring representation :math:`U_{0}`.
    It uses the Choi representation :math:`\\Lambda` as intermediate step.

    Args:
        super_op: Liouville representation of quantum channel.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        precision_tol (float, optional): Precision tolerance for eigenvalues
            found in the spectral decomposition problem. Any eigenvalue
            :math:`\\lambda <` ``precision_tol`` is set to 0 (zero).
            If ``None``, ``precision_tol`` defaults to
            ``qibo.config.PRECISION_TOL=1e-8``. Defaults to ``None``.
        validate_cp (bool, optional): If ``True``, checks if ``choi_super_op``
            is a completely positive map. If ``False``, it assumes that
            ``choi_super_op`` is completely positive (and Hermitian).
            Defaults to ``True``.
        nqubits (int, optional): total number of qubits in the system that is
            interacting with the environment. Must be equal or greater than
            the number of qubits ``kraus_ops`` acts on. If ``None``,
            defaults to the number of qubits in ``kraus_ops``.
            Defauts to ``None``.
        initial_state_env (ndarray, optional): statevector representing the
            initial state of the enviroment. If ``None``, it assumes the
            environment in its ground state. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Stinespring representation of quantum channel.
    """
    choi_super_op = liouville_to_choi(super_op, order=order, backend=backend)
    stinespring = choi_to_stinespring(
        choi_super_op,
        precision_tol=precision_tol,
        order=order,
        validate_cp=validate_cp,
        nqubits=nqubits,
        initial_state_env=initial_state_env,
        backend=backend,
    )

    return stinespring


def pauli_to_liouville(
    pauli_op,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    method: Optional[str] = None,
    backend: Optional[Backend] = None,
):
    r"""Converts Pauli-Liouville representation of a quantum channel to its
    Liouville representation :math:`\\mathcal{E}`.

    If ``method`` is ``None`` or ``"fht"``, the inverse Pauli-basis
    transform is evaluated with the inverse Fast Walsh-Hadamard transform
    implied by Theorem 1. Setting ``method="standard"`` recovers the dense
    Pauli basis-change matrix implementation.

    Args:
        pauli_op (ndarray): Pauli-Liouville representation of a quantum channel.
        normalize (bool, optional): If ``True`` assumes ``pauli_op`` is represented
            in the normalized Pauli basis. If ``False``, it assumes unnormalized
            Pauli basis. Defaults to ``False``.
        order (str, optional): If ``"row"``, returns Liouville representation in
            row-vectorization. If ``"column"``, returns column-vectorized
            superoperator. If ``"system"``, superoperator will be in
            block-vectorization. Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements. Defaults to "IXYZ".
        method (str, optional): If ``None`` or ``"fht"``, uses the Fast
            Walsh-Hadamard transform. If ``"standard"``, uses the dense
            Pauli basis-change matrix. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
           to be used in the execution. If ``None``, it uses
           the current backend. Defaults to ``None``.

    Returns:
        ndarray: superoperator in the Liouville representation.
    """
    from qibo.quantum_info.basis import pauli_to_comp_basis  # pylint: disable=C0415

    method = _check_pauli_transform_method(method)
    backend = _check_backend(backend)

    dim, nqubits = _check_pauli_superoperator_shape(pauli_op, "pauli_op")

    if method == "fht":
        return _pauli_to_liouville_fht(
            pauli_op,
            nqubits=nqubits,
            dim=dim,
            normalize=normalize,
            order=order,
            pauli_order=pauli_order,
            backend=backend,
        )

    pauli_to_comp = pauli_to_comp_basis(
        nqubits,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    return pauli_to_comp @ pauli_op @ backend.conj(pauli_to_comp).T


def pauli_to_choi(
    pauli_op,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
):
    """Converts Pauli-Liouville representation of a quantum channel
    to its Choi representation :math:`\\Lambda`.

    Args:
        pauli_op (ndarray): superoperator in the Pauli-Liouville representation.
        normalize (bool, optional): If ``True`` assumes ``pauli_op`` is represented
            in the normalized Pauli basis. If ``False``, it assumes unnormalized
            Pauli basis. Defaults to ``False``.
        order (str, optional): If ``"row"``, returns Choi representation in
            row-vectorization. If ``"column"``, returns column-vectorized
            superoperator. Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements. Defaults to "IXYZ".
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Choi representation of the superoperator.
    """
    super_op = pauli_to_liouville(
        pauli_op, normalize, order, pauli_order, backend=backend
    )
    super_op = liouville_to_choi(super_op, order, backend=backend)

    return super_op


def pauli_to_kraus(
    pauli_op,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    precision_tol: Optional[float] = None,
    backend: Optional[Backend] = None,
):
    """Converts Pauli-Liouville representation of a quantum channel
    to its Kraus representation :math:`\\{K_{\\alpha}\\}_{\\alpha}`.

    Args:
        pauli_op (ndarray): superoperator in the Pauli-Liouville representation.
        normalize (bool, optional): If ``True`` assumes ``pauli_op`` is represented
            in the normalized Pauli basis. If ``False``, it assumes unnormalized
            Pauli basis. Defaults to ``False``.
        order (str, optional): vectorization order of the Liouville and Choi
            intermediate steps. If ``"row"``, row-vectorizationcis used for both
            representations. If ``"column"``, column-vectorization is used.
            Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements. Defaults to "IXYZ".
        precision_tol (float, optional): Precision tolerance for eigenvalues
            found in the spectral decomposition problem. Any eigenvalue
            :math:`\\lambda <` ``precision_tol`` is set to 0 (zero).
            If ``None``, ``precision_tol`` defaults to
            ``qibo.config.PRECISION_TOL=1e-8``. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        (ndarray, ndarray): Kraus operators and their coefficients.
    """
    super_op = pauli_to_liouville(
        pauli_op, normalize, order, pauli_order, backend=backend
    )
    super_op = liouville_to_kraus(super_op, precision_tol, order, backend=backend)

    return super_op


def pauli_to_chi(
    pauli_op,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
):
    """Converts Pauli-Liouville representation of a quantum channel
    to its :math:`\\chi`-matrix representation.

    Args:
        pauli_op (ndarray): superoperator in the Pauli-Liouville representation.
        normalize (bool, optional): If ``True`` assumes ``pauli_op`` is represented
            in the normalized Pauli basis. If ``False``, it assumes unnormalized
            Pauli basis. Defaults to ``False``.
        order (str, optional): If ``"row"``, returns Choi representation in
            row-vectorization. If ``"column"``, returns column-vectorized
            superoperator. Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements. Defaults to "IXYZ".
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Chi-matrix representation of the quantum channel.
    """
    super_op = pauli_to_liouville(
        pauli_op, normalize, order, pauli_order, backend=backend
    )
    super_op = liouville_to_chi(
        super_op, normalize, order, pauli_order, backend=backend
    )

    return super_op


def pauli_to_stinespring(
    pauli_op,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    precision_tol: Optional[float] = None,
    validate_cp: bool = True,
    nqubits: Optional[int] = None,
    initial_state_env=None,
    backend: Optional[Backend] = None,
):
    """Converts Pauli-Liouville representation :math:`\\mathcal{E}_{P}` of quantum channel
    to its Stinespring representation :math:`U_{0}`.
    It uses the Liouville representation :math:`\\mathcal{E}` as intermediate step.

    Args:
        pauli_op (ndarray): Pauli-Liouville representation of a quantum channel.
        normalize (bool, optional): If ``True`` assumes ``pauli_op`` is represented
            in the normalized Pauli basis. If ``False``, it assumes unnormalized
            Pauli basis. Defaults to ``False``.
        order (str, optional): If ``"row"``, returns Liouville representation in
            row-vectorization. If ``"column"``, returns column-vectorized
            superoperator. If ``"system"``, superoperator will be in
            block-vectorization. Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements. Defaults to "IXYZ".
        precision_tol (float, optional): Precision tolerance for eigenvalues
            found in the spectral decomposition problem. Any eigenvalue
            :math:`\\lambda <` ``precision_tol`` is set to 0 (zero).
            If ``None``, ``precision_tol`` defaults to
            ``qibo.config.PRECISION_TOL=1e-8``. Defaults to ``None``.
        validate_cp (bool, optional): If ``True``, checks if ``choi_super_op``
            is a completely positive map. If ``False``, it assumes that
            ``choi_super_op`` is completely positive (and Hermitian).
            Defaults to ``True``.
        nqubits (int, optional): total number of qubits in the system that is
            interacting with the environment. Must be equal or greater than
            the number of qubits ``kraus_ops`` acts on. If ``None``,
            defaults to the number of qubits in ``kraus_ops``.
            Defauts to ``None``.
        initial_state_env (ndarray, optional): statevector representing the
            initial state of the enviroment. If ``None``, it assumes the
            environment in its ground state. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Stinestring representation of quantum channel.
    """
    super_op = pauli_to_liouville(
        pauli_op,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    stinespring = liouville_to_stinespring(
        super_op,
        order=order,
        precision_tol=precision_tol,
        validate_cp=validate_cp,
        nqubits=nqubits,
        initial_state_env=initial_state_env,
        backend=backend,
    )

    return stinespring


def chi_to_choi(
    chi_matrix,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
):
    """Convert the :math:`\\chi`-matrix representation of a quantum channel
    to its Choi representation :math:`\\Lambda`.

    .. math::
        \\Lambda = \\text{pauli_to_liouville}(\\chi)

    Args:
        chi_matrix: Chi-matrix representation of quantum channel.
        normalize (bool, optional): If ``True`` assumes the normalized
            Pauli basis. If ``False``, it assumes unnormalized
            Pauli basis. Defaults to ``False``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements. Defaults to "IXYZ".
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Choi representation of quantum channel.
    """
    choi_super_op = pauli_to_liouville(
        chi_matrix,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    return choi_super_op


def chi_to_liouville(
    chi_matrix,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
):
    """Converts the :math:`\\chi`-matrix representation of a quantum channel
    to its Liouville representation :math:`\\mathcal{E}`.

    .. math::
        \\mathcal{E} = \\text{pauli_to_liouville}(\\text{choi_to_liouville}(\\chi))

    Args:
        chi_matrix: Chi-matrix representation of quantum channel.
        normalize (bool, optional): If ``True`` assumes the normalized
            Pauli basis. If ``False``, it assumes unnormalized
            Pauli basis. Defaults to ``False``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements. Defaults to "IXYZ".
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Liouville representation of quantum channel.
    """
    choi_super_op = pauli_to_liouville(
        chi_matrix,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )
    super_op = choi_to_liouville(choi_super_op, order=order, backend=backend)

    return super_op


def chi_to_pauli(
    chi_matrix,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
):
    """Convert :math:`\\chi`-matrix representation of a quantum channel
    to its Pauli-Liouville representation :math:`\\mathcal{E}_P`.

    .. math::
        \\mathcal{E}_P = \\text{choi_to_pauli}(\\text{chi_to_choi}(\\chi))

    Args:
        chi_matrix: Chi-matrix representation of quantum channel.
        normalize (bool, optional): If ``True`` assumes the normalized
            Pauli basis. If ``False``, it assumes unnormalized
            Pauli basis. Defaults to ``False``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements. Defaults to "IXYZ".
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: superoperator in the Pauli-Liouville representation.
    """
    choi_super_op = pauli_to_liouville(
        chi_matrix,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )
    super_op = choi_to_pauli(
        choi_super_op,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    return super_op


def chi_to_kraus(
    chi_matrix,
    normalize: bool = False,
    precision_tol: Optional[float] = None,
    order: str = "row",
    pauli_order: str = "IXYZ",
    validate_cp: bool = True,
    backend: Optional[Backend] = None,
):
    """Converts the :math:`\\chi`-matrix representation of a quantum channel
    to its Kraus representation :math:`\\{K_{\\alpha}\\}_{\\alpha}`.

    .. math::
        \\mathcal{E}_P = \\text{choi_to_kraus}(\\text{chi_to_choi}(\\chi))

    Args:
        chi_matrix: Chi-matrix representation of quantum channel.
        normalize (bool, optional): If ``True`` assumes the normalized
            Pauli basis. If ``False``, it assumes unnormalized
            Pauli basis. Defaults to ``False``.
        precision_tol (float, optional): Precision tolerance for eigenvalues
            found in the spectral decomposition problem. Any eigenvalue
            :math:`\\lambda <` ``precision_tol`` is set to 0 (zero).
            If ``None``, ``precision_tol`` defaults to
            ``qibo.config.PRECISION_TOL=1e-8``. Defaults to ``None``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements. Defaults to "IXYZ".
        validate_cp (bool, optional): If ``True``, checks if ``choi_super_op``
            is a completely positive map. If ``False``, it assumes that
            ``choi_super_op`` is completely positive (and Hermitian).
            Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        (ndarray, ndarray): Kraus operators and their coefficients.
    """
    choi_super_op = pauli_to_liouville(
        chi_matrix,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )
    kraus_ops, coefficients = choi_to_kraus(
        choi_super_op,
        precision_tol=precision_tol,
        order=order,
        validate_cp=validate_cp,
        backend=backend,
    )

    return kraus_ops, coefficients


def chi_to_stinespring(
    chi_matrix,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    precision_tol: Optional[float] = None,
    validate_cp: bool = True,
    nqubits: Optional[int] = None,
    initial_state_env=None,
    backend: Optional[Backend] = None,
):
    """Converts :math:`\\chi`-representation of quantum channel
    to its Stinespring representation :math:`U_{0}`.
    It uses the Choi representation :math:`\\Lambda` as intermediate step.

    Args:
        chi_matrix: Chi-matrix representation of quantum channel.
        normalize (bool, optional): If ``True`` assumes the normalized
            Pauli basis. If ``False``, it assumes unnormalized
            Pauli basis. Defaults to ``False``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements. Defaults to "IXYZ".
        precision_tol (float, optional): Precision tolerance for eigenvalues
            found in the spectral decomposition problem. Any eigenvalue
            :math:`\\lambda <` ``precision_tol`` is set to 0 (zero).
            If ``None``, ``precision_tol`` defaults to
            ``qibo.config.PRECISION_TOL=1e-8``. Defaults to ``None``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        validate_cp (bool, optional): If ``True``, checks if ``choi_super_op``
            is a completely positive map. If ``False``, it assumes that
            ``choi_super_op`` is completely positive (and Hermitian).
            Defaults to ``True``.
        nqubits (int, optional): total number of qubits in the system that is
            interacting with the environment. Must be equal or greater than
            the number of qubits ``kraus_ops`` acts on. If ``None``,
            defaults to the number of qubits in ``kraus_ops``.
            Defauts to ``None``.
        initial_state_env (ndarray, optional): statevector representing the
            initial state of the enviroment. If ``None``, it assumes the
            environment in its ground state. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Stinespring representation of quantum channel.
    """
    choi_super_op = chi_to_choi(
        chi_matrix,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    stinespring = choi_to_stinespring(
        choi_super_op,
        precision_tol=precision_tol,
        order=order,
        validate_cp=validate_cp,
        nqubits=nqubits,
        initial_state_env=initial_state_env,
        backend=backend,
    )

    return stinespring


def stinespring_to_choi(
    stinespring,
    dim_env: int,
    initial_state_env=None,
    nqubits: Optional[int] = None,
    order: str = "row",
    backend: Optional[Backend] = None,
):
    """Converts Stinespring representation :math:`U_{0}` of quantum channel
    to its Choi representation :math:`\\Lambda`.

    .. math::
        \\Lambda = \\text{kraus_to_choi}(\\text{stinespring_to_kraus}(U_{0}))

    Args:
        stinespring (ndarray): quantum channel in the Stinespring representation.
        dim_env (int): dimension of the Hilbert space of the environment.
        initial_state_env (ndarray, optional): statevector representing the
            initial state of the enviroment. If ``None``, it assumes the
            environment in its ground state. Defaults to ``None``.
        nqubits (int, optional): number of qubits in the system. Defaults to ``None``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Choi representation of quantum channel.
    """
    kraus_ops = stinespring_to_kraus(
        stinespring,
        dim_env,
        initial_state_env=initial_state_env,
        nqubits=nqubits,
        backend=backend,
    )

    if nqubits is None:
        nqubits = int(math.log2(kraus_ops[0].shape[0]))

    nqubits = [tuple(range(nqubits)) for _ in range(len(kraus_ops))]

    kraus_ops = list(zip(nqubits, kraus_ops))

    choi_super_op = kraus_to_choi(kraus_ops, order=order, backend=backend)

    return choi_super_op


def stinespring_to_liouville(
    stinespring,
    dim_env: int,
    initial_state_env=None,
    nqubits: Optional[int] = None,
    order: str = "row",
    backend: Optional[Backend] = None,
):
    """Converts Stinespring representation :math:`U_{0}` of quantum channel
    to its Liouville representation :math:`\\mathcal{E}` via Stinespring Dilation,
    i.e.

    .. math::
        \\mathcal{E} = \\text{kraus_to_liouville}(\\text{stinespring_to_kraus}(U_{0}))

    Args:
        stinespring (ndarray): quantum channel in the Stinespring representation.
        dim_env (int): dimension of the Hilbert space of the environment.
        initial_state_env (ndarray, optional): statevector representing the
            initial state of the enviroment. If ``None``, it assumes the
            environment in its ground state. Defaults to ``None``.
        nqubits (int, optional): number of qubits in the system. Defaults to ``None``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Liouville representation of quantum channel.
    """
    kraus_ops = stinespring_to_kraus(
        stinespring,
        dim_env,
        initial_state_env=initial_state_env,
        nqubits=nqubits,
        backend=backend,
    )

    if nqubits is None:
        nqubits = int(math.log2(kraus_ops[0].shape[0]))

    nqubits = [tuple(range(nqubits)) for _ in range(len(kraus_ops))]

    kraus_ops = list(zip(nqubits, kraus_ops))

    super_op = kraus_to_liouville(kraus_ops, order=order, backend=backend)

    return super_op


def stinespring_to_pauli(
    stinespring,
    dim_env: int,
    initial_state_env=None,
    nqubits: Optional[int] = None,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
):
    """Converts Stinespring representation :math:`U_{0}` of quantum channel
    to its Pauli-Liouville representation :math:`\\mathcal{E}_{P}` via
    Stinespring Dilation, i.e.

    .. math::
        \\mathcal{E}_{P} = \\text{kraus_to_pauli}(\\text{stinespring_to_kraus}(U_{0}))

    Args:
        stinespring (ndarray): quantum channel in the Stinespring representation.
        dim_env (int): dimension of the Hilbert space of the environment.
        initial_state_env (ndarray, optional): statevector representing the
            initial state of the enviroment. If ``None``, it assumes the
            environment in its ground state. Defaults to ``None``.
        nqubits (int, optional): number of qubits in the system. Defaults to ``None``.
        normalize (bool, optional): If ``True`` superoperator is returned
            in the normalized Pauli basis. If ``False``, it is returned
            in the unnormalized Pauli basis. Defaults to ``False``.
        order (str, optional): If ``"row"``, intermediate step for Choi
            representation is done in row-vectorization. If ``"column"``,
            step is done in column-vectorization. If ``"system"``,
            block-vectorization is performed. Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4
            single-qubit Pauli elements. Defaults to "IXYZ".
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Pauli-Liouville representation of quantum channel.
    """
    kraus_ops = stinespring_to_kraus(
        stinespring,
        dim_env,
        initial_state_env=initial_state_env,
        nqubits=nqubits,
        backend=backend,
    )

    if nqubits is None:
        nqubits = int(math.log2(kraus_ops[0].shape[0]))

    nqubits = [tuple(range(nqubits)) for _ in range(len(kraus_ops))]

    kraus_ops = list(zip(nqubits, kraus_ops))

    super_op_pauli = kraus_to_pauli(
        kraus_ops,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    return super_op_pauli


def stinespring_to_kraus(
    stinespring,
    dim_env: int,
    initial_state_env=None,
    nqubits: Optional[int] = None,
    backend: Optional[Backend] = None,
):
    """Converts the Stinespring representation :math:`U_{0}` of quantum channel
    to its Kraus representation :math:`\\{K_{\\alpha}\\}_{\\alpha}`, i.e.

    .. math::
        K_{\\alpha} := \\bra{\\alpha} \\, U_{0} \\, \\ket{v_{0}} \\, ,

    where :math:`\\ket{v_{0}}` is the initial state of the environment
    (``initial_state_env``), :math:`D` is the dimension of the environment's
    Hilbert space, and
    :math:`\\{\\ket{\\alpha} \\, : \\, \\alpha = 0, 1, \\cdots, D - 1 \\}`
    is an orthonormal basis for the environment's Hilbert space.
    Note that :math:`\\text{dim}(\\ket{\\alpha}) = \\text{dim}(\\ket{v_{0}}) = D`,
    while :math:`\\text{dim}(U) = 2^{n} \\, D`, where :math:`n` is `nqubits`.

    Args:
        stinespring (ndarray): quantum channel in the Stinespring representation.
        dim_env (int): dimension of the Hilbert space of the environment.
        initial_state_env (ndarray, optional): statevector representing the
            initial state of the enviroment. If ``None``, it assumes the
            environment in its ground state. Defaults to ``None``.
        nqubits (int, optional): number of qubits in the system. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Kraus operators.
    """
    backend = _check_backend(backend)

    if initial_state_env is not None and len(initial_state_env.shape) != 1:
        raise_error(ValueError, "initial_state_env must be a statevector.")

    dim_stinespring = stinespring.shape[0]

    if nqubits is None:
        nqubits = int(math.log2(dim_stinespring / dim_env))

    dim = 2**nqubits

    if dim * dim_env != dim_stinespring:
        raise_error(
            ValueError,
            "Dimensions do not match. dim(`stinespring`) must be equal to `dim_env` * 2**nqubits.",
        )

    if initial_state_env is None:
        initial_state_env = backend.zeros(dim_env, dtype=complex)
        initial_state_env[0] = 1.0
        initial_state_env = backend.cast(
            initial_state_env, dtype=initial_state_env.dtype
        )
    return backend.qinfo._stinespring_to_kraus(
        stinespring, initial_state_env, dim, dim_env
    )


def stinespring_to_chi(
    stinespring,
    dim_env: int,
    initial_state_env=None,
    nqubits: Optional[int] = None,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
):
    """Converts Stinespring representation :math:`U_{0}` of quantum channel
    to its :math:`\\chi`-matrix representation via Stinespring Dilation, i.e.

    .. math::
        \\chi = \\text{kraus_to_chi}(\\text{stinespring_to_kraus}(U_{0}))

    Args:
        stinespring (ndarray): quantum channel in the Stinespring representation.
        dim_env (int): dimension of the Hilbert space of the environment.
        initial_state_env (ndarray, optional): statevector representing the
            initial state of the enviroment. If ``None``, it assumes the
            environment in its ground state. Defaults to ``None``.
        nqubits (int, optional): number of qubits in the system. Defaults to ``None``.
        normalize (bool, optional): If ``True`` assumes the normalized
            Pauli basis. If ``False``, it assumes unnormalized
            Pauli basis. Defaults to ``False``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements in the Pauli basis. Defaults to "IXYZ".
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: :math:`\\chi`-representation of quantum channel.
    """
    kraus_ops = stinespring_to_kraus(
        stinespring,
        dim_env,
        initial_state_env=initial_state_env,
        nqubits=nqubits,
        backend=backend,
    )

    if nqubits is None:
        nqubits = int(math.log2(kraus_ops[0].shape[0]))

    nqubits = [tuple(range(nqubits)) for _ in range(len(kraus_ops))]

    kraus_ops = list(zip(nqubits, kraus_ops))

    chi_super_op = kraus_to_chi(
        kraus_ops,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    return chi_super_op


def kraus_to_unitaries(
    kraus_ops,
    order: str = "row",
    precision_tol: Optional[float] = None,
    backend: Optional[Backend] = None,
):
    """Tries to convert Kraus operators into a probabilistc sum of unitaries.

    Given a set of Kraus operators :math:`\\{K_{\\alpha}\\}_{\\alpha}`,
    returns an ensemble :math:`\\{U_{\\alpha}, p_{\\alpha}\\}` that defines
    an :class:`qibo.gates.channels.UnitaryChannel` that approximates the original
    channel up to a precision tolerance in Frobenius norm.

    Args:
        kraus_ops (list): List of Kraus operators as pairs ``(qubits, Ak)``
            where ``qubits`` refers the qubit ids that :math:`A_k` acts on
            and :math:`A_k` is the corresponding matrix as a ``ArrayLike``.
        order (str, optional): _description_. Defaults to "row".
        precision_tol (float, optional): Precision tolerance for the minimization
            of the Frobenius norm :math:`\\| \\mathcal{E}_{K} - \\mathcal{E}_{U} \\|_{F}`,
            where :math:`\\mathcal{E}_{K}` is the Liouville representation of the Kraus
            channel :math:`\\{K_{\\alpha}\\}_{\\alpha}`, and :math:`\\mathcal{E}_{U}`
            is the Liouville representaton of the :class:`qibo.gates.channels.UnitaryChannel`
            that best approximates the original channel. If ``None``, ``precision_tol``
            defaults to ``1e-7``. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        (ndarray, ndarray): Unitary operators and their associated probabilities.
    """

    if precision_tol is None:
        precision_tol = 10 * PRECISION_TOL
    else:
        if not isinstance(precision_tol, float):
            raise_error(
                TypeError,
                f"precision_tol must be type float, but it is type {type(precision_tol)}.",
            )
        if precision_tol < 0.0:
            raise_error(ValueError, "precision_tol must be non-negative.")

    backend = _check_backend(backend)

    target_qubits = [q for q, _ in kraus_ops]
    nqubits = 1 + backend.max(target_qubits)
    dim = 2**nqubits

    target = kraus_to_liouville(kraus_ops, order=order, backend=backend)

    # QR decomposition
    unitaries = []
    for _, kraus in kraus_ops:
        Q, _ = np.linalg.qr(kraus)
        unitaries.append(Q)

    # unitaries in Liouville representation
    unitaries_liouville = _individual_kraus_to_liouville(
        list(zip(target_qubits, unitaries)), backend=backend
    )

    # function to minimize
    def function(x0, operators):
        operator = (1 - backend.sum(x0)) * backend.identity(dim**2, dtype=complex)
        operator = backend.cast(operator, dtype=operator.dtype)
        for prob, oper in zip(x0, operators):
            operator = operator + prob * oper

        return float(backend.matrix_norm(target - operator, order=2))

    # initial parameters as flat distribution
    x0 = [1.0 / (len(kraus_ops) + 1)] * len(kraus_ops)

    # final parameters
    probabilities = minimize(
        function,
        x0,
        args=(unitaries_liouville),
        options={"return_all": True},
    )["x"]

    final_norm = function(probabilities, unitaries_liouville)
    if final_norm > precision_tol:
        warnings.warn(
            f"precision in Frobenius norm of {final_norm} is greater then set "
            + f"precision_tol of {precision_tol}.",
            Warning,
        )

    return unitaries, probabilities


def _check_pauli_transform_method(method: Optional[str]) -> str:
    """Validate ``method`` for Pauli-basis conversions."""
    if method is not None and method not in ("dense", "fht", "standard"):
        raise_error(
            ValueError,
            f"``method`` must be either None, 'fht', 'standard', or 'dense', but it is {method}.",
        )

    if method is None or method == "fht":
        return "fht"

    return "standard"


def _check_pauli_superoperator_shape(super_op: ArrayLike, name: str) -> Tuple[int, int]:
    """Validate the shape of a Pauli or Liouville superoperator."""
    dim = math.sqrt(len(super_op))
    nqubits = math.log2(dim)

    if super_op.shape[0] != super_op.shape[1] or dim % 1 != 0 or nqubits % 1 != 0:
        raise_error(
            ValueError,
            f"{name} must be of shape (4^n, 4^n), but it is {super_op.shape}",
        )

    return int(dim), int(nqubits)


def _fast_walsh_hadamard_transform(
    array: ArrayLike, axis: int = -1, backend: Optional[Backend] = None
) -> ArrayLike:
    """Apply an unnormalized Fast Walsh-Hadamard transform along ``axis``."""
    backend = _check_backend(backend)

    axis = axis % len(array.shape)
    # array = backend.cast(array, dtype=array.dtype, copy=True)
    array = backend.cast(array, dtype=array.dtype)
    array = backend.swapaxes(array, axis, -1)

    dim = array.shape[-1]
    if dim & (dim - 1):  # pragma: no cover
        raise_error(
            ValueError, "Walsh-Hadamard transform dimension must be a power of 2."
        )

    block = 1
    while block < dim:
        shape = array.shape[:-1] + (dim // (2 * block), 2, block)
        array = backend.reshape(array, shape)
        even = array[..., 0, :]
        odd = array[..., 1, :]
        array = backend.concatenate(
            (
                backend.expand_dims(even + odd, -2),
                backend.expand_dims(even - odd, -2),
            ),
            axis=-2,
        )
        array = backend.reshape(array, array.shape[:-3] + (dim,))
        block *= 2

    return backend.swapaxes(array, axis, -1)


def _slice_axis(array: ArrayLike, axis: int, index: int) -> ArrayLike:
    """Return ``array`` sliced at ``index`` along ``axis``."""
    slices = [slice(None)] * len(array.shape)
    slices[axis] = index

    return array[tuple(slices)]


def _reorder_axis(array: ArrayLike, axis: int, permutation: Union[List[int], Tuple[int, ...], ArrayLike], backend: Optional[Backend] = None) -> ArrayLike:
    """Reorder one axis using only scalar slicing and concatenation."""
    backend = _check_backend(backend)

    axis = axis % len(array.shape)
    reordered = []
    for index in permutation:
        reordered.append(backend.expand_dims(_slice_axis(array, axis, index), axis))

    return backend.concatenate(reordered, axis=axis)


def _xor_pair_axis(array: ArrayLike, axis: int, backend: Optional[Backend] = None) -> ArrayLike:
    r"""Apply ``(r, q) -> (r \oplus q, q)`` to two adjacent binary axes."""
    backend = _check_backend(backend)

    axis = axis % len(array.shape)
    next_axis = axis + 1

    row_0_col_0 = _slice_axis(_slice_axis(array, axis, 0), next_axis - 1, 0)
    row_0_col_1 = _slice_axis(_slice_axis(array, axis, 0), next_axis - 1, 1)
    row_1_col_0 = _slice_axis(_slice_axis(array, axis, 1), next_axis - 1, 0)
    row_1_col_1 = _slice_axis(_slice_axis(array, axis, 1), next_axis - 1, 1)

    row_0 = backend.concatenate(
        (
            backend.expand_dims(row_0_col_0, next_axis - 1),
            backend.expand_dims(row_1_col_1, next_axis - 1),
        ),
        axis=next_axis - 1,
    )
    row_1 = backend.concatenate(
        (
            backend.expand_dims(row_1_col_0, next_axis - 1),
            backend.expand_dims(row_0_col_1, next_axis - 1),
        ),
        axis=next_axis - 1,
    )

    return backend.concatenate(
        (backend.expand_dims(row_0, axis), backend.expand_dims(row_1, axis)),
        axis=axis,
    )


def _xor_transform(array: ArrayLike, backend: Optional[Backend] = None) -> ArrayLike:
    """Apply the self-inverse XOR permutation along the last two axes."""
    backend = _check_backend(backend)

    dim = array.shape[-1]
    nqubits = int(math.log2(dim))
    batch_shape = array.shape[:-2]

    array = backend.reshape(array, batch_shape + (2,) * (2 * nqubits))
    offset = len(batch_shape)
    axes = tuple(range(offset)) + tuple(
        offset + index for pair in range(nqubits) for index in (pair, nqubits + pair)
    )
    array = backend.transpose(array, axes)

    for qubit in range(nqubits):
        array = _xor_pair_axis(array, offset + 2 * qubit, backend=backend)

    inverse_axes = [0] * len(axes)
    for index, axis in enumerate(axes):
        inverse_axes[axis] = index
    array = backend.transpose(array, tuple(inverse_axes))

    return backend.reshape(array, batch_shape + (dim, dim))


def _phase_matrix(dim: int, sign: int = -1, backend: Optional[Backend] = None) -> ArrayLike:
    """Return ``(sign * i) ** |r & s|`` for all pairs ``(r, s)``."""
    backend = _check_backend(backend)

    phase = backend.cast([[1.0, 1.0], [1.0, sign * 1.0j]], dtype=backend.complex128)
    phase = reduce(backend.kron, [phase] * int(math.log2(dim)))

    return phase


def _check_pauli_order(pauli_order: str) -> None:
    """Validate the single-qubit Pauli order."""
    # if set(pauli_order) != {"I", "X", "Y", "Z"}:
    if len(pauli_order) != 4 or set(pauli_order) != {"I", "X", "Y", "Z"}:
        raise_error(
            ValueError,
            f"pauli_order has to contain 4 symbols: I, X, Y, Z. Got {pauli_order} instead.",
        )


def _symplectic_coefficients_to_pauli_order(
    coefficients: ArrayLike,
    nqubits: int,
    dim: int,
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
) -> ArrayLike:
    """Vectorize coefficients ``alpha[r, s]`` according to ``pauli_order``."""
    backend = _check_backend(backend)
    _check_pauli_order(pauli_order)

    batch_shape = coefficients.shape[:-2]
    coefficients = backend.reshape(coefficients, batch_shape + (2,) * (2 * nqubits))
    offset = len(batch_shape)
    axes = tuple(range(offset)) + tuple(
        offset + index for pair in range(nqubits) for index in (pair, nqubits + pair)
    )
    coefficients = backend.transpose(coefficients, axes)
    coefficients = backend.reshape(coefficients, batch_shape + (4,) * nqubits)

    canonical_order = "IZXY"
    permutation = tuple(canonical_order.index(pauli) for pauli in pauli_order)
    for qubit in range(nqubits):
        coefficients = _reorder_axis(
            coefficients, offset + qubit, permutation, backend=backend
        )

    return backend.reshape(coefficients, batch_shape + (dim**2,))


def _pauli_order_to_symplectic_coefficients(
    vectors: ArrayLike,
    nqubits: int,
    dim: int,
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
) -> ArrayLike:
    """Convert Pauli-ordered vectors to coefficients ``alpha[r, s]``."""
    backend = _check_backend(backend)
    _check_pauli_order(pauli_order)

    batch_shape = vectors.shape[:-1]
    coefficients = backend.reshape(vectors, batch_shape + (4,) * nqubits)

    offset = len(batch_shape)
    canonical_order = "IZXY"
    permutation = tuple(pauli_order.index(pauli) for pauli in canonical_order)
    for qubit in range(nqubits):
        coefficients = _reorder_axis(
            coefficients, offset + qubit, permutation, backend=backend
        )

    coefficients = backend.reshape(coefficients, batch_shape + (2,) * (2 * nqubits))
    axes = (
        tuple(range(offset))
        + tuple(offset + 2 * qubit for qubit in range(nqubits))
        + tuple(offset + 2 * qubit + 1 for qubit in range(nqubits))
    )
    coefficients = backend.transpose(coefficients, axes)

    return backend.reshape(coefficients, batch_shape + (dim, dim))


def _operator_to_pauli_coefficients_fht(
    operators: ArrayLike, dim: int, backend: Optional[Backend] = None
) -> ArrayLike:
    """Return Pauli decomposition coefficients for a batch of operators."""
    backend = _check_backend(backend)

    coefficients = _xor_transform(operators, backend=backend)
    coefficients = _fast_walsh_hadamard_transform(
        coefficients, axis=-1, backend=backend
    )
    coefficients = coefficients * _phase_matrix(dim, sign=-1, backend=backend) / dim

    return coefficients


def _pauli_coefficients_to_operator_fht(
    coefficients: ArrayLike, dim: int, backend: Optional[Backend] = None
) -> ArrayLike:
    """Reconstruct a batch of operators from Pauli decomposition coefficients."""
    backend = _check_backend(backend)

    operators = coefficients * _phase_matrix(dim, sign=1, backend=backend) * dim
    operators = (
        _fast_walsh_hadamard_transform(operators, axis=-1, backend=backend) / dim
    )
    operators = _xor_transform(operators, backend=backend)

    return operators


def _operator_to_pauli_vectors_fht(
    operators: ArrayLike,
    nqubits: int,
    dim: int,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
) -> ArrayLike:
    """Convert a batch of operators to vectorized Pauli-basis coordinates."""
    backend = _check_backend(backend)

    coefficients = _operator_to_pauli_coefficients_fht(operators, dim, backend=backend)
    normalization = _pauli_basis_normalization(nqubits) if normalize else 1.0
    coefficients = _symplectic_coefficients_to_pauli_order(
        coefficients,
        nqubits=nqubits,
        dim=dim,
        pauli_order=pauli_order,
        backend=backend,
    )

    return coefficients * dim / normalization


def _pauli_vectors_to_operator_fht(
    vectors: ArrayLike,
    nqubits: int,
    dim: int,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
) -> ArrayLike:
    """Convert vectorized Pauli-basis coordinates to computational operators."""
    backend = _check_backend(backend)

    normalization = _pauli_basis_normalization(nqubits) if normalize else 1.0
    coefficients = _pauli_order_to_symplectic_coefficients(
        vectors / normalization,
        nqubits=nqubits,
        dim=dim,
        pauli_order=pauli_order,
        backend=backend,
    )

    return _pauli_coefficients_to_operator_fht(coefficients, dim, backend=backend)


def _to_pauli_liouville_fht(
    channel: ArrayLike,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
) -> ArrayLike:
    """Converts ``channel`` to Pauli-Liouville representation using FHT."""
    backend = _check_backend(backend)

    super_op = to_liouville(channel, order=order, backend=backend)
    dim, nqubits = _check_pauli_superoperator_shape(super_op, "super_op")

    return _liouville_to_pauli_fht(
        super_op,
        nqubits=nqubits,
        dim=dim,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )


def _liouville_to_pauli_fht(
    super_op: ArrayLike,
    nqubits: int,
    dim: int,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
) -> ArrayLike:
    """Converts Liouville representation to Pauli-Liouville using FHT."""
    backend = _check_backend(backend)

    columns = unvectorization(backend.transpose(super_op), order=order, backend=backend)
    columns = _operator_to_pauli_vectors_fht(
        columns,
        nqubits=nqubits,
        dim=dim,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )
    super_op = backend.transpose(columns)

    rows = _operator_to_pauli_vectors_fht(
        unvectorization(backend.conj(super_op), order=order, backend=backend),
        nqubits=nqubits,
        dim=dim,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    return backend.conj(rows)


def _pauli_to_liouville_fht(
    pauli_op: ArrayLike,
    nqubits: int,
    dim: int,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
) -> ArrayLike:
    """Converts Pauli-Liouville representation to Liouville using FHT."""
    backend = _check_backend(backend)

    columns = _pauli_vectors_to_operator_fht(
        backend.transpose(pauli_op),
        nqubits=nqubits,
        dim=dim,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )
    columns = vectorization(columns, order=order, backend=backend)
    super_op = backend.transpose(columns)

    rows = _pauli_vectors_to_operator_fht(
        backend.conj(super_op),
        nqubits=nqubits,
        dim=dim,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )
    rows = vectorization(rows, order=order, backend=backend)

    return backend.conj(rows)


def _reshuffling(
    super_op: ArrayLike, order: str = "row", backend: Optional[Backend] = None
) -> ArrayLike:
    """Reshuffling operation used to convert Lioville representation
    of quantum channels to their Choi representation (and vice-versa).

    For an operator :math:`A` with dimensions :math:`d^{2} \\times d^{2}`,
    the reshuffling operation consists of reshaping :math:`A` as a
    4-dimensional tensor, swapping two axes, and reshaping back to a
    :math:`d^{2} \\times d^{2}` matrix.

    If ``order="row"``, then:

    .. math::
        A_{\\alpha\\beta, \\, \\gamma\\delta} \\mapsto A_{\\alpha, \\, \\beta, \\,
            \\gamma, \\, \\delta} \\mapsto A_{\\alpha, \\, \\gamma, \\, \\beta, \\, \\delta}
            \\mapsto A_{\\alpha\\gamma, \\, \\beta\\delta}

    If ``order="column"``, then:

    .. math::
        A_{\\alpha\\beta, \\, \\gamma\\delta} \\mapsto A_{\\alpha, \\, \\beta, \\,
            \\gamma, \\, \\delta} \\mapsto A_{\\delta, \\, \\beta, \\, \\gamma, \\, \\alpha}
            \\mapsto A_{\\delta\\beta, \\, \\gamma\\alpha}

    Args:
        super_op (ndarray): Liouville (Choi) representation of a
            quantum channel.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Choi (Liouville) representation of the quantum channel.
    """
    if order not in ("row", "column"):
        raise_error(
            ValueError,
            f"Unsupported {order} order, please pick one in ('row', 'column').",
        )
    backend = _check_backend(backend)

    dim = math.sqrt(super_op.shape[0])

    if (
        super_op.shape[0] != super_op.shape[1]
        or dim % 1 != 0
        or math.log2(int(dim)) % 1 != 0
    ):
        raise_error(
            ValueError,
            f"`super_op` must be of shape (4^n, 4^n), but it is {super_op.shape}",
        )

    axes = [1, 2] if order == "row" else [0, 3]
    return backend.qinfo._reshuffling(super_op, *axes)


def _set_gate_and_target_qubits(kraus_ops):  # pragma: no cover
    """Returns Kraus operators as a set of gates acting on
    their respective ``target qubits``.

    Args:
        kraus_ops (list): List of Kraus operators as pairs ``(qubits, Ak)``
            where ``qubits`` refers the qubit ids that :math:`A_k` acts on
            and :math:`A_k` is the corresponding matrix as a ``ArrayLike``.

    Returns:
        (tuple, tuple): gates and their respective target qubits.
    """
    if isinstance(kraus_ops[0], Gate):
        gates = tuple(kraus_ops)
        target_qubits = tuple(
            sorted({q for gate in kraus_ops for q in gate.target_qubits})
        )
    else:
        gates, qubitset = [], set()
        for qubits, matrix in kraus_ops:
            rank = 2 ** len(qubits)
            shape = tuple(matrix.shape)
            if shape != (rank, rank):
                raise_error(
                    ValueError,
                    f"Invalid Kraus operator shape {shape} for "
                    + f"acting on {len(qubits)} qubits.",
                )
            qubitset.update(qubits)
            gates.append(Unitary(matrix, *list(qubits)))
        gates = tuple(gates)
        target_qubits = tuple(sorted(qubitset))

    return gates, target_qubits


def _individual_kraus_to_liouville(
    kraus_ops, order: str = "row", backend: Optional[Backend] = None
):  # pragma: no cover
    """Auxiliary, modified version of :func:`qibo.quantum_info.kraus_to_choi`
    to be used in :func:`qibo.quantum_info.kraus_to_unitaries`. In principle,
    this should be not be accessible to users.
    """
    backend = _check_backend(backend)

    gates, target_qubits = _set_gate_and_target_qubits(kraus_ops)
    nqubits = 1 + max(target_qubits)

    super_ops = []
    for gate in gates:
        kraus_op = FusedGate(*range(nqubits))
        kraus_op.append(gate)
        kraus_op = kraus_op.matrix(backend)
        kraus_op = vectorization(kraus_op, order=order, backend=backend)
        kraus_op = backend.outer(kraus_op, backend.conj(kraus_op))
        super_ops.append(choi_to_liouville(kraus_op, order=order, backend=backend))

    return super_ops
