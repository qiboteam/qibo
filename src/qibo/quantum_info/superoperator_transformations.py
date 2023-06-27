"""Module with the most commom superoperator transformations."""

import warnings
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from qibo.backends import GlobalBackend
from qibo.config import PRECISION_TOL, raise_error
from qibo.gates.abstract import Gate
from qibo.gates.gates import Unitary
from qibo.gates.special import FusedGate


def vectorization(state, order: str = "row", backend=None):
    """Returns state :math:`\\rho` in its Liouville
    representation :math:`|\\rho\\rangle\\rangle`.

    If ``order="row"``, then:

    .. math::
        |\\rho\\rangle\\rangle = \\sum_{k, l} \\, \\rho_{kl} \\, \\ket{k} \\otimes \\ket{l}

    If ``order="column"``, then:

    .. math::
        |\\rho\\rangle\\rangle = \\sum_{k, l} \\, \\rho_{kl} \\, \\ket{l} \\otimes \\ket{k}

    Args:
        state: state vector or density matrix.
        order (str, optional): If ``"row"``, vectorization is performed
            row-wise. If ``"column"``, vectorization is performed
            column-wise. If ``"system"``, a block-vectorization is
            performed. Defaults to ``"row"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ndarray: Liouville representation of ``state``.
    """
    if (
        (len(state.shape) >= 3)
        or (len(state) == 0)
        or (len(state.shape) == 2 and state.shape[0] != state.shape[1])
    ):
        raise_error(
            TypeError,
            f"Object must have dims either (k,) or (k,k), but have dims {state.shape}.",
        )

    if not isinstance(order, str):
        raise_error(
            TypeError, f"order must be type str, but it is type {type(order)} instead."
        )
    else:
        if order not in ["row", "column", "system"]:
            raise_error(
                ValueError,
                f"order must be either 'row' or 'column' or 'system', but it is {order}.",
            )

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    if len(state.shape) == 1:
        state = np.outer(state, np.conj(state))

    if order == "row":
        state = np.reshape(state, (1, -1), order="C")[0]
    elif order == "column":
        state = np.reshape(state, (1, -1), order="F")[0]
    else:
        dim = len(state)
        nqubits = int(np.log2(dim))

        new_axis = []
        for qubit in range(nqubits):
            new_axis += [qubit + nqubits, qubit]

        state = np.reshape(state, [2] * 2 * nqubits)
        state = np.transpose(state, axes=new_axis)
        state = np.reshape(state, -1)

    state = backend.cast(state, dtype=state.dtype)

    return state


def unvectorization(state, order: str = "row", backend=None):
    """Returns state :math:`\\rho` from its Liouville
    representation :math:`|\\rho\\rangle\\rangle`. This operation is
    the inverse function of :func:`vectorization`, i.e.

    .. math::
        \\begin{align}
            \\rho &= \\text{unvectorization}(|\\rho\\rangle\\rangle) \\nonumber \\\\
            &= \\text{unvectorization}(\\text{vectorization}(\\rho)) \\nonumber
        \\end{align}

    Args:
        state: quantum state in Liouville representation.
        order (str, optional): If ``"row"``, unvectorization is performed
            row-wise. If ``"column"``, unvectorization is performed
            column-wise. If ``"system"``, system-wise vectorization is
            performed. Defaults to ``"row"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ndarray: Density matrix of ``state``.
    """

    if len(state.shape) != 1:
        raise_error(
            TypeError,
            f"Object must have dims (k,), but have dims {state.shape}.",
        )

    if not isinstance(order, str):
        raise_error(
            TypeError, f"order must be type str, but it is type {type(order)} instead."
        )
    else:
        if order not in ["row", "column", "system"]:
            raise_error(
                ValueError,
                f"order must be either 'row' or 'column' or 'system', but it is {order}.",
            )

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    dim = int(np.sqrt(len(state)))

    if order in ["row", "column"]:
        order = "C" if order == "row" else "F"
        state = np.reshape(state, (dim, dim), order=order)
    else:
        nqubits = int(np.log2(dim))
        axes_old = list(np.arange(0, 2 * nqubits))
        state = np.reshape(state, [2] * 2 * nqubits)
        state = np.transpose(state, axes=axes_old[1::2] + axes_old[0::2])
        state = np.reshape(state, [2**nqubits] * 2)

    state = backend.cast(state, dtype=state.dtype)

    return state


def to_choi(channel, order: str = "row", backend=None):
    """Converts quantum ``channel`` :math:`U` to its Choi representation :math:`\\Lambda`.

    .. math::
        \\Lambda = | U \\rangle\\rangle \\langle\\langle U | \\, ,

    where :math:`| \\cdot \\rangle\\rangle` is the :func:`qibo.quantum_info.vectorization`
    operation.

    Args:
        channel (ndarray): quantum channel.
        order (str, optional): If ``"row"``, vectorization is performed
            row-wise. If ``"column"``, vectorization is performed
            column-wise. If ``"system"``, a block-vectorization is
            performed. Default is ``"row"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ndarray: quantum channel in its Choi representation.
    """
    channel = vectorization(channel, order=order, backend=backend)
    channel = np.outer(channel, np.conj(channel))

    return channel


def to_liouville(channel, order: str = "row", backend=None):
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ndarray: quantum channel in its Liouville representation.
    """
    channel = to_choi(channel, order=order, backend=backend)
    channel = _reshuffling(channel, order=order, backend=backend)

    return channel


def to_pauli_liouville(
    channel,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend=None,
):
    """Converts quantum ``channel`` :math:`U` to its Pauli-Liouville
    representation :math:`\\mathcal{E}`. It uses the Liouville representation
    as an intermediate step.

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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ndarray: quantum channel in its Pauli-Liouville representation.
    """
    from qibo.quantum_info.basis import comp_basis_to_pauli

    nqubits = int(np.log2(channel.shape[0]))

    channel = to_liouville(channel, order=order, backend=backend)

    unitary = comp_basis_to_pauli(
        nqubits, normalize, pauli_order=pauli_order, backend=backend
    )

    channel = unitary @ channel @ np.transpose(np.conj(unitary))

    return channel


def to_chi(
    channel,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend=None,
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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


def choi_to_liouville(choi_super_op, order: str = "row", backend=None):
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ndarray: Liouville representation of quantum channel.
    """

    return _reshuffling(choi_super_op, order=order, backend=backend)


def choi_to_pauli(
    choi_super_op,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend=None,
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
    backend=None,
):
    """Converts Choi representation :math:`\\Lambda` of a quantum channel :math:`\\mathcal{E}`
    into Kraus operators :math:`\\{ K_{\\alpha} \\}_{\\alpha}`.

    If :math:`\\mathcal{E}` is a completely positive (CP) map, then

    .. math::
        \\Lambda = \\sum_{\\alpha} \\, \\lambda_{\\alpha}^{2} \\,
            |\\tilde{K}_{\\alpha}\\rangle\\rangle \\langle\\langle \\tilde{K}_{\\alpha}| \\, .

    This is the spectral decomposition of :math:`\\Lambda`, Hence, the set
    :math:`\\{\\lambda_{\\alpha}, \\, \\tilde{K}_{\\alpha}\\}_{\\alpha}`
    is found by diagonalization of :math:`\\Lambda`. The Kraus operators
    :math:`\\{K_{\\alpha}\\}_{\\alpha}` are defined as

    .. math::
        K_{\\alpha} = \\lambda_{\\alpha} \\,
            \\text{unvectorization}(|\\tilde{K}_{\\alpha}\\rangle\\rangle) \\, .

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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        tuple(ndarray, ndarray): The set
        :math:`\\{K_{\\alpha}, \\, \\lambda_{\\alpha} \\}_{\\alpha}`
        of Kraus operators representing the quantum channel and their respective coefficients.
        If map is non-CP, then function returns the set
        :math:`\\{ \\{K_{L}, \\, K_{R}\\}_{\\alpha}, \\, \\lambda_{\\alpha} \\}_{\\alpha}`,
        with the left- and right-generalized Kraus operators as well as the square root of
        their corresponding singular values.
    """

    if precision_tol is not None and not isinstance(precision_tol, float):
        raise_error(
            TypeError,
            f"precision_tol must be type float, but it is type {type(precision_tol)}",
        )

    if precision_tol is not None and precision_tol < 0:
        raise_error(
            ValueError,
            f"precision_tol must be a non-negative float, but it is {precision_tol}.",
        )

    if precision_tol is None:  # pragma: no cover
        precision_tol = PRECISION_TOL

    if not isinstance(validate_cp, bool):
        raise_error(
            TypeError,
            f"validate_cp must be type bool, but it is type {type(validate_cp)}.",
        )

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    if validate_cp:
        norm = backend.calculate_norm(
            choi_super_op - np.transpose(np.conj(choi_super_op))
        )
        if norm > PRECISION_TOL:
            non_cp = True
        else:
            # using eigh because, in this case, choi_super_op is
            # *already confirmed* to be Hermitian
            eigenvalues, eigenvectors = np.linalg.eigh(choi_super_op)
            eigenvectors = np.transpose(eigenvectors)

            non_cp = bool(any(eigenvalues < -PRECISION_TOL))
    else:
        non_cp = False
        # using eigh because, in this case, choi_super_op is
        # *assumed* to be Hermitian
        eigenvalues, eigenvectors = np.linalg.eigh(choi_super_op)
        eigenvectors = np.transpose(eigenvectors)

    if non_cp:
        warnings.warn("Input choi_super_op is a non-completely positive map.")

        # using singular value decomposition because choi_super_op is non-CP
        U, coefficients, V = np.linalg.svd(choi_super_op)
        U = np.transpose(U)
        coefficients = np.sqrt(coefficients)
        V = np.conj(V)

        kraus_left, kraus_right = [], []
        for coeff, eigenvector_left, eigenvector_right in zip(coefficients, U, V):
            kraus_left.append(
                coeff * unvectorization(eigenvector_left, order=order, backend=backend)
            )
            kraus_right.append(
                coeff * unvectorization(eigenvector_right, order=order, backend=backend)
            )
        kraus_ops = backend.cast([kraus_left, kraus_right])
    else:
        # when choi_super_op is CP
        kraus_ops, coefficients = [], []
        for eig, kraus in zip(eigenvalues, eigenvectors):
            if np.abs(eig) > precision_tol:
                eig = np.sqrt(eig)
                kraus_ops.append(
                    eig * unvectorization(kraus, order=order, backend=backend)
                )
                coefficients.append(eig)

    kraus_ops = backend.cast(kraus_ops)
    coefficients = backend.cast(coefficients)

    return kraus_ops, coefficients


def choi_to_chi(
    choi_super_op,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend=None,
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.
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
    backend=None,
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
        nqubits = int(np.log2(kraus_ops[0].shape[0]))

    nqubits_list = [tuple(range(nqubits)) for _ in range(len(kraus_ops))]

    kraus_ops = list(zip(nqubits_list, kraus_ops))

    stinespring = kraus_to_stinespring(
        kraus_ops, nqubits=nqubits, initial_state_env=initial_state_env, backend=backend
    )

    return stinespring


def kraus_to_choi(kraus_ops, order: str = "row", backend=None):
    """Converts Kraus representation :math:`\\{K_{\\alpha}\\}_{\\alpha}`
    of quantum channel to its Choi representation :math:`\\Lambda`.

    .. math::
        \\Lambda = \\sum_{\\alpha} \\, |K_{\\alpha}\\rangle\\rangle \\langle\\langle K_{\\alpha}|

    Args:
        kraus_ops (list): List of Kraus operators as pairs ``(qubits, Ak)``
            where ``qubits`` refers the qubit ids that :math:`A_k`  acts on
            and :math:`A_k` is the corresponding matrix as a ``np.ndarray``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ndarray: Choi representation of the Kraus channel.
    """
    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    gates, target_qubits = _set_gate_and_target_qubits(kraus_ops)
    nqubits = 1 + max(target_qubits)
    dim = 2**nqubits

    super_op = np.zeros((dim**2, dim**2), dtype=complex)
    super_op = backend.cast(super_op, dtype=super_op.dtype)
    for gate in gates:
        kraus_op = FusedGate(*range(nqubits))
        kraus_op.append(gate)
        kraus_op = kraus_op.asmatrix(backend)
        kraus_op = vectorization(kraus_op, order=order, backend=backend)
        super_op += np.outer(kraus_op, np.conj(kraus_op))
        del kraus_op

    return super_op


def kraus_to_liouville(kraus_ops, order: str = "row", backend=None):
    """Converts from Kraus representation :math:`\\{K_{\\alpha}\\}_{\\alpha}`
    of quantum channel to its Liouville representation :math:`\\mathcal{E}`.
    It uses the Choi representation as an intermediate step.

    .. math::
        \\mathcal{E} = \\text{choi_to_liouville}(\\text{kraus_to_choi}
            (\\{K_{\\alpha}\\}_{\\alpha}))

    Args:
        kraus_ops (list): List of Kraus operators as pairs ``(qubits, Ak)``
            where ``qubits`` refers the qubit ids that :math:`A_k` acts on
            and :math:`A_k` is the corresponding matrix as a ``np.ndarray``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Defaults to ``"row"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
    backend=None,
):
    """Converts Kraus representation :math:`\\{K_{\\alpha}\\}_{\\alpha}`
    of a quantum channel to its Pauli-Liouville representation.

    Args:
        kraus_ops (list): List of Kraus operators as pairs ``(qubits, Ak)``
            where ``qubits`` refers the qubit ids that :math:`A_k` acts on
            and :math:`A_k` is the corresponding matrix as a ``np.ndarray``.
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
    backend=None,
):
    """Convert Kraus representation :math:`\\{K_{\\alpha}\\}_{\\alpha}`
    of quantum channel to  its :math:`\\chi`-matrix representation.

    .. math::
        \\chi = \\sum_{\\alpha} \\, |c_{\\alpha}\\rangle\\rangle \\langle\\langle c_{\\alpha}|,

    where :math:`|c_{\\alpha}\\rangle\\rangle \\cong |K_{\\alpha}\\rangle\\rangle`
    in Pauli-Liouville basis.

    Args:
        kraus_ops (list): List of Kraus operators as pairs ``(qubits, Ak)``
            where ``qubits`` refers the qubit ids that :math:`A_k`  acts on
            and :math:`A_k` is the corresponding matrix as a ``np.ndarray``.
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ndarray: Chi-matrix representation of the Kraus channel.
    """
    from qibo.quantum_info.basis import comp_basis_to_pauli

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

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

    super_op = np.zeros((dim**2, dim**2), dtype=complex)
    super_op = backend.cast(super_op, dtype=super_op.dtype)
    for gate in gates:
        kraus_op = FusedGate(*range(nqubits))
        kraus_op.append(gate)
        kraus_op = kraus_op.asmatrix(backend)
        kraus_op = vectorization(kraus_op, order=order, backend=backend)
        kraus_op = comp_to_pauli @ kraus_op
        super_op += np.outer(kraus_op, np.conj(kraus_op))
        del kraus_op

    return super_op


def kraus_to_stinespring(
    kraus_ops, nqubits: Optional[int] = None, initial_state_env=None, backend=None
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
            and :math:`A_k` is the corresponding matrix as a ``np.ndarray``.
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ndarray: Stinespring representation (restricted unitary) of the Kraus channel.
    """
    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

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

    dim = 2**nqubits
    dim_env = len(kraus_ops)
    dim_stinespring = dim * dim_env

    if initial_state_env is None:
        initial_state_env = np.zeros(dim_env, dtype=complex)
        initial_state_env[0] = 1.0
        initial_state_env = backend.cast(
            initial_state_env, dtype=initial_state_env.dtype
        )

    # only utility is for outer product,
    # so np.conj here to only do it once
    initial_state_env = np.conj(initial_state_env)

    stinespring = np.zeros((dim_stinespring, dim_stinespring), dtype=complex)
    stinespring = backend.cast(stinespring, dtype=stinespring.dtype)
    for alpha, gate in enumerate(gates):
        vector_alpha = np.zeros(dim_env, dtype=complex)
        vector_alpha[alpha] = 1.0
        vector_alpha = backend.cast(vector_alpha, dtype=vector_alpha.dtype)
        kraus_op = FusedGate(*range(nqubits))
        kraus_op.append(gate)
        kraus_op = kraus_op.asmatrix(backend)
        kraus_op = backend.cast(kraus_op, dtype=kraus_op.dtype)
        stinespring += np.kron(
            kraus_op,
            np.outer(vector_alpha, initial_state_env),
        )
        del kraus_op, vector_alpha

    return stinespring


def liouville_to_choi(super_op, order: str = "row", backend=None):
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ndarray: Choi representation of quantum channel.
    """

    return _reshuffling(super_op, order=order, backend=backend)


def liouville_to_pauli(
    super_op,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend=None,
):
    """Converts Liouville representation :math:`\\mathcal{E}` of a
    quantum channel to its Pauli-Liouville representation.

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
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ndarray: superoperator in the Pauli-Liouville representation.
    """
    from qibo.quantum_info.basis import comp_basis_to_pauli

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    dim = int(np.sqrt(len(super_op)))
    nqubits = int(np.log2(dim))

    if (
        super_op.shape[0] != super_op.shape[1]
        or np.mod(dim, 1) != 0
        or np.mod(nqubits, 1) != 0
    ):
        raise_error(ValueError, "super_op must be of shape (4^n, 4^n)")

    comp_to_pauli = comp_basis_to_pauli(
        nqubits,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    return comp_to_pauli @ super_op @ np.conj(np.transpose(comp_to_pauli))


def liouville_to_kraus(
    super_op, precision_tol: Optional[float] = None, order: str = "row", backend=None
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
    backend=None,
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
    backend=None,
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
    backend=None,
):
    """Converts Pauli-Liouville representation of a quantum channel to its
    Liouville representation :math:`\\mathcal{E}`.

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
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ndarray: superoperator in the Liouville representation.
    """
    from qibo.quantum_info.basis import pauli_to_comp_basis

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    dim = int(np.sqrt(len(pauli_op)))
    nqubits = int(np.log2(dim))

    if (
        pauli_op.shape[0] != pauli_op.shape[1]
        or np.mod(dim, 1) != 0
        or np.mod(nqubits, 1) != 0
    ):
        raise_error(ValueError, "pauli_op must be of shape (4^n, 4^n)")

    pauli_to_comp = pauli_to_comp_basis(
        nqubits,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    return pauli_to_comp @ pauli_op @ np.conj(np.transpose(pauli_to_comp))


def pauli_to_choi(
    pauli_op,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend=None,
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
    backend=None,
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
    backend=None,
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
    backend=None,
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
    backend=None,
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
    backend=None,
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
    backend=None,
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
    backend=None,
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
    backend=None,
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
    backend=None,
):
    """Converts Stinespring representation :math:`U_{0}` of quantum channel
    to its Choi representation :math:`\\Lambda`.

    .. math::
        \\Lambda = \\text{kraus_to_choi}(\\text{stinespring_to_kraus}(U_{0}}))

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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
        nqubits = int(np.log2(kraus_ops[0].shape[0]))

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
    backend=None,
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
        nqubits = int(np.log2(kraus_ops[0].shape[0]))

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
    backend=None,
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
        nqubits = int(np.log2(kraus_ops[0].shape[0]))

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
    backend=None,
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ndarray: Kraus operators.
    """
    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    if isinstance(dim_env, int) is False:
        raise_error(
            TypeError, f"dim_env must be type int, but it is type {type(dim_env)}."
        )

    if dim_env <= 0:
        raise_error(ValueError, "dim_env must be a positive integer.")

    if initial_state_env is not None and len(initial_state_env.shape) != 1:
        raise_error(ValueError, "initial_state_env must be a statevector.")

    if nqubits is not None:
        if isinstance(nqubits, int) is False:
            raise_error(
                TypeError, f"nqubits must be type int, but it is type {type(nqubits)}."
            )
        if nqubits <= 0:
            raise_error(ValueError, "nqubits must be a positive integer.")

    dim_stinespring = stinespring.shape[0]

    if nqubits is None:
        nqubits = int(np.log2(dim_stinespring / dim_env))

    dim = 2**nqubits

    if dim * dim_env != dim_stinespring:
        raise_error(
            ValueError,
            "Dimensions do not match. dim(`stinespring`) must be equal to `dim_env` * 2**nqubits.",
        )

    if initial_state_env is None:
        initial_state_env = np.zeros(dim_env, dtype=complex)
        initial_state_env[0] = 1.0
        initial_state_env = backend.cast(
            initial_state_env, dtype=initial_state_env.dtype
        )

    stinespring = np.reshape(stinespring, (dim, dim_env, dim, dim_env))
    stinespring = np.swapaxes(stinespring, 1, 2)

    kraus_ops = []
    for alpha in range(dim_env):
        vector_alpha = np.zeros(dim_env, dtype=complex)
        vector_alpha[alpha] = 1.0
        vector_alpha = backend.cast(vector_alpha, dtype=vector_alpha.dtype)
        kraus = np.conj(vector_alpha) @ stinespring @ initial_state_env
        kraus_ops.append(kraus)

    return kraus_ops


def stinespring_to_chi(
    stinespring,
    dim_env: int,
    initial_state_env=None,
    nqubits: Optional[int] = None,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend=None,
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
        nqubits = int(np.log2(kraus_ops[0].shape[0]))

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
    kraus_ops, order: str = "row", precision_tol: Optional[float] = None, backend=None
):
    """Tries to convert Kraus operators into a probabilistc sum of unitaries.

    Given a set of Kraus operators :math:`\\{K_{\\alpha}\\}_{\\alpha}`,
    returns an ensemble :math:`\\{U_{\\alpha}, p_{\\alpha}\\}` that defines
    an :class:`qibo.gates.channels.UnitaryChannel` that approximates the original
    channel up to a precision tolerance in Frobenius norm.

    Args:
        kraus_ops (list): List of Kraus operators as pairs ``(qubits, Ak)``
            where ``qubits`` refers the qubit ids that :math:`A_k` acts on
            and :math:`A_k` is the corresponding matrix as a ``np.ndarray``.
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    target_qubits = [q for q, _ in kraus_ops]
    nqubits = 1 + np.max(target_qubits)
    dim = 2**nqubits

    target = kraus_to_liouville(kraus_ops, order=order, backend=backend)

    # QR decomposition
    unitaries = []
    for _, kraus in kraus_ops:
        Q, _ = np.linalg.qr(kraus)
        unitaries.append(Q)
    # unitaries = np.array(unitaries)
    # unitaries = backend.cast(unitaries)

    # unitaries in Liouville representation
    unitaries_liouville = _individual_kraus_to_liouville(
        list(zip(target_qubits, unitaries)), backend=backend
    )

    # function to minimize
    def function(x0, operators):
        operator = (1 - np.sum(x0)) * np.eye(dim**2, dtype=complex)
        operator = backend.cast(operator, dtype=operator.dtype)
        for prob, oper in zip(x0, operators):
            operator += prob * oper

        return float(backend.calculate_norm(target - operator))

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


def _reshuffling(super_op, order: str = "row", backend=None):
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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ndarray: Choi (Liouville) representation of the quantum channel.
    """

    if not isinstance(order, str):
        raise_error(TypeError, f"order must be type str, but it is type {type(order)}.")

    orders = ["row", "column", "system"]
    if order not in orders:
        raise_error(
            ValueError,
            f"order must be either 'row' or 'column' or 'system', but it is {order}.",
        )
    del orders

    if order == "system":
        raise_error(
            NotImplementedError, "reshuffling not implemented for system vectorization."
        )

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    dim = np.sqrt(super_op.shape[0])

    if (
        super_op.shape[0] != super_op.shape[1]
        or np.mod(dim, 1) != 0
        or np.mod(np.log2(int(dim)), 1) != 0
    ):
        raise_error(ValueError, "super_op must be of shape (4^n, 4^n)")

    dim = int(dim)
    super_op = np.reshape(super_op, [dim] * 4)

    axes = [1, 2] if order == "row" else [0, 3]
    super_op = np.swapaxes(super_op, *axes)

    super_op = np.reshape(super_op, [dim**2, dim**2])
    super_op = backend.cast(super_op, dtype=super_op.dtype)

    return super_op


def _set_gate_and_target_qubits(kraus_ops):  # pragma: no cover
    """Returns Kraus operators as a set of gates acting on
    their respective ``target qubits``.

    Args:
        kraus_ops (list): List of Kraus operators as pairs ``(qubits, Ak)``
            where ``qubits`` refers the qubit ids that :math:`A_k` acts on
            and :math:`A_k` is the corresponding matrix as a ``np.ndarray``.

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
    kraus_ops, order: str = "row", backend=None
):  # pragma: no cover
    """Auxiliary, modified version of :func:`qibo.quantum_info.kraus_to_choi`
    to be used in :func:`qibo.quantum_info.kraus_to_unitaries`. In principle,
    this should be not be accessible to users.
    """
    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    gates, target_qubits = _set_gate_and_target_qubits(kraus_ops)
    nqubits = 1 + max(target_qubits)

    super_ops = []
    for gate in gates:
        kraus_op = FusedGate(*range(nqubits))
        kraus_op.append(gate)
        kraus_op = kraus_op.asmatrix(backend)
        kraus_op = vectorization(kraus_op, order=order, backend=backend)
        kraus_op = np.outer(kraus_op, np.conj(kraus_op))
        super_ops.append(choi_to_liouville(kraus_op, order=order, backend=backend))

    return super_ops
