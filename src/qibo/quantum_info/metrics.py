"""Distances, metrics, and measures for quantum states and channels."""

from typing import Optional, Union

import numpy as np
from scipy import sparse

from qibo.backends import _check_backend
from qibo.config import PRECISION_TOL, raise_error


def purity(state, backend=None):
    """Purity of a quantum state :math:`\\rho`.

    This is given by

    .. math::
        \\text{purity}(\\rho) = \\text{tr}(\\rho^{2}) \\, .

    Args:
        state (ndarray): statevector or density matrix.
    Returns:
        float: Purity of quantum ``state`` :math:`\\rho`.
    """
    backend = _check_backend(backend)
    if (
        (len(state.shape) >= 3)
        or (len(state) == 0)
        or (len(state.shape) == 2 and state.shape[0] != state.shape[1])
    ):
        raise_error(
            TypeError,
            f"state must have dims either (k,) or (k,k), but have dims {state.shape}.",
        )

    if len(state.shape) == 1:
        pur = backend.np.real(backend.calculate_vector_norm(state)) ** 2
    else:
        pur = backend.np.real(backend.np.trace(backend.np.matmul(state, state)))
    return float(pur)


def impurity(state, backend=None):
    """Impurity of quantum state :math:`\\rho`.

    This is given by :math:`1 - \\text{purity}(\\rho)`, where :math:`\\text{purity}`
    is defined in :func:`qibo.quantum_info.purity`.

    Args:
        state (ndarray): statevector or density matrix.

    Returns:
        float: impurity of ``state`` :math:`\\rho`.
    """
    return 1 - purity(state, backend=backend)


def trace_distance(state, target, check_hermitian: bool = False, backend=None):
    """Trace distance between two quantum states, :math:`\\rho` and
    :math:`\\sigma`:

    .. math::
        T(\\rho, \\sigma) = \\frac{1}{2} \\, \\|\\rho - \\sigma\\|_{1} = \\frac{1}{2} \\,
            \\text{tr}\\left[ \\sqrt{(\\rho - \\sigma)^{\\dagger}(\\rho - \\sigma)}
            \\right] \\, ,

    where :math:`\\|\\cdot\\|_{1}` is the Schatten 1-norm.

    Args:
        state (ndarray): statevector or density matrix.
        target (ndarray): statevector or density matrix.
        check_hermitian (bool, optional): if ``True``, checks if
            :math:`\\rho - \\sigma` is Hermitian. If ``False``,
            it assumes the difference is Hermitian.
            Defaults to ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Trace distance between ``state`` :math:`\\rho` and ``target`` :math:`\\sigma`.
    """
    backend = _check_backend(backend)

    if state.shape != target.shape:
        raise_error(
            TypeError,
            f"State has dims {state.shape} while target has dims {target.shape}.",
        )

    if (len(state.shape) >= 3) or (len(state) == 0):
        raise_error(
            TypeError,
            "Both objects must have dims either (k,) or (k,l), "
            + f"but have dims {state.shape} and {target.shape}",
        )

    if isinstance(check_hermitian, bool) is False:
        raise_error(
            TypeError,
            f"check_hermitian must be type bool, but it is type {type(check_hermitian)}.",
        )

    if len(state.shape) == 1:
        state = backend.np.outer(backend.np.conj(state), state)
        target = backend.np.outer(backend.np.conj(target), target)

    difference = state - target
    if check_hermitian is True:
        hermitian = bool(
            float(
                backend.calculate_matrix_norm(
                    backend.np.transpose(backend.np.conj(difference), (1, 0))
                    - difference,
                    order=2,
                )
            )
            <= PRECISION_TOL
        )
        if (
            not hermitian and backend.__class__.__name__ == "CupyBackend"
        ):  # pragma: no cover
            raise_error(
                NotImplementedError,
                "CupyBackend does not support `np.linalg.eigvals`"
                + "for non-Hermitian `state - target`.",
            )
        eigenvalues = backend.calculate_eigenvalues(difference, hermitian=hermitian)
    else:
        eigenvalues = backend.calculate_eigenvalues(difference)

    distance = backend.np.sum(backend.np.absolute(eigenvalues)) / 2

    return distance


def hilbert_schmidt_inner_product(operator_A, operator_B, backend=None):
    """Calculate the Hilbert-Schmidt inner product between two operators.

    Given two operators :math:`A, \\, B \\in \\mathcal{H}`, the Hilbert-Schmidt
    inner product between the two is given by

    .. math::
        \\braket{A, \\, B}_{\\text{HS}} = \\text{tr}\\left(A^{\\dagger} \\, B\\right) \\, .

    Args:
        operator_A (ndarray): operator :math:`A`.
        operator_B (ndarray): operator :math:`B`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Hilbert-Schmidt inner product :math:`\\braket{A, \\, B}_{\\text{HS}}`.
    """
    backend = _check_backend(backend)

    inner_product = backend.np.trace(backend.np.conj(operator_A.T) @ operator_B)

    return backend.np.real(inner_product)


def hilbert_schmidt_distance(state, target, backend=None):
    """Calculate the Hilbert-Schmidt distance between two quantum states:

    .. math::
        \\braket{\\rho - \\sigma, \\, \\rho - \\sigma}_{\\text{HS}} =
            \\text{tr}\\left((\\rho - \\sigma)^{2}\\right) \\, ,

    where :math:`\\braket{\\cdot, \\, \\cdot}_{\\text{HS}}` is the
    :func:`qibo.quantum_info.hilbert_schmidt_inner_product`.

    Args:
        state (ndarray): statevector or density matrix.
        target (ndarray): statevector or density matrix.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Hilbert-Schmidt distance between ``state`` :math:`\\rho`
        and ``target`` :math:`\\sigma`.

    References:
        1. P. J. Coles, M. Cerezo, and L. Cincio, *Strong bound between trace distance
        and Hilbert-Schmidt distance for low-rank states*, `Phys. Rev. A 100, 022103
        <https://doi.org/10.1103/PhysRevA.100.022103>`_ (2019).
    """
    backend = _check_backend(backend)

    if state.shape != target.shape:
        raise_error(
            TypeError,
            f"State has dims {state.shape} while target has dims {target.shape}.",
        )

    if (len(state.shape) >= 3) or (len(state) == 0):
        raise_error(
            TypeError,
            "Both objects must have dims either (k,) or (k,l), "
            + f"but have dims {state.shape} and {target.shape}",
        )

    if len(state.shape) == 1:
        state = backend.np.outer(backend.np.conj(state), state)
        target = backend.np.outer(backend.np.conj(target), target)

    difference = state - target

    return hilbert_schmidt_inner_product(difference, difference, backend=backend)


def fidelity(state, target, check_hermitian: bool = False, backend=None):
    """Fidelity :math:`F(\\rho, \\sigma)` between ``state`` :math:`\\rho` and
    ``target`` state :math:`\\sigma`. In general,

    .. math::
        F(\\rho, \\sigma) = \\text{tr}^{2}\\left( \\sqrt{\\sqrt{\\sigma} \\,
        \\rho^{\\dagger} \\, \\sqrt{\\sigma}} \\right) \\, .

    However, when at least one of the states is pure, then

    .. math::
        F(\\rho, \\sigma) = \\text{tr}(\\rho \\, \\sigma)

    Args:
        state (ndarray): statevector or density matrix.
        target (ndarray): statevector or density matrix.
        check_hermitian (bool, optional): if ``True``, checks if ``state`` is Hermitian.
            Defaults to ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Fidelity between ``state`` :math:`\\rho` and ``target`` :math:`\\sigma`.
    """
    backend = _check_backend(backend)
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    if state.shape != target.shape:
        raise_error(
            TypeError,
            f"State has dims {state.shape} while target has dims {target.shape}.",
        )

    if len(state.shape) >= 3 or len(state.shape) == 0:
        raise_error(
            TypeError,
            "Both objects must have dims either (k,) or (k,l), "
            + f"but have dims {state.shape} and {target.shape}",
        )

    if isinstance(check_hermitian, bool) is False:
        raise_error(
            TypeError,
            f"check_hermitian must be type bool, but it is type {type(check_hermitian)}.",
        )

    # check purity if both states are density matrices
    if len(state.shape) == 2 and len(target.shape) == 2:
        purity_state = purity(state, backend=backend)
        purity_target = purity(target, backend=backend)

        # if both states are mixed, default to full fidelity calculation
        if (
            abs(purity_state - 1) > PRECISION_TOL
            and abs(purity_target - 1) > PRECISION_TOL
        ):
            hermitian = check_hermitian is False or _check_hermitian(
                state, backend=backend
            )
            # using eigh since rho is supposed to be Hermitian
            eigenvalues, eigenvectors = backend.calculate_eigenvectors(
                state, hermitian=hermitian
            )
            state = np.zeros(state.shape, dtype=complex)
            state = backend.cast(state, dtype=state.dtype)
            for eig, eigvec in zip(
                eigenvalues, backend.np.transpose(eigenvectors, (1, 0))
            ):
                matrix = backend.np.sqrt(eig) * backend.np.outer(
                    eigvec, backend.np.conj(eigvec)
                )
                matrix = backend.cast(matrix, dtype=matrix.dtype)
                state = state + matrix
                del matrix

            fid = state @ target @ state

            # since sqrt(rho) is Hermitian, we can use eigh again
            eigenvalues, eigenvectors = backend.calculate_eigenvectors(
                fid, hermitian=hermitian
            )
            fid = np.zeros(state.shape, dtype=complex)
            fid = backend.cast(fid, dtype=fid.dtype)
            for eig, eigvec in zip(
                eigenvalues, backend.np.transpose(eigenvectors, (1, 0))
            ):
                if backend.np.real(eig) > PRECISION_TOL:
                    matrix = backend.np.sqrt(eig) * backend.np.outer(
                        eigvec, backend.np.conj(eigvec)
                    )
                    matrix = backend.cast(matrix, dtype=matrix.dtype)
                    fid = fid + matrix
                    del matrix

            fid = backend.np.real(backend.np.trace(fid))

            return fid

    # if any of the states is pure, perform lighter calculation
    fid = (
        backend.np.abs(backend.np.matmul(backend.np.conj(state), target)) ** 2
        if len(state.shape) == 1
        else backend.np.real(backend.np.trace(backend.np.matmul(state, target)))
    )

    return fid


def infidelity(state, target, check_hermitian: bool = False, backend=None):
    """Infidelity between ``state`` :math:`\\rho` and ``target`` state
    :math:`\\sigma`, which is given by

    .. math::
        1 - F(\\rho, \\, \\sigma) \\, ,

    where :math:`F(\\rho, \\, \\sigma)` is the :func:`qibo.quantum_info.fidelity`
    between ``state`` and ``target``.

    Args:
        state (ndarray): statevector or density matrix.
        target (ndarray): statevector or density matrix.
        check_hermitian (bool, optional): if ``True``, checks if ``state`` is Hermitian.
            Defaults to ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Infidelity between ``state`` :math:`\\rho` and ``target`` :math:`\\sigma`.
    """
    return 1 - fidelity(state, target, check_hermitian=check_hermitian, backend=backend)


def bures_angle(state, target, check_hermitian: bool = False, backend=None):
    """Calculates the Bures angle :math:`D_{A}` between a ``state``
    :math:`\\rho` and a ``target`` state :math:`\\sigma`. This is given by

    .. math::
        D_{A}(\\rho, \\, \\sigma) = \\text{arccos}\\left(\\sqrt{F(\\rho, \\, \\sigma)}\\right) \\, ,

    where :math:`F(\\rho, \\sigma)` is the :func:`qibo.quantum_info.fidelity`
    between `state` and `target`.

    Args:
        state (ndarray): statevector or density matrix.
        target (ndarray): statevector or density matrix.
        check_hermitian (bool, optional): if ``True``, checks if ``state`` is Hermitian.
            Defaults to ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Bures angle between ``state`` and ``target``.
    """
    backend = _check_backend(backend)

    angle = backend.np.arccos(
        backend.np.sqrt(fidelity(state, target, check_hermitian, backend=backend))
    )

    return angle


def bures_distance(state, target, check_hermitian: bool = False, backend=None):
    """Calculates the Bures distance :math:`D_{B}` between a ``state``
    :math:`\\rho` and a ``target`` state :math:`\\sigma`. This is given by

    .. math::
        D_{B}(\\rho, \\, \\sigma) = \\sqrt{2 \\, \\left(1 - \\sqrt{F(\\rho, \\, \\sigma)}\\right)}

    where :math:`F(\\rho, \\sigma)` is the :func:`qibo.quantum_info.fidelity`
    between `state` and `target`.

    Args:
        state (ndarray): statevector or density matrix.
        target (ndarray): statevector or density matrix.
        check_hermitian (bool, optional): if ``True``, checks if ``state`` is Hermitian.
            Defaults to ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Bures distance between ``state`` and ``target``.
    """
    backend = _check_backend(backend)
    sqrt_fid = backend.np.sqrt(
        fidelity(state, target, check_hermitian, backend=backend)
    )
    distance = backend.np.sqrt(2 * (1 - sqrt_fid))

    return distance


def process_fidelity(channel, target=None, check_unitary: bool = False, backend=None):
    """Process fidelity between a quantum ``channel`` :math:`\\mathcal{E}` and
    a ``target`` unitary channel :math:`U`. The process fidelity is defined as

    .. math::
        F_{\\text{pro}}(\\mathcal{E}, \\mathcal{U}) = \\frac{1}{d^{2}} \\,
            \\text{tr}(\\mathcal{E}^{\\dagger} \\, \\mathcal{U})

    Args:
        channel: quantum channel :math:`\\mathcal{E}`.
        target (optional): quantum channel :math:`U`. If ``None``, target is the
            Identity channel. Defaults to ``None``.
        check_unitary (bool, optional): if ``True``, checks if one of the
            input channels is unitary. Default: ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Process fidelity between ``channel`` and ``target``.
    """
    backend = _check_backend(backend)

    if target is not None:
        if channel.shape != target.shape:
            raise_error(
                TypeError,
                f"Channels must have the same dims, but {channel.shape} != {target.shape}",
            )

    dim = int(np.sqrt(channel.shape[0]))

    if check_unitary is True:
        norm_channel = float(
            backend.calculate_matrix_norm(
                backend.np.matmul(
                    backend.np.conj(backend.np.transpose(channel, (1, 0))), channel
                )
                - backend.np.eye(dim**2)
            )
        )
        if target is None and norm_channel > PRECISION_TOL:
            raise_error(TypeError, "Channel is not unitary and Target is None.")
        if target is not None:
            norm_target = float(
                backend.calculate_vector_norm(
                    backend.np.matmul(
                        backend.np.conj(backend.np.transpose(target, (1, 0))), target
                    )
                    - backend.np.eye(dim**2)
                )
            )
            if (norm_channel > PRECISION_TOL) and (norm_target > PRECISION_TOL):
                raise_error(TypeError, "Neither channel is unitary.")

    if target is None:
        # With no target, return process fidelity with Identity channel
        process_fid = backend.np.real(backend.np.trace(channel)) / dim**2
        process_fid = float(process_fid)

        return process_fid

    process_fid = backend.np.matmul(
        backend.np.transpose(backend.np.conj(channel), (1, 0)), target
    )
    process_fid = backend.np.real(backend.np.trace(process_fid)) / dim**2

    return process_fid


def process_infidelity(channel, target=None, check_unitary: bool = False, backend=None):
    """Process infidelity between quantum channel :math:`\\mathcal{E}` and a
    ``target`` unitary channel :math:`U`. The process infidelity is defined as

    .. math::
        1 - F_{\\text{pro}}(\\mathcal{E}, \\mathcal{U}) \\, ,

    where :math:`F_{\\text{pro}}` is the :func:`qibo.quantum_info.process_fidelity`.

    Args:
        channel: quantum channel :math:`\\mathcal{E}`.
        target (optional): quantum channel :math:`U`. If ``None``, target is the
            Identity channel. Defaults to ``None``.
        check_unitary (bool, optional): if ``True``, checks if one of the
            input channels is unitary. Defaults to ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.


    Returns:
        float: Process infidelity between ``channel`` :math:`\\mathcal{E}`
        and ``target`` :math:`U`.
    """
    return 1 - process_fidelity(
        channel, target=target, check_unitary=check_unitary, backend=backend
    )


def average_gate_fidelity(
    channel, target=None, check_unitary: bool = False, backend=None
):
    """Average gate fidelity between a quantum ``channel`` :math:`\\mathcal{E}`
    and a ``target`` unitary channel :math:`U`. The average gate fidelity is
    defined as

    .. math::
        F_{\\text{avg}}(\\mathcal{E}, \\mathcal{U}) = \\frac{d \\,
            F_{pro}(\\mathcal{E}, \\mathcal{U}) + 1}{d + 1}

    where :math:`d` is the dimension of the channels and
    :math:`F_{pro}(\\mathcal{E}, \\mathcal{U})` is the
    :meth:`~qibo.metrics.process_fidelily` of channel
    :math:`\\mathcal{E}` with respect to the unitary
    channel :math:`\\mathcal{U}`.

    Args:
        channel: quantum channel :math:`\\mathcal{E}`.
        target (optional): quantum channel :math:`\\mathcal{U}`.
            If ``None``, target is the Identity channel. Defaults to ``None``.
        check_unitary (bool, optional): if ``True``, checks if one of the
            input channels is unitary. Default: ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Process fidelity between channel :math:`\\mathcal{E}`
        and target unitary channel :math:`\\mathcal{U}`.
    """

    dim = channel.shape[0]

    process_fid = process_fidelity(
        channel, target, check_unitary=check_unitary, backend=backend
    )
    process_fid = (dim * process_fid + 1) / (dim + 1)

    return process_fid


def gate_error(channel, target=None, check_unitary: bool = False, backend=None):
    """Gate error between a quantum ``channel`` :math:`\\mathcal{E}` and a
    ``target`` unitary channel :math:`U`, which is defined as

    .. math::
        E(\\mathcal{E}, \\mathcal{U}) = 1 - F_{\\text{avg}}(\\mathcal{E}, \\mathcal{U}) \\, ,

    where :math:`F_{\\text{avg}}(\\mathcal{E}, \\mathcal{U})` is the
    :func:`qibo.quantum_info.average_gate_fidelity`.

    Args:
        channel: quantum channel :math:`\\mathcal{E}`.
        target (optional): quantum channel :math:`\\mathcal{U}`. If ``None``,
            target is the Identity channel. Defaults to ``None``.
        check_unitary (bool, optional): if ``True``, checks if one of the
            input channels is unitary. Default: ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Gate error between ``channel`` :math:`\\mathcal{E}`
        and ``target`` :math:`\\mathcal{U}`.
    """
    error = 1 - average_gate_fidelity(
        channel, target, check_unitary=check_unitary, backend=backend
    )

    return error


def diamond_norm(channel, target=None, backend=None, **kwargs):  # pragma: no cover
    """Calculates the diamond norm :math:`\\|\\mathcal{E}\\|_{\\diamond}` of
    ``channel`` :math:`\\mathcal{E}`, which is given by

    .. math::
        \\|\\mathcal{E}\\|_{\\diamond} = \\max_{\\rho} \\, \\| \\left(\\mathcal{E} \\otimes I_{d^{2}}\\right)(\\rho) \\|_{1} \\, ,

    where :math:`I_{d^{2}}` is the :math:`d^{2} \\times d^{2}` Identity operator,
    :math:`d = 2^{n}`, :math:`n` is the number of qubits,
    and :math:`\\|\\cdot\\|_{1}` denotes the trace norm.

    If a ``target`` channel :math:`\\Lambda` is specified,
    then it calculates :math:`\\| \\mathcal{E} - \\Lambda\\|_{\\diamond}`.

    Example::

        from qibo.quantum_info import diamond_norm, random_unitary, to_choi

        nqubits = 1
        dim = 2**nqubits

        unitary = random_unitary(dim)
        unitary = to_choi(unitary, order="row")

        unitary_2 = random_unitary(dim)
        unitary_2 = to_choi(unitary_2, order="row")

        dnorm = diamond_norm(unitary, unitary_2)

    Args:
        channel (ndarray): row-vectorized Choi representation of a quantum channel.
        target (ndarray, optional): row-vectorized Choi representation of a target
            quantum channel. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.
        kwargs: optional arguments to pass to CVXPY solver. For more information,
            please visit `CVXPY's API documentation
            <https://www.cvxpy.org/api_reference/cvxpy.problems.html#problem>`_.

    Returns:
        float: diamond norm of either ``channel`` or ``channel - target``.

    .. note::
        This function requires the optional CVXPY package to be installed.
    """
    import cvxpy

    backend = _check_backend(backend)

    if target is not None:
        if channel.shape != target.shape:
            raise_error(
                TypeError,
                f"Channels must have the same dims, but {channel.shape} != {target.shape}",
            )

    if target is not None:
        channel -= target

    # `CVXPY` only works with `numpy`, so this function has to
    # convert any channel to the `numpy` backend by default
    backend = _check_backend(backend)

    channel = backend.to_numpy(channel)

    channel = backend.np.transpose(channel, (1, 0))
    channel_real = backend.np.real(channel)
    channel_imag = backend.np.imag(channel)

    dim = int(np.sqrt(channel.shape[0]))

    first_variables_real = cvxpy.Variable(shape=(dim, dim))
    first_variables_imag = cvxpy.Variable(shape=(dim, dim))
    first_variables = cvxpy.bmat(
        [
            [first_variables_real, -first_variables_imag],
            [first_variables_imag, first_variables_real],
        ]
    )

    second_variables_real = cvxpy.Variable(shape=(dim, dim))
    second_variables_imag = cvxpy.Variable(shape=(dim, dim))
    second_variables = cvxpy.bmat(
        [
            [second_variables_real, -second_variables_imag],
            [second_variables_imag, second_variables_real],
        ]
    )

    variables_real = cvxpy.Variable(shape=(dim**2, dim**2))
    variables_imag = cvxpy.Variable(shape=(dim**2, dim**2))
    identity = sparse.eye(dim)

    constraints_real = cvxpy.bmat(
        [
            [cvxpy.kron(identity, first_variables_real), variables_real],
            [variables_real.T, cvxpy.kron(identity, second_variables_real)],
        ]
    )
    constraints_imag = cvxpy.bmat(
        [
            [cvxpy.kron(identity, first_variables_imag), variables_imag],
            [-variables_imag.T, cvxpy.kron(identity, second_variables_imag)],
        ]
    )
    constraints_block = cvxpy.bmat(
        [[constraints_real, -constraints_imag], [constraints_imag, constraints_real]]
    )

    constraints = [
        first_variables >> 0,
        first_variables_real == first_variables_real.T,
        first_variables_imag == -first_variables_imag.T,
        cvxpy.trace(first_variables_real) == 1,
        second_variables >> 0,
        second_variables_real == second_variables_real.T,
        second_variables_imag == -second_variables_imag.T,
        cvxpy.trace(second_variables_real) == 1,
        constraints_block >> 0,
    ]

    objective_function = cvxpy.Maximize(
        cvxpy.trace(channel_real @ variables_real)
        + cvxpy.trace(channel_imag @ variables_imag)
    )
    problem = cvxpy.Problem(objective=objective_function, constraints=constraints)
    solution = problem.solve(**kwargs)

    return solution


def expressibility(
    circuit,
    power_t: int,
    samples: int,
    order: Optional[Union[int, float, str]] = 2,
    backend=None,
):
    """Returns the expressibility :math:`\\|A\\|` of a parametrized circuit,
    where

    .. math::
        A = \\int_{\\text{Haar}} d\\psi \\, \\left(|\\psi\\rangle\\right.\\left.
            \\langle\\psi|\\right)^{\\otimes t} - \\int_{\\Theta} d\\psi \\,
            \\left(|\\psi_{\\theta}\\rangle\\right.\\left.
            \\langle\\psi_{\\theta}|\\right)^{\\otimes t}

    Args:
        circuit (:class:`qibo.models.Circuit`): Parametrized circuit.
        power_t (int): power that defines the :math:`t`-design.
        samples (int): number of samples to estimate the integrals.
        order (int or float or str, optional): order of the norm :math:`\\|A\\|`.
            For specifications, see :meth:`qibo.backends.abstract.calculate_norm`.
            Defaults to :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Expressibility of parametrized circuit.
    """

    if isinstance(power_t, int) is False:
        raise_error(
            TypeError, f"power_t must be type int, but it is type {type(power_t)}."
        )

    if isinstance(samples, int) is False:
        raise_error(
            TypeError, f"samples must be type int, but it is type {type(samples)}."
        )

    from qibo.quantum_info.utils import (  # pylint: disable=C0415
        haar_integral,
        pqc_integral,
    )

    backend = _check_backend(backend)

    deviation = haar_integral(
        circuit.nqubits, power_t, samples=None, backend=backend
    ) - pqc_integral(circuit, power_t, samples, backend=backend)

    fid = float(backend.calculate_vector_norm(deviation, order=order))

    return fid


def frame_potential(
    circuit,
    power_t: int,
    samples: int = None,
    backend=None,
):
    """Returns the frame potential of a parametrized circuit under uniform
    sampling of the parameters.

    For :math:`n` qubits and moment :math:`t`, the frame potential
    :math:`\\mathcal{F}_{\\mathcal{U}}^{(t)}` if given by [1]

    .. math::
        \\mathcal{F}_{\\mathcal{U}}^{(t)} = \\int_{U,V \\in \\mathcal{U}} \\,
            \\text{d}U \\, \\text{d}V \\, \\bigl| \\, \\text{tr}(U^{\\dagger} \\, V)
            \\, \\bigr|^{2t} \\, ,

    where :math:`\\mathcal{U}` is the group of unitaries defined by the parametrized circuit.
    The frame potential is approximated by the average

    .. math::
        \\mathcal{F}_{\\mathcal{U}}^{(t)} \\approx \\frac{1}{N} \\,
            \\sum_{k=1}^{N} \\, \\bigl| \\, \\text{tr}(U_{k}^{\\dagger} \\, V_{k}) \\, \\bigr|^{2t} \\, ,

    where :math:`N` is the number of ``samples``.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): Parametrized circuit.
        power_t (int): power that defines the :math:`t`-design.
        samples (int): number of samples to estimate the integral.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Frame potential of the parametrized circuit.

    References:
        1. M. Liu *et al.*, *Estimating the randomness of quantum circuit ensembles up to 50 qubits*.
        `arXiv:2205.09900 [quant-ph] <https://arxiv.org/abs/2205.09900>`_.
    """
    if not isinstance(power_t, int):
        raise_error(
            TypeError, f"power_t must be type int, but it is type {type(power_t)}."
        )

    if not isinstance(samples, int):
        raise_error(
            TypeError, f"samples must be type int, but it is type {type(samples)}."
        )

    backend = _check_backend(backend)

    nqubits = circuit.nqubits
    dim = 2**nqubits

    potential = 0
    for _ in range(samples):
        unitary_1 = circuit.copy()
        params_1 = np.random.uniform(-np.pi, np.pi, circuit.trainable_gates.nparams)
        unitary_1.set_parameters(params_1)
        unitary_1 = unitary_1.unitary(backend) / np.sqrt(dim)

        for _ in range(samples):
            unitary_2 = circuit.copy()
            params_2 = np.random.uniform(-np.pi, np.pi, circuit.trainable_gates.nparams)
            unitary_2.set_parameters(params_2)
            unitary_2 = unitary_2.unitary(backend) / np.sqrt(dim)

            potential += backend.np.abs(
                backend.np.trace(
                    backend.np.transpose(backend.np.conj(unitary_1), (1, 0)) @ unitary_2
                )
            ) ** (2 * power_t)

    return potential / samples**2


def quantum_fisher_information_matrix(
    circuit,
    parameters=None,
    initial_state=None,
    return_complex: bool = True,
    backend=None,
):
    """Calculate the Quantum Fisher Information Matrix (QFIM) of a parametrized ``circuit``.

    Given a set of ``parameters`` :math:`\\theta = \\{\\theta_{k}\\}_{k\\in[M]}` and a
    parameterized unitary ``circuit`` :math:`U(\\theta)` acting on an ``initial_state``
    :math:`\\ket{\\phi}`, the QFIM is such that its elements can be calculated as

    .. math::
        \\mathbf{F}_{jk} = 4 \\, \\text{Re}\\left\\{ \\braket{\\partial_{j} \\psi | \\partial_{k}
            \\psi} - \\braket{\\partial_{j} \\psi | \\psi}\\!\\braket{\\psi | \\partial_{k} \\psi}
            \\right\\} \\, ,

    where we have used the short notations :math:`\\ket{\\psi} \\equiv \\ket{\\psi(\\theta)}
    = U(\\theta) \\ket{\\phi}`, and :math:`\\ket{\\partial_{k} \\psi} \\equiv \\frac{\\partial}
    {\\partial\\theta_{k}} \\ket{\\psi(\\theta)}`.
    If the ``initial_state`` :math:`\\ket{\\phi}` is not specified, it defaults to
    :math:`\\ket{0}^{\\otimes n}`.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): parametrized circuit :math:`U(\\theta)`.
        parameters (ndarray, optional): parameters whose QFIM to calculate.
            If ``None``, QFIM is calculated with the paremeters from ``circuit``, i.e.
            ``parameters = circuit.get_parameters()``. Defaults to ``None``.
        initial_state (ndarray, optional): Initial configuration. It can be specified
            by the setting the state vector using an array or a circuit. If ``None``,
            the initial state is :math:`\\ket{0}^{\\otimes n}`. Defaults to ``None``.
        return_complex (bool, optional): If ``True``, calculates the Jacobian matrix
            of real and imaginary parts of :math:`\\ket{\\psi(\\theta)}`. If ``False``,
            calculates only the Jacobian matrix of the real part. Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses :class:`qibo.backends.GlobalBackend`.
            Defaults to ``None``.

    Returns:
        ndarray: Quantum Fisher Information :math:`\\mathbf{F}`.
    """
    backend = _check_backend(backend)

    if parameters is None:
        parameters = circuit.get_parameters()
        parameters = backend.cast(parameters, dtype=float).flatten()

    jacobian = backend.calculate_jacobian_matrix(
        circuit, parameters, initial_state, return_complex
    )

    if return_complex:
        jacobian = jacobian[0] + 1j * jacobian[1]

    jacobian = backend.cast(jacobian, dtype=np.complex128)

    copied = circuit.copy(deep=True)
    copied.set_parameters(parameters)

    state = backend.execute_circuit(copied, initial_state=initial_state).state()

    overlaps = jacobian.T @ state

    qfim = jacobian.T @ jacobian
    qfim = qfim - backend.np.outer(overlaps, backend.np.conj(overlaps.T))

    return 4 * backend.np.real(qfim)


def _check_hermitian(matrix, backend=None):
    """Checks if a given matrix is Hermitian.

    Args:
        matrix: input array.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        bool: whether the matrix is Hermitian.
    """
    backend = _check_backend(backend)

    norm = backend.calculate_matrix_norm(backend.np.conj(matrix).T - matrix, order=2)

    hermitian = bool(float(norm) <= PRECISION_TOL)

    return hermitian
