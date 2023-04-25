import numpy as np

from qibo.backends import GlobalBackend
from qibo.config import PRECISION_TOL, raise_error


def purity(state):
    """Purity of a quantum state :math:`\\rho`, which is given by :math:`\\text{Tr}(\\rho^{2})`.

    Args:
        state: statevector or density matrix.
    Returns:
        float: Purity of quantum state :math:`\\rho`.
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

    if len(state.shape) == 1:
        pur = np.abs(np.dot(np.conj(state), state)) ** 2
    else:
        pur = np.real(np.trace(np.dot(state, state)))

    pur = float(pur)

    return pur


def entropy(state, base: float = 2, validate: bool = False, backend=None):
    """The von-Neumann entropy :math:`S(\\rho)` of a quantum state :math:`\\rho`, which
    is given by

    .. math::
        S(\\rho) = - \\text{Tr}\\left[\\rho \\, \\log(\\rho)\\right]

    Args:
        state: state vector or density matrix.
        base (float, optional): the base of the log. Default: 2.
        validate (bool, optional): if ``True``, checks if ``state`` is Hermitian. If ``False``,
            it assumes ``state`` is Hermitian . Default: ``False``.
        backend (``qibo.backends.abstract.Backend``, optional): backend to be used
            in the execution. If ``None``, it uses ``GlobalBackend()``.
            Defaults to ``None``.

    Returns:
        float: The von-Neumann entropy :math:`S(\\rho)`.
    """
    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    if base <= 0.0:
        raise_error(ValueError, "log base must be non-negative.")

    if (
        (len(state.shape) >= 3)
        or (len(state) == 0)
        or (len(state.shape) == 2 and state.shape[0] != state.shape[1])
    ):
        raise_error(
            TypeError,
            f"Object must have dims either (k,) or (k,k), but have dims {state.shape}.",
        )

    if purity(state) == 1.0:
        ent = 0.0
    else:
        if validate:
            hermitian = bool(
                backend.calculate_norm(np.transpose(np.conj(state)) - state)
                <= PRECISION_TOL
            )
            if (
                not hermitian and backend.__class__.__name__ == "CupyBackend"
            ):  # pragma: no cover
                raise_error(
                    NotImplementedError,
                    f"CupyBackend does not support `np.linalg.eigvals` for non-Hermitian `state`.",
                )
            eigenvalues = (
                np.linalg.eigvalsh(state) if hermitian else np.linalg.eigvals(state)
            )
        else:
            eigenvalues = np.linalg.eigvalsh(state)

        if base == 2:
            log_prob = np.where(eigenvalues != 0, np.log2(eigenvalues), 0.0)
        elif base == 10:
            log_prob = np.where(eigenvalues != 0, np.log10(eigenvalues), 0.0)
        elif base == np.e:
            log_prob = np.where(eigenvalues != 0, np.log(eigenvalues), 0.0)
        else:
            log_prob = np.where(
                eigenvalues != 0, np.log(eigenvalues) / np.log(base), 0.0
            )

        ent = -np.sum(eigenvalues * log_prob)
        # absolute value if entropy == 0.0 to avoid returning -0.0
        ent = np.abs(ent) if ent == 0.0 else ent

    ent = float(ent)

    return ent


def trace_distance(state, target, validate: bool = False, backend=None):
    """Trace distance between two quantum states, :math:`\\rho` and :math:`\\sigma`:

    .. math::
        T(\\rho, \\sigma) = \\frac{1}{2} \\, \\|\\rho - \\sigma\\|_{1} = \\frac{1}{2} \\, \\text{Tr}\\left[ \\sqrt{(\\rho - \\sigma)^{\\dagger}(\\rho - \\sigma)} \\right] \\, ,

    where :math:`\\|\\cdot\\|_{1}` is the Schatten 1-norm.

    Args:
        state: state vector or density matrix.
        target: state vector or density matrix.
        validate (bool, optional): if ``True``, checks if :math:`\\rho - \\sigma` is Hermitian.
            If ``False``, it assumes the difference is Hermitian. Default: ``False``.
        backend (``qibo.backends.abstract.Backend``, optional): backend to be used
            in the execution. If ``None``, it uses ``GlobalBackend()``.
            Defaults to ``None``.

    Returns:
        float: Trace distance between state :math:`\\rho` and target :math:`\\sigma`.
    """
    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

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
        state = np.outer(np.conj(state), state)
        target = np.outer(np.conj(target), target)

    difference = state - target
    if validate:
        hermitian = bool(
            backend.calculate_norm(np.transpose(np.conj(difference)) - difference)
            <= PRECISION_TOL
        )
        if (
            not hermitian and backend.__class__.__name__ == "CupyBackend"
        ):  # pragma: no cover
            raise_error(
                NotImplementedError,
                f"CupyBackend does not support `np.linalg.eigvals` for non-Hermitian `state - target`.",
            )
        eigenvalues = (
            np.linalg.eigvalsh(difference)
            if hermitian
            else np.linalg.eigvals(difference)
        )
    else:
        eigenvalues = np.linalg.eigvalsh(difference)

    distance = np.sum(np.absolute(eigenvalues)) / 2
    distance = float(distance)

    return distance


def hilbert_schmidt_distance(state, target):
    """Hilbert-Schmidt distance between two quantum states:

    .. math::
        <\\rho \\, , \\, \\sigma>_{\\text{HS}} = \\text{Tr}\\left((\\rho - \\sigma)^{2}\\right)

    Args:
        state: state vector or density matrix.
        target: state vector or density matrix.

    Returns:
        float: Hilbert-Schmidt distance between state :math:`\\rho` and target :math:`\\sigma`.
    """

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
        state = np.outer(np.conj(state), state)
        target = np.outer(np.conj(target), target)

    distance = np.real(np.trace((state - target) ** 2))
    distance = float(distance)

    return distance


def fidelity(state, target, validate: bool = False):
    """Fidelity between two quantum states (when at least one state is pure).

    .. math::
        F(\\rho, \\sigma) = \\text{Tr}^{2}\\left( \\sqrt{\\sqrt{\\sigma} \\, \\rho^{\\dagger} \\, \\sqrt{\\sigma}} \\right) = \\text{Tr}(\\rho \\, \\sigma)

    where the last equality holds because the ``target`` state
    :math:`\\sigma` is assumed to be pure.

    Args:
        state: state vector or density matrix.
        target: state vector or density matrix.
        validate (bool, optional): if ``True``, checks if one of the
            input states is pure. Default is ``False``.

    Returns:
        float: Fidelity between state :math:`\\rho` and target :math:`\\sigma`.
    """

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

    if validate:
        purity_state = purity(state)
        purity_target = purity(target)
        if (
            (purity_state < 1.0 - PRECISION_TOL) or (purity_state > 1.0 + PRECISION_TOL)
        ) and (
            (purity_target < 1.0 - PRECISION_TOL)
            or (purity_target > 1.0 + PRECISION_TOL)
        ):
            raise_error(
                ValueError,
                f"Neither state is pure. Purity state: {purity_state} , "
                + f"Purity target: {purity_target}.",
            )

    if len(state.shape) == 1 and len(target.shape) == 1:
        fid = np.abs(np.dot(np.conj(state), target)) ** 2
    elif len(state.shape) == 2 and len(target.shape) == 2:
        fid = np.real(np.trace(np.dot(state, target)))

    fid = float(fid)

    return fid


def process_fidelity(channel, target=None, validate: bool = False, backend=None):
    """Process fidelity between two quantum channels (when at least one channel is` unitary),

    .. math::
        F_{pro}(\\mathcal{E}, \\mathcal{U}) = \\frac{1}{d^{2}} \\, \\text{Tr}(\\mathcal{E}^{\\dagger} \\, \\mathcal{U})

    Args:
        channel: quantum channel.
        target (optional): quantum channel. If None, target is the Identity channel.
            Default: ``None``.
        validate (bool, optional): if True, checks if one of the
            input channels is unitary. Default: ``False``.
        backend (``qibo.backends.abstract.Backend``, optional): backend to be used
            in the execution. If ``None``, it uses ``GlobalBackend()``.
            Defaults to ``None``.

    Returns:
        float: Process fidelity between channels :math:`\\mathcal{E}`
            and target :math:`\\mathcal{U}`.
    """
    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    if target is not None:
        if channel.shape != target.shape:
            raise_error(
                TypeError,
                f"Channels must have the same dims, but {channel.shape} != {target.shape}",
            )

    dim = int(np.sqrt(channel.shape[0]))

    if validate:
        norm_channel = backend.calculate_norm(
            np.dot(np.conj(np.transpose(channel)), channel) - np.eye(dim**2)
        )
        if target is None and norm_channel > PRECISION_TOL:
            raise_error(TypeError, "Channel is not unitary and Target is None.")
        if target is not None:
            norm_target = backend.calculate_norm(
                np.dot(np.conj(np.transpose(target)), target) - np.eye(dim**2)
            )
            if (norm_channel > PRECISION_TOL) and (norm_target > PRECISION_TOL):
                raise_error(TypeError, "Neither channel is unitary.")

    if target is None:
        # With no target, return process fidelity with Identity channel
        fid = np.real(np.trace(channel)) / dim**2
        fid = float(fid)
        return fid

    fid = np.real(np.trace(np.dot(np.conj(np.transpose(channel)), target))) / dim**2
    fid = float(fid)

    return fid


def average_gate_fidelity(channel, target=None, backend=None):
    """Average gate fidelity between two quantum channels (when at least one channel is unitary),

    .. math::
        F_{\\text{avg}}(\\mathcal{E}, \\mathcal{U}) = \\frac{d \\, F_{pro}(\\mathcal{E}, \\mathcal{U}) + 1}{d + 1}

    where :math:`d` is the dimension of the channels and
    :math:`F_{pro}(\\mathcal{E}, \\mathcal{U})` is the
    :meth:`~qibo.metrics.process_fidelily` of channel
    :math:`\\mathcal{E}` with respect to the unitary
    channel :math:`\\mathcal{U}`.

    Args:
        channel: quantum channel :math:`\\mathcal{E}`.
        target (optional): quantum channel :math:`\\mathcal{U}`.
            If ``None``, target is the Identity channel.
            Default is ``None``.
        backend (``qibo.backends.abstract.Backend``, optional): backend to be used
            in the execution. If ``None``, it uses ``GlobalBackend()``.
            Defaults to ``None``.

    Returns:
        float: Process fidelity between channel :math:`\\mathcal{E}`
            and target unitary channel :math:`\\mathcal{U}`.
    """

    dim = channel.shape[0]

    process_fid = process_fidelity(channel, target, backend=backend)
    process_fid = (dim * process_fid + 1) / (dim + 1)

    return process_fid


def gate_error(channel, target=None, backend=None):
    """Gate error between two quantum channels (when at least one is unitary), which is
    defined as

    .. math::
        E(\\mathcal{E}, \\mathcal{U}) = 1 - F_{\\text{avg}}(\\mathcal{E}, \\mathcal{U}) \\, ,

    where :math:`F_{\\text{avg}}(\\mathcal{E}, \\mathcal{U})` is the ``average_gate_fidelity()``
    between channel :math:`\\mathcal{E}` and target :math:`\\mathcal{U}`.

    Args:
        channel: quantum channel :math:`\\mathcal{E}`.
        target (optional): quantum channel :math:`\\mathcal{U}`. If ``None``,
            target is the Identity channel. Default is ``None``.
        backend (``qibo.backends.abstract.Backend``, optional): backend to be used
            in the execution. If ``None``, it uses ``GlobalBackend()``.
            Defaults to ``None``.

    Returns:
        float: Gate error between :math:`\\mathcal{E}` and :math:`\\mathcal{U}`.
    """
    error = 1 - average_gate_fidelity(channel, target, backend=backend)

    return error
