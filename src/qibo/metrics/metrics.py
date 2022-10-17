# -*- coding: utf-8 -*-
import numpy as np
import scipy


def trace_distance(state: np.ndarray, target: np.ndarray):
    """Trace distance between two quantum states

    ..math::
        T(\\rho, \\sigma) \\coloneqq \\frac{1}{2} \\, ||\\rho - \\sigma||_{1}

    where :math:`||\\cdot||_{1}` is the Schatten 1-norm.

    Args:
        state: state vector or density matrix.
        target: state vector or density matrix.

    Returns:
        Trace distance between state :math:`\\rho` and target :math:`\\sigma`.

    """

    if state.shape != target.shape:
        raise TypeError(
            f"State has dims {state.shape} while target has dims {target.shape}."
        )
    elif len(state.shape) >= 3:
        raise TypeError(
            f"Both objects must have dims either (k,) or (k,l), but have dims {state.shape} and {target.shape}"
        )

    if len(state.shape) == 1:
        state = np.outer(np.conj(state), state)
        target = np.outer(np.conj(target), target)

    difference = state - target
    difference_sqrt, _ = scipy.linalg.sqrtm(np.dot(np.conj(np.transpose(difference)), difference))
    return np.trace(difference_sqrt) / 2


def hilbert_schmidt_distance(state, target):
    """Hilbert-Schmidt distance between two quantum states

    ..math::
        <\\rho, \\sigma>_{\\text{HS}} = \\text{Tr}\\left[(\\rho - \\sigma)^{2}\\right]

    Args:
        state: state vector or density matrix.
        target: state vector or density matrix.

    Returns:
        Hilbert-Schmidt distance between state :math:`\\rho` and target :math:`\\sigma`.

    """

    if state.shape != target.shape:
        raise TypeError(
            f"State has dims {state.shape} while target has dims {target.shape}."
        )
    elif len(state.shape) >= 3:
        raise TypeError(
            f"Both objects must have dims either (k,) or (k,l), but have dims {state.shape} and {target.shape}"
        )

    if len(state.shape) == 1:
        state = np.outer(np.conj(state), state)
        target = np.outer(np.conj(target), target)

    return np.trace((state - target) ** 2)


def fidelity(state, target):
    """Fidelity between two quantum states (when at least one state is pure).

    ..math::
        F(\\rho, \\sigma) = \\Tr(\\rho \\, \\sigma)

    Args:
        state: state vector or density matrix.
        target: state vector or density matrix.

    Returns:
        Fidelity between state :math:`\\rho` and target :math:`\\sigma`.

    """

    if state.shape != target.shape:
        raise TypeError(
            f"State has dims {state.shape} while target has dims {target.shape}."
        )
    elif len(state.shape) >= 3:
        raise TypeError(
            f"Both objects must have dims either (k,) or (k,l), but have dims {state.shape} and {target.shape}"
        )

    if len(state.shape) == 1 and len(target.shape) == 1:
        fid = np.abs(np.dot(np.conj(state), target)) ** 2
    elif len(state.shape) == 2 and len(target.shape) == 2:
        fid = np.trace(np.dot(state, target))

    return fid


def process_fidelity(channel, target=None):
    """Process fidelity between two quantum channels (when at least one channel is` unitary),

    ..math::
        F_{pro}(\\mathcal{E}, \\mathcal{U}) = \\frac{1}{d^{2}} \\, \\Tr(\\mathcal{E}^{\\dagger}, \\mathcal{U})

    Args:
        channel: quantum channel.
        target: quantum channel. If None, target is the Identity channel.

    Returns:
        Process fidelity between channels :math:`\\mathcal{E}` and target :math:`\\mathcal{U}`.

    """

    if target:
        if channel.shape != target.shape:
            raise TypeError(
                f"Channels must have the same dims, but {channel.shape} != {target.shape}"
            )
    d = channel.shape[0]
    if target is None:
        # With no target, return process fidelity with Identity channel
        return np.trace(channel) / d**2
    else:
        return np.trace(
            np.dot(np.conj(np.transpose(channel)), target)
        ) / d**2


def average_gate_fidelity(channel, target=None):
    """Average gate fidelity between two quantum channels (when at least one channel is unitary),

    ..math::
        F_{avg}(\\mathcal{E}, \\mathcal{U}) = \\frac{d * F_{pro}(\\mathcal{E}, \\mathcal{U}) + 1}{d + 1}

    where :math:`d` is the dimension of the channels and :math:`F_{pro}(\\mathcal{E}, \\mathcal{U})` is the
    :meth:`~qibo.metrics.process_fidelily` of channel :math:`\\mathcal{E}` with respect to the unitary
    channel :math:`\\mathcal{U}`.

    Args:
        channel: quantum channel.
        target: quantum channel. If None, target is the Identity channel.

    Returns:
        Process fidelity between channel :math:`\\mathcal{E}` and target unitary channel :math:`\\mathcal{U}`.

    """

    d = channel.shape[0]
    return (d * process_fidelity(channel, target) + 1) / (d + 1)
