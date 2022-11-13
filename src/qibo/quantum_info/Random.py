#%%
import numpy as np

from qibo.config import PRECISION_TOL, raise_error


def random_ginibre_unitary_matrix(dims: int, rank: int = None):
    """."""

    if dims <= 0:
        raise_error(ValueError, f"dims must be type int and positive.")

    if rank is None:
        rank = dims
    else:
        if rank > dims:
            raise_error(
                ValueError, f"rank ({rank}) cannot be greater than dims ({dims})."
            )
        elif rank <= 0:
            raise_error(ValueError, f"rank ({rank}) must be an int between 1 and dims.")

    dims = (dims, rank)

    matrix = np.random.normal(0, 1 / 2, size=dims) + 1.0j * np.random.normal(
        0, 1 / 2, size=dims
    )
    return matrix


def random_hermitian_operator(
    dims: int, semidefinite: bool = False, normalize: bool = False
):
    """."""

    if dims <= 0:
        raise_error(ValueError, f"dims ({dims}) must be type int and positive.")

    operator = random_ginibre_unitary_matrix(dims, dims)
    if semidefinite:
        operator = np.dot(np.transpose(np.conj(operator)), operator)
    else:
        operator = (operator + np.transpose(np.conj(operator))) / 2

    if normalize:
        operator = operator / np.trace(operator)

    return operator


def random_unitary(dims: int, measure: str = "haar"):
    """."""

    if dims <= 0:
        raise_error(ValueError, f"dims must be type int and positive.")

    if measure != "haar":
        raise_error(NotImplementedError, f"measure {measure} not implemented.")

    gaussian_matrix = random_ginibre_unitary_matrix(dims, dims)

    Q, R = np.linalg.qr(gaussian_matrix)
    D = np.diag(R)
    D = D / np.abs(D)
    R = np.diag(D)
    unitary = np.dot(Q, R)

    return unitary


def random_statevector(dim: int):
    """."""

    if dim <= 0:
        raise_error(ValueError, "dim must be of type int and >= 1")

    probabilities = np.random.rand(dim)
    probabilities = probabilities / np.sum(probabilities)
    phases = 2 * np.pi * np.random.rand(dim)

    return np.sqrt(probabilities) * np.exp(1.0j * phases)


def random_density_matrix(
    dims, rank: int = None, pure: bool = False, method: str = "Hilbert-Schmidt"
):
    """."""

    if dims <= 0:
        raise_error(ValueError, f"dims must be type int and positive.")
    if rank > dims:
        raise_error(ValueError, f"rank ({rank}) cannot be greater than dims ({dims}).")
    if rank <= 0:
        raise_error(ValueError, f"rank ({rank}) must be an int between 1 and dims.")

    if pure:
        state = random_statevector(dims)
        state = np.outer(state, np.transpose(np.conj(state)))
    else:
        if method == "Hilbert-Schmidt":
            state = random_ginibre_unitary_matrix(dims, rank)
            state = np.dot(state, np.transpose(np.conj(state)))
            state = state / np.trace(state)
        elif method == "Bures":
            state = np.eye(dims) + random_unitary(dims)
            state = np.dot(state, random_ginibre_unitary_matrix(dims, rank))
            state = np.dot(state, np.transpose(np.conj(state)))
            state = state / np.trace(state)
        else:
            raise_error(NotImplementedError, f"method {method} is not implemented.")

    return state
