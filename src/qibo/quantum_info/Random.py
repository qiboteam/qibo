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

    if measure is not None and measure != "haar":
        raise_error(ValueError, f"measure {measure} not implemented.")

    if measure == "haar":
        gaussian_matrix = random_ginibre_unitary_matrix(dims, dims)

        Q, R = np.linalg.qr(gaussian_matrix)
        D = np.diag(R)
        D = D / np.abs(D)
        R = np.diag(D)
        unitary = np.dot(Q, R)
    elif measure is None:
        from qibo import get_backend

        if get_backend() == "qibojit (cupy)":
            raise_error(
                NotImplementedError,
                f"measure not implemented for the backend {get_backend()}.",
            )
        else:
            from scipy.linalg import expm

            matrix_1 = np.random.randn(dims, dims)
            matrix_2 = np.random.randn(dims, dims)
            H = (matrix_1 + np.transpose(matrix_1)) + 1.0j * (
                matrix_2 - np.transpose(matrix_2.T)
            )
            unitary = expm(-1.0j * H / 2)

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
            raise_error(ValueError, f"method {method} not found.")

    return state


def stochastic_matrix(
    dims: int,
    bistochastic: bool = False,
    max_iterations: int = None,
    precision_tol: float = None,
):
    """."""

    if precision_tol is not None and precision_tol < 0.0:
        raise_error(ValueError, f"precision_tol must be non-negative.")
    if max_iterations is not None and max_iterations <= 0.0:
        raise_error(ValueError, f"precision_tol must be non-negative.")

    if precision_tol is None:
        precision_tol = PRECISION_TOL
    if max_iterations is None:
        max_iterations = 20

    matrix = np.random.rand(dims, dims)
    row_sum = matrix.sum(axis=1)

    if bistochastic:
        column_sum = matrix.sum(axis=0)
        count = 0
        while (
            count <= max_iterations - 1
            and (
                np.any(row_sum >= 1 + precision_tol)
                or np.any(row_sum <= 1 - precision_tol)
            )
            or (
                np.any(column_sum >= 1 + precision_tol)
                or np.any(column_sum <= 1 - precision_tol)
            )
        ):
            matrix = matrix / matrix.sum(axis=0)
            matrix = matrix / matrix.sum(axis=1)[:, np.newaxis]
            row_sum = matrix.sum(axis=1)
            column_sum = matrix.sum(axis=0)
            count += 1
        if count == max_iterations - 1:
            import warnings

            warnings.warn("Reached max iterations.", RuntimeError)
    else:
        matrix = matrix / np.outer(row_sum, [1] * dims)

    return matrix
