import numpy as np

from qibo.config import MAX_ITERATIONS, PRECISION_TOL, raise_error


def random_gaussian_matrix(dims: int, rank: int = None):
    """Generate a random Gaussian Unitary Matrix.

    Gaussian unitary matrices

    """

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

    matrix = np.random.normal(size=dims) + 1.0j * np.random.normal(size=dims)

    return matrix


def random_hermitian_operator(
    dims: int, semidefinite: bool = False, normalize: bool = False
):
    """."""

    if dims <= 0:
        raise_error(ValueError, f"dims ({dims}) must be type int and positive.")
    if not isinstance(semidefinite, bool) or not isinstance(normalize, bool):
        raise_error(TypeError, f"semidefinite and normalize must be type bool.")

    operator = random_gaussian_matrix(dims, dims)
    if semidefinite:
        operator = np.dot(np.transpose(np.conj(operator)), operator)
    else:
        operator = (operator + np.transpose(np.conj(operator))) / 2

    if normalize:
        operator = operator / np.linalg.norm(operator)  # / np.trace(operator)

    return operator


def random_unitary(dims: int, measure: str = "haar"):
    """."""

    if dims <= 0:
        raise_error(ValueError, f"dims must be type int and positive.")

    if measure is not None:
        if not isinstance(measure, str):
            raise_error(
                TypeError, f"measure must be type str but it is type {type(measure)}."
            )
        if measure != "haar":
            raise_error(ValueError, f"measure {measure} not implemented.")

    if measure == "haar":
        gaussian_matrix = random_gaussian_matrix(dims, dims)

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


def random_statevector(dims: int):
    """."""

    if dims <= 0:
        raise_error(ValueError, "dim must be of type int and >= 1")

    probabilities = np.random.rand(dims)
    probabilities = probabilities / np.sum(probabilities)
    phases = 2 * np.pi * np.random.rand(dims)

    return np.sqrt(probabilities) * np.exp(1.0j * phases)


def random_density_matrix(
    dims, rank: int = None, pure: bool = False, method: str = "Hilbert-Schmidt"
):
    """."""

    if dims <= 0:
        raise_error(ValueError, f"dims must be type int and positive.")
    if rank is not None and rank > dims:
        raise_error(ValueError, f"rank ({rank}) cannot be greater than dims ({dims}).")
    if rank is not None and rank <= 0:
        raise_error(ValueError, f"rank ({rank}) must be an int between 1 and dims.")
    if not isinstance(pure, bool):
        raise_error(TypeError, f"pure must be type bool, but it is type {type(pure)}.")
    if not isinstance(method, str):
        raise_error(
            TypeError, f"method must be type str, but it is type {type(method)}."
        )

    if pure:
        state = random_statevector(dims)
        state = np.outer(state, np.transpose(np.conj(state)))
    else:
        if method == "Hilbert-Schmidt":
            state = random_gaussian_matrix(dims, rank)
            state = np.dot(state, np.transpose(np.conj(state)))
            state = state / np.trace(state)
        elif method == "Bures":
            state = np.eye(dims) + random_unitary(dims)
            state = np.dot(state, random_gaussian_matrix(dims, rank))
            state = np.dot(state, np.transpose(np.conj(state)))
            state = state / np.trace(state)
        else:
            raise_error(ValueError, f"method {method} not found.")

    return state


def stochastic_matrix(
    dims: int,
    bistochastic: bool = False,
    precision_tol: float = None,
    max_iterations: int = None,
):
    """."""
    if dims <= 0:
        raise_error(ValueError, f"dims must be type int and positive.")
    if not isinstance(bistochastic, bool):
        raise_error(
            TypeError,
            f"bistochastic must be type bool, but it is type {type(bistochastic)}.",
        )
    if precision_tol is not None:
        if not isinstance(precision_tol, float):
            raise_error(
                TypeError,
                f"precision_tol must be type float, but it is type {type(precision_tol)}.",
            )
        if precision_tol < 0.0:
            raise_error(ValueError, f"precision_tol must be non-negative.")
    if max_iterations is not None:
        if not isinstance(max_iterations, int):
            raise_error(
                TypeError,
                f"max_iterations must be type int, but it is type {type(precision_tol)}.",
            )
        if max_iterations <= 0.0:
            raise_error(ValueError, f"max_iterations must be a positive int.")

    if precision_tol is None:
        precision_tol = PRECISION_TOL
    if max_iterations is None:
        max_iterations = MAX_ITERATIONS

    matrix = np.random.rand(dims, dims)
    row_sum = matrix.sum(axis=1)

    if bistochastic:
        column_sum = matrix.sum(axis=0)
        count = 0
        while count <= max_iterations - 1 and (
            (
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
        if count == max_iterations:
            import warnings

            warnings.warn("Reached max iterations.", RuntimeWarning)
    else:
        matrix = matrix / np.outer(row_sum, [1] * dims)

    return matrix


# %%
m = random_gaussian_matrix(4)
np.linalg.norm(np.linalg.inv(m) - m.T.conj())
