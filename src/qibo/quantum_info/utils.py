import numpy as np

from qibo.config import PRECISION_TOL, raise_error

# Phases corresponding to all 24 single-qubit Clifford gates
ONEQUBIT_CLIFFORD_PARAMS = [
    (0, 0, 0, 0),
    (np.pi, 1, 0, 0),
    (np.pi, 0, 1, 0),
    (np.pi, 0, 0, 1),
    (np.pi / 2, 1, 0, 0),
    (-np.pi / 2, 1, 0, 0),
    (np.pi / 2, 0, 1, 0),
    (-np.pi / 2, 0, 1, 0),
    (np.pi / 2, 0, 0, 1),
    (-np.pi / 2, 0, 0, 1),
    (np.pi, 1 / np.sqrt(2), 1 / np.sqrt(2), 0),
    (np.pi, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)),
    (np.pi, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)),
    (np.pi, -1 / np.sqrt(2), 1 / np.sqrt(2), 0),
    (np.pi, 1 / np.sqrt(2), 0, -1 / np.sqrt(2)),
    (np.pi, 0, -1 / np.sqrt(2), 1 / np.sqrt(2)),
    (2 * np.pi / 3, 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
    (-2 * np.pi / 3, 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
    (2 * np.pi / 3, -1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
    (-2 * np.pi / 3, -1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
    (2 * np.pi / 3, 1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)),
    (-2 * np.pi / 3, 1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)),
    (2 * np.pi / 3, 1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)),
    (-2 * np.pi / 3, 1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)),
]


def shannon_entropy(probability_array, base: float = 2):
    """Calculate the Shannon entropy of a probability array :math:`\\mathbf{p}`, which is given by

    .. math::
        H(\\mathbf{p}) = - \\sum_{k = 0}^{d^{2} - 1} \\, p_{k} \\, \\log_{b}(p_{k}) \\, ,

    where :math:`d = \\text{dim}(\\mathcal{H})` is the dimension of the Hilbert space :math:`\\mathcal{H}`,
    :math:`b` is the log base (default 2), and :math:`0 \\log_{b}(0) \\equiv 0`.

    Args:
        probability_array: a probability array :math:`\\mathbf{p}`.
        base (float): the base of the log. Default: 2.

    Returns:
        (float): The Shannon entropy :math:`H(\\mathcal{p})`.

    """

    if base <= 0:
        raise_error(ValueError, "log base must be non-negative.")

    if len(probability_array.shape) != 1:
        raise_error(
            TypeError,
            f"Probability array must have dims (k,) but it has {probability_array.shape}.",
        )

    if len(probability_array) == 0:
        raise_error(TypeError, "Empty array.")

    if any(probability_array < 0) or any(probability_array > 1.0):
        raise_error(
            ValueError,
            "All elements of the probability array must be between 0. and 1..",
        )

    if (np.sum(probability_array) > 1.0 + PRECISION_TOL) or (
        np.sum(probability_array) < 1.0 - PRECISION_TOL
    ):
        raise_error(ValueError, "Probability array must sum to 1.")

    if base == 2:
        log_prob = np.where(probability_array != 0, np.log2(probability_array), 0.0)
    elif base == 10:
        log_prob = np.where(probability_array != 0, np.log10(probability_array), 0.0)
    elif base == np.e:
        log_prob = np.where(probability_array != 0, np.log(probability_array), 0.0)
    else:
        log_prob = np.where(
            probability_array != 0, np.log(probability_array) / np.log(base), 0.0
        )

    entropy = -np.sum(probability_array * log_prob)

    # absolute value if entropy == 0.0 to avoid returning -0.0
    entropy = np.abs(entropy) if entropy == 0.0 else entropy

    return entropy


def hellinger_distance(prob_dist_p, prob_dist_q, validate: bool = False):
    """Calculate the Hellinger distance :math:`H(p, q)` between
    two discrete probability distributions, :math:`\\mathbf{p}` and :math:`\\mathbf{q}`.
    It is defined as

    .. math::
        H(\\mathbf{p} \\, , \\, \\mathbf{q}) = \\frac{1}{\\sqrt{2}} \\, || \\sqrt{\\mathbf{p}} - \\sqrt{\\mathbf{q}} ||_{2}

    where :math:`||\\cdot||_{2}` is the Euclidean norm.

    Args:
        prob_dist_p: (discrete) probability distribution :math:`p`.
        prob_dist_q: (discrete) probability distribution :math:`q`.
        validate (bool): if True, checks if :math:`p` and :math:`q` are proper
            probability distributions. Default: False.

    Returns:
        (float): Hellinger distance :math:`H(p, q)`.

    """

    if (len(prob_dist_p.shape) != 1) or (len(prob_dist_q.shape) != 1):
        raise_error(
            TypeError,
            f"Probability arrays must have dims (k,) but have dims {prob_dist_p.shape} and {prob_dist_q.shape}.",
        )

    if (len(prob_dist_p) == 0) or (len(prob_dist_q) == 0):
        raise_error(TypeError, "At least one of the arrays is empty.")

    if validate:
        if (any(prob_dist_p < 0) or any(prob_dist_p > 1.0)) or (
            any(prob_dist_q < 0) or any(prob_dist_q > 1.0)
        ):
            raise_error(
                ValueError,
                "All elements of the probability array must be between 0. and 1..",
            )
        if (np.sum(prob_dist_p) > 1.0 + PRECISION_TOL) or (
            np.sum(prob_dist_p) < 1.0 - PRECISION_TOL
        ):
            raise_error(ValueError, "First probability array must sum to 1.")

        if (np.sum(prob_dist_q) > 1.0 + PRECISION_TOL) or (
            np.sum(prob_dist_q) < 1.0 - PRECISION_TOL
        ):
            raise_error(ValueError, "Second probability array must sum to 1.")

    return np.linalg.norm(np.sqrt(prob_dist_p) - np.sqrt(prob_dist_q)) / np.sqrt(2)


def hellinger_fidelity(prob_dist_p, prob_dist_q, validate: bool = False):
    """Calculate the Hellinger fidelity between two discrete
    probability distributions, :math:`p` and :math:`q`. The fidelity is
    defined as :math:`(1 - H^{2}(p, q))^{2}`, where :math:`H(p, q)`
    is the Hellinger distance.

    Args:
        prob_dist_p: (discrete) probability distribution :math:`p`.
        prob_dist_q: (discrete) probability distribution :math:`q`.
        validate (bool): if True, checks if :math:`p` and :math:`q` are proper
            probability distributions. Default: False.

    Returns:
        (float): Hellinger fidelity.

    """

    return (1 - hellinger_distance(prob_dist_p, prob_dist_q, validate) ** 2) ** 2
