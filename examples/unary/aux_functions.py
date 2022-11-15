import numpy as np
from scipy.special import erfinv


def log_normal(x, mu, sig):
    """Lognormal probability distribution function normalized for representation in finite intervals.
    Args:
        x (np.array): variable of the probability distribution.
        mu (real): mean of the lognormal distribution.
        sig (real): sigma of the lognormal distribution.

    Returns:
        f (np.array): normalized probability distribution for the intervals of x.
    """
    dx = x[1] - x[0]
    log_norm = (
        1
        / (x * sig * np.sqrt(2 * np.pi))
        * np.exp(-np.power(np.log(x) - mu, 2.0) / (2 * np.power(sig, 2.0)))
    )
    f = log_norm * dx / (np.sum(log_norm * dx))
    return f


def classical_payoff(S0, sig, r, T, K, samples=1000000):
    """Function computing the payoff classically given some data.
    Args:
        S0 (real): initial asset price.
        sig (real): market volatility.
        r (real): market rate.
        T (real): maturity time.
        K (real): strike price.
        samples (int): total precision of the classical calculation.

    Returns:
        cl_payoff (real): classically computed payoff.
    """
    mu = (r - 0.5 * sig**2) * T + np.log(
        S0
    )  # Define all the parameters to be used in the computation
    mean = np.exp(
        mu + 0.5 * T * sig**2
    )  # Set the relevant zone of study and create the mapping between qubit and option price, and
    # generate the target lognormal distribution within the interval
    variance = (np.exp(T * sig**2) - 1) * np.exp(2 * mu + T * sig**2)
    Sp = np.linspace(
        max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), samples
    )
    lnp = log_normal(Sp, mu, sig * np.sqrt(T))
    cl_payoff = 0
    for i in range(len(Sp)):
        if K < Sp[i]:
            cl_payoff += lnp[i] * (Sp[i] - K)
    return cl_payoff


def get_theta(m_s, ones_s, zeroes_s, alpha=0.05):
    """Function to extract measurement values of a in an iterative AE algorithm.
    Args:
        m_s (list): set of m to be used.
        ones_s (list): Number of outcomes with 1, as many as m_s.
        zeroes_s (real): Number of outcomes with 0, as many as m_s.
        alpha (real): Confidence level.

    Returns:
        theta_s (list): results for the angle estimation
        err_theta_s (list): errors on the angle estimation
    """
    z = erfinv(1 - alpha / 2)
    ones_s = np.array(ones_s)
    zeroes_s = np.array(zeroes_s)
    valid_s = ones_s + zeroes_s
    a_s = ones_s / valid_s
    theta_s = np.zeros(len(m_s))
    err_theta_s = np.zeros(len(m_s))
    if m_s[0] == 0:
        theta_s[0] = np.arcsin(np.sqrt(a_s[0]))
        err_theta_s[0] = z / 2 / np.sqrt(valid_s[0])
    else:
        raise ValueError("AE does not start with m=0")
    for j, m in enumerate(m_s[1:]):
        aux_theta = np.arcsin(np.sqrt(a_s[j + 1]))
        theta = (
            [aux_theta]
            + [np.pi * k + aux_theta for k in range(1, m + 1, 1)]
            + [np.pi * k - aux_theta for k in range(1, m + 1, 1)]
        )
        theta = np.array(theta) / (2 * m + 1)
        arg = np.argmin(np.abs(theta - theta_s[j]))
        new_theta = theta[arg]
        new_error_theta = z / 2 / np.sqrt(valid_s[j + 1]) / (2 * m + 1)
        theta_s[j + 1] = (
            theta_s[j] / err_theta_s[j] ** 2 + new_theta / new_error_theta**2
        ) / (1 / err_theta_s[j] ** 2 + 1 / new_error_theta**2)
        err_theta_s[j + 1] = (1 / err_theta_s[j] ** 2 + 1 / new_error_theta**2) ** (
            -1 / 2
        )
    return theta_s, err_theta_s
