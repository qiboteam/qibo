import numpy as np
from scipy.special import erfinv
import matplotlib.pyplot as plt

def name_folder_data(data):
    """
    Auxiliary function to create names in a fast way
    :param data: Data to create name (S0, sig, r, T, K)
    :return: name
    """
    string = 'results/S0(%s)_sig(%s)_r(%s)_T(%s)_K(%s)' % data
    return string

def log_normal(x, mu, sig):
    """
    Lognormal probability distribution function normalized for representation in finite intervals
    :param x: variable of the probability distribution
    :param mu: variable mu
    :param sig: variable sigma
    :return: Probability distribution for x according to log_normal(mu, sig)
    """
    dx = x[1]-x[0]
    log_norm = 1 / (x * sig * np.sqrt(2 * np.pi)) * np.exp(- np.power(np.log(x) - mu, 2.) / (2 * np.power(sig, 2.)))
    f = log_norm*dx/(np.sum(log_norm * dx))
    return f

def classical_payoff(S0, sig, r, T, K, samples=10000):
    """
    Function computing the payoff classically given some data.
    :param S0: initial price
    :param sig: volatilities
    :param r: interest rate
    :param T: maturity date
    :param K: strike
    :return: classical payoff
    """
    mu = (r - 0.5 * sig ** 2) * T + np.log(S0)  # Define all the parameters to be used in the computation
    mean = np.exp(mu + 0.5 * T * sig ** 2)  # Set the relevant zone of study and create the mapping between qubit and option price, and
    # generate the target lognormal distribution within the interval
    variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)
    Sp = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), samples)
    lnp = log_normal(Sp, mu, sig * np.sqrt(T))
    cl_payoff = 0
    for i in range(len(Sp)):
        if K < Sp[i]:
            cl_payoff += lnp[i] * (Sp[i] - K)

    return cl_payoff

def KL(p, q):
    """
    Function to compute Kullback-Leibler divergence
    :param p: Probability distribution
    :param q: Target probability distribution
    :return:
    """
    return np.sum(p * np.log(p / q))


def get_theta(m_s, ones_s, zeroes_s, alpha=0.05):
    """
    Function to extract measurement values of a in an iterative AE algorithm
    :param m_s: Set of m
    :param ones_s: Number of outcomes with 1, as many as m_s
    :param zeroes_s: Number of outcomes with 0, as many as m_s
    :param alpha: Confidence level.
    :return: Results and uncertainties for theta
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
        raise ValueError('AE does not start with m=0')

    for j, m in enumerate(m_s[1:]):
        aux_theta = np.arcsin(np.sqrt(a_s[j + 1]))
        theta = [aux_theta] + [np.pi * k + aux_theta for k in range(1, m + 1, 1)] + \
                [np.pi * k - aux_theta for k in range(1, m + 1, 1)]
        theta = np.array(theta) / (2 * m + 1)
        arg = np.argmin(np.abs(theta - theta_s[j]))
        new_theta = theta[arg]
        new_error_theta = z / 2 / np.sqrt(valid_s[j + 1]) / (2 * m + 1)

        theta_s[j + 1] = (theta_s[j] / err_theta_s[j] ** 2 + new_theta / new_error_theta**2 ) /\
                         (1 / err_theta_s[j] ** 2 + 1 / new_error_theta**2 )
        err_theta_s[j + 1] = (1 / err_theta_s[j] ** 2 + 1 / new_error_theta**2 )**(-1/2)


    return theta_s, err_theta_s



def experimental_data(data, conf):
    """
    Function for computing weighted averages and confidences of data
    :param data: Measured values
    :param conf: Confidences of data
    :return: (weighted average, weighted error), (mean of errors, confidence of errors)
    """
    conf_mean = np.mean(conf)
    conf_std = np.std(conf)

    return errors_experiment(data, conf), (conf_mean, conf_std)

def errors_experiment(data, conf):
    """
    Function for computing weighted averages
    :param data: Measured values
    :param conf: Confidences of data
    :return: (weighted average, weighted error)
    """

    mean = np.sum(data / conf ** 2) / np.sum(1 / conf ** 2)
    error = np.sum(1 / conf ** 2) ** (-1 / 2)
    error = max(error, np.max(conf))

    return mean, error

