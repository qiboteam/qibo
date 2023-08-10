import matplotlib.pyplot as plt
import numpy as np

from qibo import Circuit, gates
from qibo.models.iqae import IQAE


def fourier_series(coeffs, x_val, period=2 * np.pi):
    """Compute the Fourier series for a given set of coefficients in exponencial form."""

    xval_len = len(x_val)
    x_vals = x_val.reshape(len(x_val))
    n = 0
    series = coeffs[0] * np.exp(1j * n * x_vals * (2 * np.pi / period))
    for i in range(1, int((len(coeffs)) / 2) + 1):
        n += 1
        serie_i = coeffs[i] * np.exp(1j * n * x_vals * (2 * np.pi / period))
        series += serie_i + np.conjugate(serie_i)
    y_vals = np.real(series)
    return y_vals


def plot_data_fourier(function_test, xtest, y_fourier):
    """Plot the Fourier representation alongside the function test and provides the efficiency of the fit."""

    SSE = 0
    SST = 0
    average = np.sum(function_test) / function_test.size
    N = len(function_test)

    for i in range(N):
        SSE += (function_test[i] - y_fourier[i]) ** 2
        SST += (function_test[i] - average) ** 2

    R = 1 - SSE / SST
    R = round(float(R) * 100, 1)

    sort_indices = np.argsort(xtest)
    plt.plot(xtest[sort_indices], function_test[sort_indices], label="Target")
    plt.plot(
        xtest[sort_indices],
        y_fourier[sort_indices],
        label=f"Quantum Fourier \nAccuracy: {R}%",
    )

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend(loc="upper center")

    return plt, R


def exp_fourier_to_trig(fourier_coeffs):
    """Convert the Fourier coefficients from exponential form to trigonometric form."""

    n = len(fourier_coeffs)
    c0 = fourier_coeffs[0]
    b_coeffs = [2 * np.real(fourier_coeffs[i]) for i in range(1, int(n / 2) + 1)]
    a_coeffs = [-2 * np.imag(fourier_coeffs[i]) for i in range(1, int(n / 2) + 1)]
    return [c0] + b_coeffs + a_coeffs


def coeffs_array_to_class(fourier_coeffs):
    """Convert the Fourier coefficients from array to a Class to be dealt in an easier way."""

    fourier_class = []

    for i in range(len(fourier_coeffs)):
        fourier_class_term = Fourier_terms(
            fourier_coeffs[i, 2], int(fourier_coeffs[i, 1]), int(fourier_coeffs[i, 0])
        )
        fourier_class.append(fourier_class_term)
    return fourier_class


def trig_to_final_array(trig_coeffs):
    """Convert the array with the trigonometric coefficients into an array that tracks if a
    coefficient corresponds to a Cosine (0) or to a Sine (1)."""

    array_list = np.zeros(shape=(len(trig_coeffs), 3))
    array_list[0, 0] = 0  # 0 means Cos
    array_list[0, 1] = 0
    array_list[0, 2] = np.real(trig_coeffs[0])
    for i in range(1, int((len(trig_coeffs)) / 2) + 1):
        # cosine
        array_list[i, 0] = 0  # 0 means Cos
        array_list[i, 1] = i
        array_list[i, 2] = trig_coeffs[i]
    for i in range(int((len(trig_coeffs)) / 2) + 1, len(trig_coeffs)):
        # sine
        array_list[i, 0] = 1  # 1 means Sin
        array_list[i, 1] = i - (len(trig_coeffs) - 1) / 2
        array_list[i, 2] = trig_coeffs[i]
    return array_list


class Fourier_terms:
    """Class that contains all the information about a Fourier term"""

    def __init__(self, coefficient, angle, function):
        self.coeff = coefficient
        self.angle = angle
        self.function = function  # 0 cosine, 1 sine
