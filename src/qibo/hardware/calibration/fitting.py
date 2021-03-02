import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

def linear(x,*p) :
    return x*p[0] + p[1]

def exp(x,*p) :
    return p[0] + p[1]*np.exp(x*p[2])

def base_exp(x,*p) :
    return p[0] + p[1] * np.exp(-x / p[2]) * (x >= 0)

def base_exp_2(x,*p) :
    return p[0] + p[1] * np.exp(-x / p[2]) * (x >= 0) + p[3] * np.exp(-x / p[4]) * (x>=0)

def lorentzian(x,*p) :
    # A lorentzian peak with:
    #   Constant Background          : p[0]
    #   Peak height above background : p[1]
    #   Central value                : p[2]
    #   Full Width at Half Maximum   : p[3]
    return p[0] + (p[1] / np.pi) / (1.0 + ((x - p[2]) / p[3])**2)

def gaussian(x,*p) :
    # A gaussian peak with:
    #   Constant Background          : p[0]
    #   Peak height above background : p[1]
    #   Central value                : p[2]
    #   Standard deviation           : p[3]
    return p[0] + p[1] * np.exp(-1 * (x - p[2])**2 / (2 * p[3]**2))

def rabi(x, *p) :
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   Period    T                  : p[2]
    #   Phase                        : p[3]
    #   Arbitrary parameter T_2      : p[4]
    return p[0] + p[1] * np.sin(2 * np.pi / p[2] * x + p[3]) * np.exp(-x / p[4])

def sin(x, *p) :
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   Period    T                  : p[2]
    #   Phase                        : p[3]
    return p[0] + p[1] * np.sin(2 * np.pi / p[2] * x + p[3])

def fit_rabi(amp_array, time_array):
    pguess = [
        np.mean(amp_array),
        max(amp_array) - min(amp_array),
        35e-9,
        np.pi/2,
        0.1e-6
    ]
    popt, pcov = curve_fit(rabi, time_array, amp_array, p0=pguess)
    pi_pulse_duration = popt[2] / 2
    return pi_pulse_duration

def fit_spinecho(amp_array, time_array):
    pguess = [
        min(amp_array),
        max(amp_array) - min(amp_array),
        -2e-6
    ]
    popt, pcov = curve_fit(exp, time_array, amp_array, p0=pguess)
    t2 = abs(popt[2])
    return t2

def fit_t1(amp_array, time_array):
    pguess = [
        max(amp_array),
        max(amp_array) - min(amp_array),
        5e-6
    ]
    popt, pcov = curve_fit(exp, time_array, amp_array, p0=pguess)
    t1 = abs(popt[2])
    return t1

def fit_pulse(amp_array, freq_array):
    y = amp_array
    noise = np.mean(np.abs(np.diff(y)))/np.mean(y)/(max(y)-min(y))

    # p = (int(len(y)/10) // 2 )* 2 + 1
    p = 20 * int(noise) + 1

    if p > len(amp_array):
        raise Exception("Data is too noisy")

    ys = savgol_filter(y, p, 2)
    return freq_array[np.argmin(ys)]
