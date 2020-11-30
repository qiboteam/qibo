# -*- coding: utf-8 -*-
import argparse
import numpy as np
import utils
from scipy.linalg import sqrtm, solve
from scipy.optimize import minimize

filename = ["tomo_181120-00.json",
            "tomo_181120-01.json",
            "tomo_181120-10.json",
            "tomo_181120-11.json",
            "tomo_181120-bell.json",
            "tomo_181120-hadamard-tunable.json",
            "tomo_181120-hadamard-fixed.json",
            "tomo_181120-bell_beta.json"]

parser = argparse.ArgumentParser()
parser.add_argument("--index", default=0, type=int)
parser.add_argument("--plot", action="store_true")


def find_beta(states):
    """Finds beta from state data."""
    refer_A = np.matrix([[1, 1, 1, 1],
                        [1, 1, -1, -1],
                        [1, -1, 1, -1],
                        [1, -1, -1, 1]])
    refer_B = np.array([states["00"], states["01"], states["10"], states["11"]])
    beta = solve(refer_A, refer_B)
    return np.array(beta).flatten()


def linear_estimation(gate, amp, sets):
    """Linear estimation of the density matrix."""
    dtype = gate.dtype
    rho_linear = np.zeros_like(gate[0])
    for i, (seti, seta, setb) in enumerate(sets):
        setA = gate[seti][:, seta, setb]
        minus = sum(np.dot(gate[seti][:, sa, sb], rho_linear[sb, sa])
                    for _, sa, sb in sets[:i])
        setB = amp[seti].astype(dtype) - minus
        rho_linear[setb, seta] = solve(setA, setB)
    return rho_linear


def fit_mle(rho_guess, gate, amp, tol=1e-9):
    """Fits density matrix to minimize MLE."""
    t_guess = utils.cholesky(rho_guess)
    t_guess = utils.to_vector(t_guess)
    res = minimize(utils.MLE, t_guess, args=(amp, gate), tol=tol)
    t_fit = utils.from_vector(res.x)
    rho_fit = np.array(t_fit.H * t_fit / np.trace(t_fit.H * t_fit))
    return rho_fit, res


def main(index, plot):
    # Extract state data and define ``gate``
    states = utils.extract_data("data/states_181120.json")
    states = {k: np.sqrt(v[0] ** 2 + v[1] ** 2) for k, v in states.items()}
    beta = find_beta(states)
    gate = np.array(utils.matrices.gate(beta))

    # Extract tomography amplitudes
    data = utils.extract_data("data/" + filename[index])
    amp = np.sqrt([v[0] ** 2 + v[1] ** 2 for v in data.values()])

    # Linear calculation method
    sets = [
        ([0, 1, 2, 15], (0, 1, 2, 3), (0, 1, 2, 3)),
        ([11, 12, 13, 14], (1, 0, 3, 2), (0, 1, 2, 3)),
        ([3, 6, 7, 10], (2, 3, 0, 1), (0, 1, 2, 3)),
        ([4, 5, 8, 9], (3, 2, 1, 0), (0, 1, 2, 3))
    ]
    rho_linear = linear_estimation(gate, amp, sets)

    # Maximum likelihood estimation
    rho_fit, res = fit_mle(rho_linear, gate, amp)

    rho_theory = utils.matrices.rho_theory(index)
    sqrt_rho_theory = sqrtm(rho_theory)
    fidelity = abs(np.trace(sqrtm(sqrt_rho_theory @ rho_fit @ sqrt_rho_theory))) * 100
    print("Convergence:", res.success)
    print("Fidelity:", fidelity)

    if plot:
        from plot import plot
        plot(rho_theory, rho_linear, rho_fit, fidelity, res.success)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
