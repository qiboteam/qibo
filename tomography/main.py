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


def main(index, plot):
    # Extract state data
    beta = utils.find_beta("data/states_181120.json")
    gate = np.array(utils.matrices.gate(beta))

    # Extract tomography amplitudes
    data = utils.extract_data("data/" + filename[index])
    amp = np.sqrt([v[0] ** 2 + v[1] ** 2 for v in data.values()])

    # Linear calculation method
    allsets = [
        ([0, 1, 2, 15], (0, 1, 2, 3), (0, 1, 2, 3)),
        ([11, 12, 13, 14], (1, 0, 3, 2), (0, 1, 2, 3)),
        ([3, 6, 7, 10], (2, 3, 0, 1), (0, 1, 2, 3)),
        ([4, 5, 8, 9], (3, 2, 1, 0), (0, 1, 2, 3))
    ]
    dtype = complex
    rho_linear = np.zeros((4,4), dtype=dtype)
    for i, (seti, seta, setb) in enumerate(allsets):
        setA = gate[seti][:, seta, setb]
        minus = sum(np.dot(gate[seti][:, sa, sb], rho_linear[sb, sa])
                    for _, sa, sb in allsets[:i])
        setB = amp[seti].astype(dtype) - minus
        rho_linear[setb, seta] = solve(setA, setB)


    # Maximum likelihood estimation
    T_linear = utils.cholesky(rho_linear)
    t_guess = utils.to_vector(T_linear)
    res = minimize(utils.MLE, t_guess , args = (amp, gate), tol = 1e-9)
    print("Convergence:", res.success)
    T = utils.from_vector(res.x)
    rho_fit = np.array(T.H*T/np.trace(T.H*T))

    if plot:
        from plot import plot
        plot()
    else:
        r = sqrtm(utils.matrices.rho_th_plot(index))
        fidelity = abs(np.trace(sqrtm(r @ rho_fit @ r))) * 100
        print("Fidelity:", fidelity)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
