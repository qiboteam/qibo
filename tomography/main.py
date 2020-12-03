# -*- coding: utf-8 -*-
import argparse
import numpy as np
import tomography
from data import data

parser = argparse.ArgumentParser()
parser.add_argument("--index", default=0, type=int)
parser.add_argument("--plot", action="store_true")


def main(index, plot):
    # Extract state data and define ``gate``
    states = data.extract(data.statefile)
    states = {k: np.sqrt(v[0] ** 2 + v[1] ** 2) for k, v in states.items()}

    # Extract tomography amplitudes
    filename, circuit = data.measurementfiles[index]
    amp = data.extract(filename)
    amp = np.sqrt([v[0] ** 2 + v[1] ** 2 for v in amp.values()])

    # Create tomography object
    tom = tomography.Tomography(states, amp)
    tom.minimize()

    rho_theory = circuit().numpy()
    fidelity = tom.fidelity(rho_theory)
    print("Convergence:", tom.success)
    print("Fidelity:", fidelity)

    if plot:
        from plot import plot
        plot(tom, rho_theory)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
