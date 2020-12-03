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
    state = data.extract(data.statefile)
    state = np.stack(list(state.values()))
    state = np.sqrt((state ** 2).sum(axis=1))

    # Extract tomography amplitudes
    filename, circuit = data.measurementfiles[index]
    amp = data.extract(filename)
    amp = np.stack(list(amp.values()))
    amp = np.sqrt((amp ** 2).sum(axis=1))

    # Create tomography object
    tom = tomography.Tomography(amp, state)
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
