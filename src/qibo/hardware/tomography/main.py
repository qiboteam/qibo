# -*- coding: utf-8 -*-
import argparse
import json
import numpy as np
from qibo.hardware import tomography


def rho_theory(i):
    rho = np.zeros((4, 4), dtype=complex)
    rho[i, i] = 1
    return rho


def extract(filename):
    with open(filename,"r") as r:
        raw = json.loads(r.read())
    return raw


state_file = "./data/states_181120.json"

measurement_files = [("./data/tomo_181120-00.json", rho_theory(0)),
                     ("./data/tomo_181120-01.json", rho_theory(1)),
                     ("./data/tomo_181120-10.json", rho_theory(2)),
                     ("./data/tomo_181120-11.json", rho_theory(3))]


parser = argparse.ArgumentParser()
parser.add_argument("--index", default=0, type=int)
parser.add_argument("--plot", action="store_true")


def main(index, plot):
    """Perform tomography to estimate density matrix from measurements.

    Args:
        index (int): Which experimental json file to use.
            See ``measurement_files`` list defined above for available files.
        plot (bool): Plot histograms comparing the estimated density matrices
            with the theoretical ones. If ``False`` only the fidelity is
            calculated.
    """
    # Extract state data and define ``gate``
    state = extract(state_file)
    state = np.stack(list(state.values()))
    state = np.sqrt((state ** 2).sum(axis=1))

    # Extract tomography amplitudes
    filename, rho_theory = measurement_files[index]
    amp = extract(filename)
    amp = np.stack(list(amp.values()))
    amp = np.sqrt((amp ** 2).sum(axis=1))

    # Create tomography object
    tom = tomography.Tomography(amp, state)
    # Optimize denisty matrix by minimizing MLE
    tom.minimize()

    fidelity = tom.fidelity(rho_theory)
    print("Convergence:", tom.success)
    print("Fidelity:", fidelity)

    if plot:
        from plot import plot # pylint: disable=import-error
        plot(tom, rho_theory)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
