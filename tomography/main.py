# -*- coding: utf-8 -*-
import argparse
import json
import utils
import numpy as np
import tomography

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


def extract_data(filename):
    with open(filename,"r") as r:
        r = r.read()
        raw = json.loads(r)
    return {k: np.array(v) for k, v in raw.items()}


def main(index, plot):
    # Extract state data and define ``gate``
    states = extract_data("data/states_181120.json")
    states = {k: np.sqrt(v[0] ** 2 + v[1] ** 2) for k, v in states.items()}

    # Extract tomography amplitudes
    data = extract_data("data/" + filename[index])
    amp = np.sqrt([v[0] ** 2 + v[1] ** 2 for v in data.values()])

    # Create tomography object
    tom = tomography.Tomography(states, amp)
    tom.minimize()

    rho_theory = utils.matrices.rho_theory(index)
    fidelity = tom.fidelity(rho_theory)
    print("Convergence:", tom.success)
    print("Fidelity:", fidelity)

    if plot:
        from plot import plot
        plot(tom, rho_theory)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
