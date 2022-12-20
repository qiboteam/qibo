import argparse

import numpy as np
from functions import compute_chsh, cost_function, set_parametrized_circuits

from qibo import set_backend
from qibo.optimizers import optimize


def main(nshots, backend):
    """Variationally find a maximally entangled state and the correct measurement angle
       for violation of Bell inequalities.
    Args:
        nshots: number of shots to use for the minimization.
        backend: choice of backend to run the example in.

    """
    set_backend(backend)
    initial_parameters = np.random.uniform(0, 2 * np.pi, 2)
    circuits = set_parametrized_circuits()
    best, params, _ = optimize(
        cost_function, initial_parameters, args=(circuits, nshots)
    )
    print(f"Cost: {best}\n")
    print(f"Parameters: {params}\n")
    print(f"Angles for the RY gates: {(params*180/np.pi)%360} in degrees.\n")
    frequencies = []
    for circuit in circuits:
        circuit.set_parameters(params)
        frequencies.append(circuit(nshots=nshots).frequencies())
    chsh = compute_chsh(frequencies, nshots)
    print(f"CHSH inequality value: {chsh}\n")
    print(f"Target: {np.sqrt(2)*2}\n")
    print(f"Relative distance: {100*np.abs(np.abs(chsh)-np.sqrt(2)*2)/np.sqrt(2)*2}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nshots",
        default=10000,
        type=int,
        help="Number of shots for each circuit base",
    )
    parser.add_argument(
        "--backend", default="qibojit", type=str, help="Backend to use for the example"
    )
    args = vars(parser.parse_args())
    main(**args)
