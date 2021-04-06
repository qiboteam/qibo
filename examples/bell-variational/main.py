import numpy as np
from qibo.optimizers import optimize
from functions import set_parametrized_circuits, cost_function, compute_chsh
import argparse


def main(nshots):
    '''Variationally find a maximally entangled state and the correct measurement angle
       for violation of Bell inequalities.
    Args:
        nshots: number of shots to use for the minimization.
        
    '''
    initial_parameters = np.random.uniform(0, 2*np.pi, 2)
    circuits = set_parametrized_circuits()
    best, params, _ = optimize(cost_function, initial_parameters, args=(circuits, nshots))
    print(f'Cost: {best}\n')
    print(f'Parameters: {params}\n')
    frequencies = []
    for circuit in circuits:
        circuit.set_parameters(params)
        frequencies.append(circuit(nshots=nshots).frequencies())
    chsh = compute_chsh(frequencies)
    print(f'CHSH inequality value: {chsh}\n')
    print(f'Target: {np.sqrt(2)*2}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nshots", default=10000, type=int, help='Number of shots for each circuit base')
    args = vars(parser.parse_args())
    main(**args)
