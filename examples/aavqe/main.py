import numpy as np
from qibo import gates, models, hamiltonians
import argparse

def main(nqubits, nlayers, maxsteps, T_max):
    circuit = models.Circuit(nqubits)
    for l in range(nlayers):
        circuit.add((gates.RY(q, theta=0) for q in range(nqubits)))
        circuit.add((gates.CZ(q, q + 1) for q in range(0, nqubits - 1, 2)))
        circuit.add((gates.RY(q, theta=0) for q in range(nqubits)))
        circuit.add((gates.CZ(q, q + 1) for q in range(1, nqubits - 2, 2)))
        circuit.add(gates.CZ(0, nqubits - 1))
    circuit.add((gates.RY(q, theta=0) for q in range(nqubits)))
    problem_hamiltonian = hamiltonians.XXZ(nqubits)
    easy_hamiltonian = hamiltonians.X(nqubits)
    s = lambda t: t
    aavqe = models.variational.AAVQE(circuit, easy_hamiltonian, problem_hamiltonian, s, nsteps=maxsteps, t_max=T_max)

    initial_parameters = np.random.uniform(0, 2 * np.pi*0.1,
                                           2 * nqubits * nlayers + nqubits)
    best, params = aavqe.minimize(initial_parameters)

    print('Final parameters: ', params)
    print('Final energy: ', best)

    #We compute the difference from the exact value to check performance
    eigenvalue = problem_hamiltonian.eigenvalues()
    print('Difference from exact value: ',best - eigenvalue[0].real)
    print('Log difference: ',-np.log10(best - eigenvalue[0].real))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", default=6, type=int)
    parser.add_argument("--nlayers", default=5, type=int)
    parser.add_argument("--maxsteps", default=10, type=int)
    parser.add_argument("--T_max", default=5, type=int)
    args = parser.parse_args()
    main(**vars(args))
