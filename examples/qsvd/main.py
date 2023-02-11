#!/usr/bin/env python3
import argparse

import numpy as np
from qsvd import QSVD

parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=6, help="Number of qubits", type=int)
parser.add_argument(
    "--subsize",
    default=3,
    help="Subsize of the bipartition with qubits 0,1,...,subzize-1",
    type=int,
)
parser.add_argument(
    "--nlayers", default=5, help="Number of layers of the variational circuit", type=int
)
parser.add_argument(
    "--nshots",
    default=int(1e5),
    help="Number of shots used when sampling the circuit",
    type=int,
)
parser.add_argument(
    "--RY",
    action="store_true",
    help="Use Ry rotations or RxRzRx rotations in the ansatz",
)
parser.add_argument(
    "--method", default="Powell", help="Classical otimizer employed", type=str
)
parser.add_argument(
    "--maxiter", default=None, help="Maximum number of iterations.", type=int
)


def main(nqubits, subsize, nlayers, nshots, RY, method, maxiter):
    # We initialize the QSVD
    Qsvd = QSVD(nqubits, subsize, nlayers, RY=RY)

    # We choose an initial random state
    initial_state = np.random.uniform(-0.5, 0.5, 2**nqubits) + 1j * np.random.uniform(
        -0.5, 0.5, 2**nqubits
    )
    initial_state = initial_state / np.linalg.norm(initial_state)
    if nqubits <= 6:
        print("Initial random state: ", initial_state)

    # We compute the exact Schmidt coefficients and von Neuman entropy
    densmatrix = np.outer(initial_state, np.conjugate(initial_state))
    n = nqubits - subsize
    m = subsize
    if subsize <= nqubits / 2:
        reduced_dens = np.trace(
            densmatrix.reshape(2**n, 2**m, 2**n, 2**m), axis1=1, axis2=3
        )
    else:
        reduced_dens = np.trace(
            densmatrix.reshape(2**n, 2**m, 2**n, 2**m), axis1=0, axis2=2
        )

    schmidt = np.linalg.eigvalsh(reduced_dens)
    vneumann = -np.sum(schmidt[schmidt > 0] * np.log2(schmidt[schmidt > 0]))
    schmidt = -np.sort(-np.sqrt(np.abs(schmidt)))
    print("Exact Schmidt coefficients: ", schmidt)
    print("Exact von Neumann entropy: ", vneumann)

    # We choose initial random parameters
    if not RY:  # if Rx,Rz,Rx rotations are employed in the anstaz
        initial_parameters = (
            2 * np.pi * np.random.rand(6 * nqubits * nlayers + 3 * nqubits)
        )

    else:  # if Ry rotations are employed in the anstaz
        initial_parameters = 2 * np.pi * np.random.rand(2 * nqubits * nlayers + nqubits)

    # We train the QSVD
    print("Training QSVD...")
    cost_function, optimal_angles = Qsvd.minimize(
        initial_parameters,
        init_state=initial_state,
        nshots=nshots,
        method=method,
        maxiter=maxiter,
    )

    # We use the optimal angles to compute the Schmidt coefficients of the bipartion
    Schmidt_coefficients = Qsvd.Schmidt_coeff(optimal_angles, initial_state)
    print("QSVD Schmidt coefficients: ", Schmidt_coefficients)

    # We compute the von Neumann entropy using the Schmidt coefficients
    VonNeumann_entropy = Qsvd.VonNeumann_entropy(optimal_angles, initial_state)
    print("QSVD von Neumann entropy: ", VonNeumann_entropy)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
