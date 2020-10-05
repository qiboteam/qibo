import argparse
import time
import numpy as np
from qibo.config import DTYPES, K
from qibo import hamiltonians, models, matrices


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=4, type=int)
parser.add_argument("--dt", default=1e-2, type=float)
parser.add_argument("--T", default=10.0, type=float)
parser.add_argument("--efficient", action="store_true")


def efficient_x(h1):
    nqubits = h1.nqubits
    def ground_state():
        n = K.cast(2 ** nqubits, dtype=DTYPES.get('DTYPEINT'))
        state = K.ones(n, dtype=DTYPES.get('DTYPECPX'))
        return state / K.math.sqrt(K.cast(n, dtype=state.dtype))

    term = np.kron(matrices.X, matrices.I)
    term = hamiltonians.Hamiltonian(2, term, numpy=True)
    parts = [{k: term for k in part.keys()} for part in h1.parts]
    return hamiltonians.TrotterHamiltonian(*parts, ground_state=ground_state)


def print_parts(h):
    for part in h.parts:
        for k, v in part.items():
            print(k, v)


def main(nqubits, dt, T, efficient):
    h1 = hamiltonians.TFIM(nqubits=nqubits, h=1.0, trotter=True)
    if efficient:
        h0 = efficient_x(h1)
    else:
        h0 = hamiltonians.X(nqubits=nqubits, trotter=True)

    print_parts(h0)
    print()
    print_parts(h1)

    evolve = models.AdiabaticEvolution(h0, h1, lambda t: t, dt)

    print("\n\n\n")
    print_parts(evolve.h0)
    print()
    print_parts(evolve.h1)

    final_state = evolve(final_time=T)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    start_time = time.time()
    main(**args)
    print("Time:", time.time() - start_time)