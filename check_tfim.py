import argparse
import time
import numpy as np
from qibo.config import DTYPES, K
from qibo import hamiltonians, models, matrices


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=4, type=int)
parser.add_argument("--dt", default=1e-2, type=float)
parser.add_argument("--T", default=10.0, type=float)


def efficient_x(h1):
    nqubits = h1.nqubits
    def ground_state():
        n = K.cast(2 ** nqubits, dtype=DTYPES.get('DTYPEINT'))
        state = K.ones(n, dtype=DTYPES.get('DTYPECPX'))
        return state / K.math.sqrt(K.cast(n, dtype=state.dtype))

    term = - np.kron(matrices.X, matrices.I)
    term = hamiltonians.Hamiltonian(2, term, numpy=True)
    parts = [{k: term for k in part.keys()} for part in h1.parts]
    return hamiltonians.TrotterHamiltonian(*parts, ground_state=ground_state)


def main(nqubits, dt, T):
    h0 = hamiltonians.X(nqubits=nqubits, trotter=True)
    h1 = hamiltonians.TFIM(nqubits=nqubits, h=1.0, trotter=True)
    evolve = models.AdiabaticEvolution(h0, h1, lambda t: t, dt)
    final_state = evolve(final_time=T)

    h1 = hamiltonians.TFIM(nqubits=nqubits, h=1.0, trotter=True)
    h0 = efficient_x(h1)
    evolve_eff = models.AdiabaticEvolution(h0, h1, lambda t: t, dt)
    target_state = evolve_eff(final_time=T)

    target_h0 = hamiltonians.X(nqubits=nqubits, numpy=True).matrix
    target_h1 = hamiltonians.TFIM(nqubits=nqubits, h=1.0, numpy=True).matrix
    np.testing.assert_allclose(evolve_eff.h0.dense.matrix, target_h0)
    np.testing.assert_allclose(evolve_eff.h1.dense.matrix, target_h1)
    np.testing.assert_allclose(evolve.h0.dense.matrix, target_h0)
    np.testing.assert_allclose(evolve.h1.dense.matrix, target_h1)
    np.testing.assert_allclose(final_state, target_state)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    start_time = time.time()
    main(**args)
    print("Time:", time.time() - start_time)