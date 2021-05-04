import numpy as np


def random_state(nqubits):
    """Generates a random normalized state vector as numpy array."""
    nstates = 2 ** nqubits
    initial_state = np.random.random(nstates) + 1j * np.random.random(nstates)
    return initial_state / np.sqrt((np.abs(initial_state) ** 2).sum())


def random_density_matrix(nqubits):
    """Generates a random normalized density matrix."""
    shape = 2 * (2 ** nqubits,)
    m = np.random.random(shape) + 1j * np.random.random(shape)
    rho = (m + m.T.conj()) / 2.0
    # Normalize
    ids = np.arange(2 ** nqubits)
    rho[ids, ids] = rho[ids, ids] / np.trace(rho)
    return rho
