import numpy as np

from qibo import gates
from qibo.models.circuit import Circuit


def convert_bit_to_energy(hamiltonian, bitstring):
    """
    Given a binary string and a hamiltonian, we compute the corresponding energy.
    make sure the bitstring is of the right length
    """
    n = len(bitstring)
    c = Circuit(n)
    active_bit = [i for i in range(n) if bitstring[i] == "1"]
    for i in active_bit:
        c.add(gates.X(i))
    result = c()  # this is an execution result, a quantum state
    return hamiltonian.expectation(result.state())


def convert_state_to_count(state):
    """
    This is a function that convert a quantum state to a dictionary keeping track of
    energy and its frequency.
    d[energy] records the frequency
    """
    return np.abs(state) ** 2


def compute_cvar(probabilities, values, alpha):
    """
    Auxilliary method to computes CVaR for given probabilities, values, and confidence level.

    Args:
        probabilities (list): list/array of probabilities
        values (list): list/array of corresponding values
        alpha (float): confidence level

    Returns:
        CVaR
    """
    sorted_indices = np.argsort(values)
    probs = np.array(probabilities)[sorted_indices]
    vals = np.array(values)[sorted_indices]
    cvar = 0
    total_prob = 0
    for i, (p, v) in enumerate(zip(probs, vals)):
        if p >= alpha - total_prob:
            p = alpha - total_prob
        total_prob += p
        cvar += p * v
    cvar /= total_prob
    return cvar


def cvar(hamiltonian, state, alpha=0.1):
    """
    Given the hamiltonian and state, this function estimate the
    corresponding cvar function
    """
    counts = convert_state_to_count(state)
    probabilities = np.zeros(len(counts))
    values = np.zeros(len(counts))
    m = int(np.log2(state.size))
    for i, p in enumerate(counts):
        values[i] = convert_bit_to_energy(hamiltonian, bin(i)[2:].zfill(m))
        probabilities[i] = p
    cvar_ans = compute_cvar(probabilities, values, alpha)
    return cvar_ans


def gibbs(hamiltonian, state, eta=0.1):
    """
    Given the hamiltonian and the state, and optional eta value
    it estimate the gibbs function value.
    """
    counts = convert_state_to_count(state)
    avg = 0
    sum_count = 0
    m = int(np.log2(state.size))
    for bitstring, count in enumerate(counts):
        obj = convert_bit_to_energy(hamiltonian, bin(bitstring)[2:].zfill(m))
        avg += np.exp(-eta * obj)
        sum_count += count
    return -np.log(avg / sum_count)
