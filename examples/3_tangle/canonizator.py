from qibo.models import Circuit
from qibo import gates
import numpy as np
from scipy.optimize import minimize

p_ = 0
shots_ = 10000

def ansatz(theta, p=0):
    C = Circuit(3)
    index = 0
    if p == 0:
        for i in range(3):
            C.add(gates.RZ(i, theta[index]))
            C.add(gates.RY(i, theta[index + 1]))
            C.add(gates.RZ(i, theta[index + 2]))
            index += 3
        for i in range(3):
            C.add(gates.M(i))

    else:
        from qibo import set_backend
        set_backend("matmuleinsum")
        raise RuntimeWarning('Backend changed to MatmulEinsum')
        for i in range(3):
            C.add(gates.RZ(i, theta[index]))
            C.add(gates.RY(i, theta[index + 1]))
            C.add(gates.RZ(i, theta[index + 2]))
            C.add(gates.NoiseChannel(i, px=p/3, py=p/3, pz=p/3))
            index += 3
        for i in range(3):
            C.add(gates.NoiseChannel(i, px=10 * p))
            C.add(gates.M(i))

    return C

def cost_function(theta, state, p=p_, shots=shots_):
    C = ansatz(theta, p)
    measurements = C.execute(state, nshots=shots).frequencies(binary=False)

    return (measurements[1] + measurements[2] + measurements[3]) / shots

def canonize(state, p=p_, shots=shots_):
    theta = np.zeros(9)
    result = minimize(cost_function, theta, args=(state, p, shots), method='powell')
    return result.fun, result.x

def canonical_tangle(state, theta, p=p_, shots=shots_, post_selection=True):
    C = ansatz(theta, p)
    result = C.execute(state, nshots=shots).frequencies(binary=False)
    measures = np.zeros(8)
    for i in range(8):
        try:
            measures[i] = result[i] / shots
        except:
            measures[i] = 0
    if post_selection:
        measures[1] = 0
        measures[2] = 0
        measures[3] = 0
        measures = measures / np.sum(measures)

    return 4*opt_hyperdeterminant(measures)

def hyperdeterminant(state):
    """
    :param state: quantum state
    :return: the hyperdeterminant of the state
    """
    hyp = (
    state[0]**2 * state[7]**2 + state[1]**2 * state[6]**2 + state[2]**2 * state[5]**2 + state[3]**2 * state[4]**2
    ) - 2*(
    state[0]*state[7]*state[3]*state[4] + state[0]*state[7]*state[5]*state[2] + state[0]*state[7]*state[6]*state[1] + state[3]*state[4]*state[5]*state[2] + state[3]*state[4]*state[6]*state[1] + state[5]*state[2]*state[6]*state[1]
    ) + 4 * (
    state[0]*state[6]*state[5]*state[3] + state[7]*state[1]*state[2]*state[4]
    )
    return hyp

def opt_hyperdeterminant(measures):
    """

    :param state: quantum state
    :return: the hyperdeterminant of the state
    """
    hyp = (
    measures[0] * measures[7]
    )
    return hyp

def create_random_state(seed):
    np.random.seed(seed)
    state = (np.random.rand(8) - .5) + 1j*(np.random.rand(8) - .5)
    state = state / np.linalg.norm(state)

    return state

def compute_random_tangle(seed):
    state = create_random_state(seed)
    return 4 * np.abs(hyperdeterminant(state))



