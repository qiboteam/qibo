from qibo.models import Circuit
from qibo import gates
import numpy as np
from scipy.optimize import minimize


def ansatz(theta, p=0):
    """Ansatz for driving a random state into its up-tp-phases canonical form.
        Args:
            theta (array): parameters of the unitary rotations
            p=0 (float): probability of occuring a single-qubit depolarizing error
        Returns:
            Qibo circuit
    """
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


def cost_function(theta, state, p=0, shots=1000):
    """Cost function encoding the difference between a state and its up-to-phases canonical form
        Args:
            theta (array): parameters of the unitary rotations
            state (cplx array): three-qubit random state
            p=0 (float): probability of occuring a single-qubit depolarizing error
            shots (int): Shots used for measuring every circuit.
        Returns:
            float, cost function
    """
    C = ansatz(theta, p)
    measurements = C.execute(state, nshots=shots).frequencies(binary=False)

    return (measurements[1] + measurements[2] + measurements[3]) / shots


def canonize(state, p=0, shots=1000):
    """Function to transform a given state into its up-to-phases canonical form
        Args:
            state (cplx array): three-qubit random state
            p=0 (float): probability of occuring a single-qubit depolarizing error
            shots (int): Shots used for measuring every circuit.
        Returns:
            Value cost function, parameters to canonize the given state
        """

    theta = np.zeros(9)
    result = minimize(cost_function, theta, args=(state, p, shots), method='powell')
    return result.fun, result.x

def canonical_tangle(state, theta, p=0, shots=1000, post_selection=True):
    """Tangle of a canonized quantum state
        Args:
            state (cplx array): three-qubit random state
            theta (array): parameters of the unitary rotations
            p=0 (float): probability of occuring a single-qubit depolarizing error
            shots (int): Shots used for measuring every circuit.
            post_selection (bool): whether post selection is applied or not
        Returns:
            tangle
    """
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
    """Hyperdeterminant of any quantum state
        Args:
            state (cplx array): three-qubit random state
        Returns:
            Hyperdeterminant
    """
    indices = [(1, [(0, 0, 7, 7), (1, 1, 6, 6), (2, 2, 5, 5), (3, 3, 4, 4)]),
               (-2, [(0, 7, 3, 4), (0, 7, 5, 2), (0, 7, 6, 1), (3, 4, 5, 2), (3, 4, 6, 1), (5, 2, 6, 1)]),
               (4, [(0, 6, 5, 3), (7, 1, 2, 4)])]
    hyp = sum(coeff * sum(state[i] * state[j] * state[k] * state[l] for i, j, k, l in ids)
              for coeff, ids in indices)

    return hyp


def opt_hyperdeterminant(measures):
    """Hyperdeterminant of a canonized quantum state from its outcomes
        Args:
            measures (array): outcomes of the canonized state
        Returns:
            Hyperdeterminant
    """
    hyp = measures[0] * measures[7]
    return hyp


def create_random_state(seed):
    """Function to create a random quantum state from sees
        Args:
            seed (int): random seed
        Returns:
            Random quantum state
    """
    np.random.seed(seed)
    state = (np.random.rand(8) - .5) + 1j*(np.random.rand(8) - .5)
    state = state / np.linalg.norm(state)

    return state


def compute_random_tangle(seed):
    """Function to compute the tangle of a randomly created random quantum state from seed
        Args:
            seed (int): random seed
        Returns:
            Tangle
    """
    state = create_random_state(seed)
    return 4 * np.abs(hyperdeterminant(state))



