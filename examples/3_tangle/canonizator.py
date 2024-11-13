import numpy as np
from scipy.optimize import minimize

from qibo import Circuit, gates


def ansatz(p=0):
    """Ansatz for driving a random state into its up-tp-phases canonical form.

    Args:
      p (float): probability of occuring a single-qubit depolarizing error

    Returns:
      Qibo circuit implementing the variational ansatz.
    """
    C = Circuit(3, density_matrix=p > 0)
    for i in range(3):
        C.add(gates.RZ(i, theta=0))
        C.add(gates.RY(i, theta=0))
        C.add(gates.RZ(i, theta=0))
        if p > 0:
            C.add(
                gates.PauliNoiseChannel(i, [("X", p / 3), ("Y", p / 3), ("Z", p / 3)])
            )
    for i in range(3):
        if p > 0:
            C.add(gates.PauliNoiseChannel(i, [("X", 10 * p)]))
        C.add(gates.M(i))
    return C


def cost_function(theta, state, circuit, shots: int = 1000):
    """Cost function encoding the difference between a state and its up-to-phases canonical form.

    Args:
        theta (ndarray): parameters of the unitary rotations.
        state (ndarray): three-qubit random state.
        circuit (:class:`qibo.models.Circuit`): variational circuit.
        shots (int, optional): shots used for measuring every circuit. Defaults to :math:`1000`.

    Returns:
        float: Cost function
    """
    circuit.set_parameters(theta)
    measurements = circuit(state, nshots=shots).frequencies(binary=False)
    return (measurements[1] + measurements[2] + measurements[3]) / shots


def canonize(state, circuit, shots: int = 1000):
    """Function to transform a given state into its up-to-phases canonical form.

    Args:
        state (ndarray): three-qubit random state.
        circuit (:class:`qibo.models.Circuit`): variational circuit.
        shots (int): shots used for measuring every circuit. Defaults to :math:`1000`.

    Returns:
        Value cost function, parameters to canonize the given state
    """
    theta = np.zeros(9)
    result = minimize(
        cost_function, theta, args=(state, circuit, shots), method="powell"
    )
    return result.fun, result.x


def canonical_tangle(
    state, theta, circuit, shots: int = 1000, post_selection: bool = True
):
    """Tangle of a canonized quantum state.

    Args:
        state (ndarray): three-qubit random state.
        theta (array): parameters of the unitary rotations.
        circuit (:class:`qibo.models.Circuit`): variational circuit.
        shots (int, optional): shots used for measuring every circuit. Defaults to :math:`1000`.
        post_selection (bool, optional): whether post selection is applied or not

    Returns:
        float: tangle
    """
    circuit.set_parameters(theta)
    result = circuit(state, nshots=shots).frequencies(binary=False)
    measures = np.zeros(8)
    for i, r in result.items():
        measures[i] = result[i] / shots
    if post_selection:
        measures[1] = 0
        measures[2] = 0
        measures[3] = 0
        measures = measures / np.sum(measures)
    return 4 * opt_hyperdeterminant(measures)


def hyperdeterminant(state):
    """Hyperdeterminant of any quantum state
    Args:
        state (cplx array): three-qubit random state
    Returns:
        Hyperdeterminant
    """
    indices = [
        (1, [(0, 0, 7, 7), (1, 1, 6, 6), (2, 2, 5, 5), (3, 3, 4, 4)]),
        (
            -2,
            [
                (0, 7, 3, 4),
                (0, 7, 5, 2),
                (0, 7, 6, 1),
                (3, 4, 5, 2),
                (3, 4, 6, 1),
                (5, 2, 6, 1),
            ],
        ),
        (4, [(0, 6, 5, 3), (7, 1, 2, 4)]),
    ]
    hyp = sum(
        coeff * sum(state[i] * state[j] * state[k] * state[l] for i, j, k, l in ids)
        for coeff, ids in indices
    )

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


def create_random_state(seed, density_matrix=False):
    """Function to create a random quantum state from sees
    Args:
        seed (int): random seed
    Returns:
        Random quantum state
    """
    np.random.seed(seed)
    state = (np.random.rand(8) - 0.5) + 1j * (np.random.rand(8) - 0.5)
    state = state / np.linalg.norm(state)
    if density_matrix:
        return np.tensordot(np.conj(state), state, axes=0)
    else:
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
