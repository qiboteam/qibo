from qibo.symbols import X, Y, Z
from qibo.models import Circuit, QAOA
from qibo.core.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo import gates
import numpy as np


def calculate_two_to_one(num_cities):
    """
    Args:
        num_cities: number of cities

    Returns:
        a matrix to faciliate conversion from 2 dimension to one dimension
    """
    return np.arange(num_cities ** 2).reshape(num_cities, num_cities)


def tsp_phaser(distance_matrix, dense=True):
    """

    Args:
        distance_matrix: a numpy matrix encoding the distances
        dense: whether the hamiltonian is dense,

    Returns: a hamiltonian representing the TSP(traveling salesman problem) phaser according to
    `arxiv:1709.03489<https://arxiv.org/abs/1709.03489>` by Hadfield (2017).

    """
    num_cities = distance_matrix.shape[0]
    two_to_one = calculate_two_to_one(num_cities)
    form = 0
    for i in range(num_cities):
        for u in range(num_cities):
            for v in range(num_cities):
                if u != v:
                    form += distance_matrix[u, v] * Z(int(two_to_one[u, i]))* Z(
                        int(two_to_one[v, (i + 1) % num_cities]))
    ham = SymbolicHamiltonian(form)
    if dense:
        ham = ham.dense
    return ham


def tsp_mixer(num_cities, dense=True):
    """
    Args:
        num_cities: number of cities in TSP
        dense: whether the hamiltonian is dense

    Returns: a mixer hamiltonian representing the TSP(traveling salesman problem) phaser according to
    `arxiv:1709.03489<https://arxiv.org/abs/1709.03489>` by Hadfield (2017).

    """
    two_to_one = calculate_two_to_one(num_cities)
    splus = lambda u, i: X(int(two_to_one[u, i])) + 1j * Y(int(two_to_one[u, i]))
    sminus = lambda u, i: X(int(two_to_one[u, i])) - 1j * Y(int(two_to_one[u, i]))
    form = 0
    for i in range(num_cities):
        for u in range(num_cities):
            for v in range(num_cities):
                if u != v:
                    form += splus(u, i) * splus(v, (i + 1) % num_cities) * sminus(u, (
                            i + 1) % num_cities) * sminus(v, i) + sminus(u, i) * sminus(v, (
                            i + 1) % num_cities) * splus(u, (i + 1) % num_cities) * splus(
                        v, i)
    ham = SymbolicHamiltonian(form)
    if dense:
        ham = ham.dense
    return ham


class tsp:
    """
    This is a TSP (traveling salesman problem) class that enables us to implement TSP according to
    `arxiv:1709.03489<https://arxiv.org/abs/1709.03489>` by Hadfield (2017).
    ..testcode::
    num_cities = 3
    distance_matrix = np.array([[0, 0.9, 0.8],
     [0.4, 0, 0.1],
     [0,  0.7, 0]])
    # there are two possible cycles, one with distance 1, one with distance 1.9
    distance_matrix = distance_matrix.round(1)
    small_tsp = tsp(distance_matrix)
    obj_hamil, mixer = small_tsp.hamiltonians(dense=False)
    initial_parameters = np.random.uniform(0, 1, 2)
    initial_state = small_tsp.prepare_initial_state([i for i in range(num_cities)])
    qaoa = QAOA(obj_hamil, mixer=mixer)


    def evaluate_dist(config):
        m = int(np.sqrt(len(config)))
        cauchy = [-1] * m  # Cauchy's notation for permutation
        for i in range(m):
            for j in range(m):
                if config[m*i+j] == '1':
                    cauchy[j] = i # citi i is in slot j
        return sum(distance_matrix[cauchy[i]][cauchy[(i+1)%m]] for i in range(m))


    def amplitude_config(long_config):
        amplitude, config = long_config.split('|')
        prob = np.linalg.norm(eval(amplitude))**2
        dist = evaluate_dist(config)
        return prob*dist


    def qaoa_function_of_layer(layer):
        best_energy, final_parameters, extra = qaoa.minimize(initial_p=[0.1] * 8,
                                                             initial_state=initial_state)
        qaoa.set_parameters(final_parameters)
        quantum_state = qaoa.execute(initial_state)
        quantum_state_symbolic = quantum_state.symbolic().split('>')[:-1]
        return sum(amplitude_config(long_config) for long_config in quantum_state_symbolic)


    print([qaoa_function_of_layer(i) for i in [2, 4, 6, 8]])
    # example of output [1.12205683883, 1.7475528727300003, 1.26960151366, 1.4211151713499999]
    """

    def __init__(self, distance_matrix):
        """
        Args:
            distance_matrix: a numpy matrix encoding the distance matrix.
        """
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.two_to_one = calculate_two_to_one(self.num_cities)

    def hamiltonians(self, dense=True):
        """
        Args:
            dense: Indicates if the Hamiltonian is dense.
        Returns: Return a pair of Hamiltonian for the objective as well as the mixer.
        """
        return tsp_phaser(self.distance_matrix, dense), tsp_mixer(self.num_cities, dense)

    def prepare_initial_state(self, ordering):
        """
        To run QAOA by Hadsfield, we need to start from a valid permutation function to ensure feasibility.
        Args:
            ordering is a list which is a permutation from 0 to n-1
        Returns: return an initial state that can be used to start TSP QAOA.
        """
        c = Circuit(len(ordering) ** 2)
        for i in range(len(ordering)):
            c.add(gates.X(int(self.two_to_one[ordering[i], i])))
        result = c()
        return result.state(numpy=True)






