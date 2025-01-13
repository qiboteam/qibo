import numpy as np

from qibo import gates
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models.circuit import Circuit
from qibo.symbols import X, Y, Z


def calculate_two_to_one(num_cities):
    return np.arange(num_cities**2).reshape(num_cities, num_cities)


def tsp_phaser(distance_matrix, backend=None):
    num_cities = distance_matrix.shape[0]
    two_to_one = calculate_two_to_one(num_cities)
    form = 0
    for i in range(num_cities):
        for u in range(num_cities):
            for v in range(num_cities):
                if u != v:
                    form += (
                        distance_matrix[u, v]
                        * Z(int(two_to_one[u, i]), backend=backend)
                        * Z(int(two_to_one[v, (i + 1) % num_cities]), backend=backend)
                    )
    ham = SymbolicHamiltonian(form, backend=backend)
    return ham


def tsp_mixer(num_cities, backend=None):
    two_to_one = calculate_two_to_one(num_cities)
    splus = lambda u, i: X(int(two_to_one[u, i]), backend=backend) + 1j * Y(
        int(two_to_one[u, i]), backend=backend
    )
    sminus = lambda u, i: X(int(two_to_one[u, i]), backend=backend) - 1j * Y(
        int(two_to_one[u, i]), backend=backend
    )
    form = 0
    for i in range(num_cities):
        for u in range(num_cities):
            for v in range(num_cities):
                if u != v:
                    form += splus(u, i) * splus(v, (i + 1) % num_cities) * sminus(
                        u, (i + 1) % num_cities
                    ) * sminus(v, i) + sminus(u, i) * sminus(
                        v, (i + 1) % num_cities
                    ) * splus(
                        u, (i + 1) % num_cities
                    ) * splus(
                        v, i
                    )
    ham = SymbolicHamiltonian(form, backend=backend)
    return ham


class TSP:
    """
    The travelling salesman problem (also called the travelling salesperson problem or TSP)
    asks the following question: "Given a list of cities and the distances between each pair of cities,
    what is the shortest possible route for a salesman to visit each city exactly once and return to the origin city?"
    It is an NP-hard problem in combinatorial optimization. It is also important in theoretical computer science and
    operations research.

    This is a TSP class that enables us to implement TSP according to
    `arxiv:1709.03489 <https://arxiv.org/abs/1709.03489>`_ by Hadfield (2017).

    Args:
            distance_matrix: a numpy matrix encoding the distance matrix.
            backend: Backend to use for calculations. If not given the global backend will be used.

    Example:
        .. testcode::

            from qibo.models.tsp import TSP
            import numpy as np
            from collections import defaultdict
            from qibo import gates
            from qibo.models import QAOA
            from qibo.result import CircuitResult


            def convert_to_standard_Cauchy(config):
                m = int(np.sqrt(len(config)))
                cauchy = [-1] * m  # Cauchy's notation for permutation, e.g. (1,2,0) or (2,0,1)
                for i in range(m):
                    for j in range(m):
                        if config[m * i + j] == '1':
                            cauchy[j] = i  # citi i is in slot j
                for i in range(m):
                    if cauchy[i] == 0:
                        cauchy = cauchy[i:] + cauchy[:i]
                        return tuple(cauchy)  # now, the cauchy notation for permutation begins with 0


            def evaluate_dist(cauchy):
                '''
                Given a permutation of 0 to n-1, we compute the distance of the tour

                '''
                m = len(cauchy)
                return sum(distance_matrix[cauchy[i]][cauchy[(i+1)%m]] for i in range(m))


            def qaoa_function_of_layer(layer, distance_matrix):
                '''
                This is a function to study the impact of the number of layers on QAOA, it takes
                in the number of layers and compute the distance of the mode of the histogram obtained
                from QAOA

                '''
                small_tsp = TSP(distance_matrix)
                obj_hamil, mixer = small_tsp.hamiltonians()
                qaoa = QAOA(obj_hamil, mixer=mixer)
                best_energy, final_parameters, extra = qaoa.minimize(initial_p=[0.1] * layer,
                                                     initial_state=initial_state, method='BFGS')
                qaoa.set_parameters(final_parameters)
                quantum_state = qaoa.execute(initial_state)
                circuit = Circuit(9)
                circuit.add(gates.M(*range(9)))
                result = CircuitResult(quantum_state, circuit.measurements, small_tsp.backend, nshots=1000)
                freq_counter = result.frequencies()
                # let's combine freq_counter here, first convert each key and sum up the frequency
                cauchy_dict = defaultdict(int)
                for freq_key in freq_counter:
                    standard_cauchy_key = convert_to_standard_Cauchy(freq_key)
                    cauchy_dict[standard_cauchy_key] += freq_counter[freq_key]
                max_key = max(cauchy_dict, key=cauchy_dict.get)
                return evaluate_dist(max_key)

            np.random.seed(42)
            num_cities = 3
            distance_matrix = np.array([[0, 0.9, 0.8], [0.4, 0, 0.1],[0, 0.7, 0]])
            distance_matrix = distance_matrix.round(1)
            small_tsp = TSP(distance_matrix)
            initial_parameters = np.random.uniform(0, 1, 2)
            initial_state = small_tsp.prepare_initial_state([i for i in range(num_cities)])
            qaoa_function_of_layer(2, distance_matrix)

    """

    def __init__(self, distance_matrix, backend=None):
        from qibo.backends import _check_backend

        self.backend = _check_backend(backend)

        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.two_to_one = calculate_two_to_one(self.num_cities)

    def hamiltonians(self):
        """
        Returns:
            The pair of Hamiltonian describes the phaser hamiltonian
            and the mixer hamiltonian.

        """
        return (
            tsp_phaser(self.distance_matrix, backend=self.backend),
            tsp_mixer(self.num_cities, backend=self.backend),
        )

    def prepare_initial_state(self, ordering):
        """
        To run QAOA by Hadsfield, we need to start from a valid permutation function to ensure feasibility.

        Args:
            ordering (array): A list describing permutation from 0 to n-1

        Returns:
            An initial state that is used to start TSP QAOA.

        """
        circuit = Circuit(len(ordering) ** 2)
        for i in range(len(ordering)):
            circuit.add(gates.X(int(self.two_to_one[ordering[i], i])))
        result = self.backend.execute_circuit(circuit)
        return result.state()
