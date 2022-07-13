import pytest
import numpy as np
from collections import defaultdict
from qibo import gates
from qibo.models import QAOA, Circuit
from qibo.models.tsp import TSP
from qibo.states import CircuitResult


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


def qaoa_function_of_layer(backend, layer):
    '''
    This is a function to study the impact of the number of layers on QAOA, it takes
    in the number of layers and compute the distance of the mode of the histogram obtained
    from QAOA
    '''
    num_cities = 3
    distance_matrix = np.array([[0, 0.9, 0.8],
                                [0.4, 0, 0.1],
                                [0, 0.7, 0]])
    # there are two possible cycles, one with distance 1, one with distance 1.9
    distance_matrix = distance_matrix.round(1)

    small_tsp = TSP(distance_matrix, backend=backend)
    initial_state = small_tsp.prepare_initial_state([i for i in range(num_cities)])
    obj_hamil, mixer = small_tsp.hamiltonians()
    qaoa = QAOA(obj_hamil, mixer=mixer)
    initial_state = backend.cast(initial_state, copy=True)
    best_energy, final_parameters, extra = qaoa.minimize(initial_p=[0.1 for i in range(layer)],
                                                         initial_state=initial_state, method='BFGS')
    qaoa.set_parameters(final_parameters)
    quantum_state = qaoa.execute(initial_state)
    circuit = Circuit(9)
    circuit.add(gates.M(*range(9)))
    result = CircuitResult(small_tsp.backend, circuit, quantum_state, nshots=1000)
    freq_counter = result.frequencies()
    # let's combine freq_counter here, first convert each key and sum up the frequency
    cauchy_dict = defaultdict(int)
    for freq_key in freq_counter:
        standard_cauchy_key = convert_to_standard_Cauchy(freq_key)
        cauchy_dict[standard_cauchy_key] += freq_counter[freq_key]
    max_key = max(cauchy_dict, key=cauchy_dict.get)
    # Given a permutation of 0 to n-1, we compute the distance of the tour
    m = len(max_key)
    return sum(distance_matrix[max_key[i]][max_key[(i + 1) % m]] for i in range(m))


@pytest.mark.parametrize("test_layer, expected", [(4, 1.0), (6, 1.0), (8, 1.0)])
def test_tsp(backend, test_layer, expected):
    backend.set_seed(2)
    tmp = qaoa_function_of_layer(backend, test_layer)
    assert abs(tmp - expected) <= 0.001
