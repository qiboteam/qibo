import numpy as np
import pytest

from qibo import gates
from qibo.models import QAOA, Circuit
from qibo.models.tsp import TSP

from .test_models_variational import assert_regression_fixture


def qaoa_function_of_layer(backend, layer):
    """
    This is a function to study the impact of the number of layers on QAOA, it takes
    in the number of layers and compute the distance of the mode of the histogram obtained
    from QAOA
    """
    num_cities = 3
    distance_matrix = np.array([[0, 0.9, 0.8], [0.4, 0, 0.1], [0, 0.7, 0]])
    # there are two possible cycles, one with distance 1, one with distance 1.9
    distance_matrix = distance_matrix.round(1)

    small_tsp = TSP(distance_matrix, backend=backend)
    initial_state = small_tsp.prepare_initial_state([i for i in range(num_cities)])
    obj_hamil, mixer = small_tsp.hamiltonians()
    qaoa = QAOA(obj_hamil, mixer=mixer)
    initial_state = backend.cast(initial_state, copy=True)
    best_energy, final_parameters, extra = qaoa.minimize(
        initial_p=[0.1 for i in range(layer)],
        initial_state=initial_state,
        method="BFGS",
        options={"maxiter": 1},
    )
    qaoa.set_parameters(final_parameters)
    return qaoa.execute(initial_state)


@pytest.mark.parametrize("nlayers", [2, 4])
def test_tsp(backend, nlayers):
    final_state = backend.to_numpy(qaoa_function_of_layer(backend, nlayers))
    assert_regression_fixture(
        backend, final_state.real, f"tsp_layer{nlayers}_real.out", rtol=1e-3, atol=1e-5
    )
    assert_regression_fixture(
        backend, final_state.imag, f"tsp_layer{nlayers}_imag.out", rtol=1e-3, atol=1e-5
    )
