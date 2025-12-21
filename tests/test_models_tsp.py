import pathlib

import numpy as np
import pytest

from qibo.models import QAOA
from qibo.models.tsp import TSP

REGRESSION_FOLDER = pathlib.Path(__file__).with_name("regressions")


def assert_regression_fixture(backend, array, filename, rtol=1e-5, atol=1e-12):
    """Check array matches data inside filename.

    Args:
        array: numpy array/
        filename: fixture filename

    If filename does not exists, this function
    creates the missing file otherwise it loads
    from file and compare.
    """

    def load(filename):
        return np.loadtxt(filename)

    filename = REGRESSION_FOLDER / filename
    try:
        array_fixture = load(filename)
    except:  # pragma: no cover
        # case not tested in GitHub workflows because files exist
        np.savetxt(filename, array)
        array_fixture = load(filename)
    backend.assert_allclose(array, array_fixture, rtol=rtol, atol=atol)


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
    # if nlayers == 4 and backend.platform in ("cupy", "cuquantum"):
    #     pytest.skip("Failing for cupy and cuquantum.")
    final_state = backend.to_numpy(qaoa_function_of_layer(backend, nlayers))
    atol = 4e-5 if backend.platform in ("cupy", "cuquantum") else 1e-5
    assert_regression_fixture(
        backend, final_state.real, f"tsp_layer{nlayers}_real.out", rtol=1e-3, atol=atol
    )
    assert_regression_fixture(
        backend, final_state.imag, f"tsp_layer{nlayers}_imag.out", rtol=1e-3, atol=atol
    )
