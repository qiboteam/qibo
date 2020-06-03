import numpy as np
import pytest
from qibo.benchmarks import benchmark_models as models


@pytest.mark.parametrize("nqubits", [6, 7, 8, 9, 10])
@pytest.mark.parametrize("backend", ["Custom"])
@pytest.mark.parametrize("nlayers", [1, 2, 3])
def test_variational_agreement(nqubits, backend, nlayers):
    theta = 2 * np.pi * np.random.random(nlayers * 2 * nqubits)
    circuit = models.VariationalCircuit(nqubits, backend, nlayers,
                                        np.copy(theta))
    opt_circuit = models.OptimizedVariationalCircuit(nqubits, backend, nlayers,
                                                     np.copy(theta))
    target_state = circuit().numpy()
    final_state = opt_circuit().numpy()
    np.testing.assert_allclose(target_state, final_state, atol=1e-7)
