import pytest
import numpy as np
from scipy.linalg import expm
from qibo import models, hamiltonians
from qibo.tests import utils


@pytest.mark.parametrize("trotter", [False, True])
def test_qaoa_execution(trotter):
    h = hamiltonians.TFIM(4, h=1.0, trotter=trotter)
    m = hamiltonians.X(4, trotter=trotter)
    params = 0.01 * np.random.random(4) # Trotter requires small p's!
    state = utils.random_numpy_state(4)
    if trotter:
        atol = 1e-6
    else:
        atol = 0

    target_state = np.copy(state)
    h_matrix = h.matrix.numpy()
    m_matrix = m.matrix.numpy()
    for i, p in enumerate(params):
        if i % 2:
            u = expm(-1j * p * m_matrix)
        else:
            u = expm(-1j * p * h_matrix)
        target_state = u @ target_state

    qaoa = models.QAOA(h, mixer=m)
    qaoa.set_parameters(params)
    final_state = qaoa(np.copy(state))
    np.testing.assert_allclose(final_state, target_state, atol=atol)
