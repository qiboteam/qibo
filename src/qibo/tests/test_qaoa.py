import pytest
import numpy as np
from scipy.linalg import expm
from qibo import models, hamiltonians
from qibo.tests import utils


@pytest.mark.parametrize("trotter", [False])
def test_qaoa_execution(trotter):
    h = hamiltonians.TFIM(4, h=1.0, trotter=trotter)
    m = hamiltonians.X(4, trotter=trotter)
    params = np.random.random(4)
    state = utils.random_numpy_state(4)

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
    np.testing.assert_allclose(final_state, target_state)
