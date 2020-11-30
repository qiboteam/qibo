"""
Testing HEP models.
"""
import numpy as np
import pytest
from qibo.hep import qPDF


@pytest.mark.parametrize('ansatz', ['Weighted', 'Fourier'])
@pytest.mark.parametrize('layers', [3, 5])
@pytest.mark.parametrize('nqubits', [1, 8])
@pytest.mark.parametrize('multi_output', [True, False])
def test_qpdf(ansatz, layers, nqubits, multi_output):
    """Performs a qPDF circuit minimization test."""
    model = qPDF(ansatz, layers, nqubits, multi_output)
    params = np.random.rand(model.nparams)
    result = model.predict(params, [0.1])
    if multi_output:
        assert result.shape[1] == nqubits
        assert result.all() > 0
    else:
        assert result > 0
