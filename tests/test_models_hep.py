"""Testing HEP models."""

import numpy as np
import pytest

from qibo.models.hep import qPDF

test_names = "ansatz,layers,nqubits,multi_output,output"
test_values = [
    ("Fourier", 3, 1, True, np.array([[1.214777]])),
    ("Weighted", 5, 1, True, np.array([[0.506931]])),
    ("Fourier", 5, 1, True, np.array([[0.475777]])),
    (
        "Weighted",
        3,
        8,
        True,
        np.array(
            [
                [
                    0.102331,
                    0.196291,
                    0.250836,
                    0.246776,
                    2.63353,
                    0.658077,
                    0.421151,
                    0.025667,
                ]
            ]
        ),
    ),
    (
        "Fourier",
        3,
        8,
        True,
        np.array(
            [
                [
                    1.441044,
                    15.967666,
                    4.176307,
                    21.950216,
                    4.55334,
                    1.860604,
                    33.80421,
                    25.587542,
                ]
            ]
        ),
    ),
    (
        "Weighted",
        5,
        8,
        True,
        np.array(
            [
                [
                    0.028491,
                    0.239515,
                    0.163759,
                    0.090604,
                    1.913209,
                    0.36426,
                    0.357044,
                    0.028548,
                ]
            ]
        ),
    ),
    (
        "Fourier",
        5,
        8,
        True,
        np.array(
            [
                [
                    2.473517,
                    5.402246,
                    1.073041,
                    84.60309,
                    2.271607,
                    2.877924,
                    27.200738,
                    1.657042,
                ]
            ]
        ),
    ),
    ("Weighted", 3, 1, False, np.array([[0.613767]])),
    ("Fourier", 3, 1, False, np.array([[1.214777]])),
    ("Weighted", 5, 1, False, np.array([[0.506931]])),
    ("Fourier", 5, 1, False, np.array([[0.475777]])),
    ("Weighted", 3, 8, False, np.array([[0.102331]])),
    ("Fourier", 3, 8, False, np.array([[1.441044]])),
    ("Weighted", 5, 8, False, np.array([[0.028491]])),
    ("Fourier", 5, 8, False, np.array([[2.473517]])),
]


@pytest.mark.parametrize(test_names, test_values)
def test_qpdf(backend, ansatz, layers, nqubits, multi_output, output):
    """Performs a qPDF circuit minimization test."""
    model = qPDF(ansatz, layers, nqubits, multi_output, backend=backend)
    np.random.seed(0)
    params = np.random.rand(model.nparams)
    result = model.predict(params, [0.1])
    atol = 1e-5
    rtol = 1e-7
    np.testing.assert_allclose(result, output, rtol=rtol, atol=atol)
