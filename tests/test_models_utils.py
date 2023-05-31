"""Test `fourier_coefficients' in `qibo/models/utils.py`."""
import numpy as np
import pytest

from qibo.models.utils import fourier_coefficients


def test_fourier_coefficients_raising_errors():
    result = fourier_coefficients(function, 1, 2, lowpass_filter=True)
    with pytest.raises(ValueError):
        # n_inputs!=len(degree)
        result = fourier_coefficients(
            function, n_inputs=2, degree=2, lowpass_filter=True
        )
    with pytest.raises(ValueError):
        # n_inputs!=len(filter_threshold)
        result = fourier_coefficients(
            function, 1, 2, lowpass_filter=True, filter_threshold=[5, 6]
        )
    result = fourier_coefficients(
        function, n_inputs=1, degree=2, lowpass_filter=True, filter_threshold=5
    )
    result = fourier_coefficients(function, n_inputs=1, degree=2, lowpass_filter=False)


def test_fourier_coefficients_expected_result():
    result = fourier_coefficients(function, 1, 2, lowpass_filter=True)
    # The coefficients should have this form:
    coeffs = np.array([1, 1 - 2j, 1.5 - 2.5j, 1.5 + 2.5j, 1 + 2j])
    assert coeffs.all() == result.all()


def function(x):
    y = 1 + 2 * np.cos(x) + 3 * np.cos(2 * x) + 4 * np.sin(x) + 5 * np.sin(2 * x)
    return y
