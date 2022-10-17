# -*- coding: utf-8 -*-
import numpy as np
import pytest

from qibo.quantum_info import *


@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_shannon_entropy(backend, base):
    prob_array = np.asarray([1.0, 0.0])
    result = shannon_entropy(prob_array, base)
    backend.assert_allclose(result, 0.0)

    if base == 2:
        prob_array = np.asarray([0.5, 0.5])
        result = shannon_entropy(prob_array, base)
        backend.assert_allclose(result, 2.0)


def test_shannon_entropy_errors():
    with pytest.raises(ValueError):
        x = np.asarray([1.0, 0.0])
        shannon_entropy(x, -2)
    with pytest.raises(TypeError):
        x = np.asarray([[1.0, 0.0]])
        shannon_entropy(x)
    with pytest.raises(TypeError):
        x = np.asarray([])
        shannon_entropy(x)
    with pytest.raises(ValueError):
        x = np.asarray([0.5, 0.4999999])
        shannon_entropy(x)
