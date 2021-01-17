import pytest
import numpy as np
import tensorflow as tf
from qibo import backends
from numpy.random import random as rand


def test_backend_errors():
    with pytest.raises(ValueError):
        bk = backends._construct_backend("test")
    with pytest.raises(ValueError):
        backends.set_backend("a_b_c")
