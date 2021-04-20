import pytest
import numpy as np
from qibo import K
from numpy.random import random as rand


METHODS = [
    ("cast", [rand(4)]),
    ("diag", [rand(4)]),
    ("reshape", {"x": rand(4), "shape": (2, 2)}),
    ("stack", [[rand(4), rand(4)]]),
    ("concatenate", {"x": [rand((2, 3)), rand((2, 4))], "axis": 1}),
    ("expand_dims", {"x": rand((5, 5)), "axis": 1}),
    ("copy", [rand(5)]),
    ("range", {"start": 0, "finish": 10, "step": 2}),
    ("range", {"start": 0, "finish": 10, "step": 2, "dtype": "DTYPE"}),
    ("eye", [5]),
    ("zeros", [(2, 3)]),
    ("ones", [(2, 3)]),
    ("zeros_like", [rand((4, 4))]),
    ("ones_like", [rand((4, 4))]),
    ("real", [rand(5)]),
    ("imag", [rand(5)]),
    ("conj", [rand(5)]),
    ("mod", [np.random.randint(10), np.random.randint(2, 10)]),
    ("right_shift", [np.random.randint(10), np.random.randint(10)]),
    ("exp", [rand(5)]),
    ("sin", [rand(5)]),
    ("cos", [rand(5)]),
    ("pow", {"base": rand(5), "exponent": 4}),
    ("square", [rand(5)]),
    ("sqrt", [rand(5)]),
    ("log", [rand(5)]),
    ("abs", [rand(5)]),
    ("expm", [rand((4, 4))]),
    ("trace", [rand((6, 6))]),
    ("sum", [rand((4, 4))]),
    ("sum", {"x": rand((4, 4, 3)), "axis": 1}),
    ("matmul", [rand((4, 6)), rand((6, 5))]),
    ("outer", [rand((4,)), rand((3,))]),
    ("kron", [rand((4, 4)), rand((5, 5))]),
    ("einsum", ["xy,axby->ab", rand((2, 2)), rand(4 * (2,))]),
    ("tensordot", [rand((2, 2)), rand(4 * (2,)), [[0, 1], [1, 3]]]),
    ("transpose", [rand((3, 3, 3)), [0, 2, 1]]),
    ("inv", [rand((4, 4))]),
    ("eigvalsh", [rand((4, 4))]),
    ("less", [rand(10), rand(10)]),
    ("array_equal", [rand(10), rand(10)]),
    ("squeeze", {"x": rand((5, 1, 2)), "axis": 1}),
    ("gather_nd", [rand((5, 3)), [0, 1]]),
    ("initial_state", [5, True]),
    ("initial_state", [3, False])
]
@pytest.mark.parametrize("method,kwargs", METHODS)
def test_backend_methods(tested_backend, target_backend, method, kwargs):
    tested_backend = K.construct_backend(tested_backend)
    target_backend = K.construct_backend(target_backend)
    tested_func = getattr(tested_backend, method)
    target_func = getattr(target_backend, method)
    if isinstance(kwargs, dict):
        np.testing.assert_allclose(tested_func(**kwargs), target_func(**kwargs))
    else:
        if method in {"kron", "inv"} and "numpy" not in tested_backend.name:
            with pytest.raises(NotImplementedError):
                tested_func(*kwargs)
        else:
            np.testing.assert_allclose(tested_func(*kwargs), target_func(*kwargs))


def test_backend_eigh(tested_backend, target_backend):
    tested_backend = K.construct_backend(tested_backend)
    target_backend = K.construct_backend(target_backend)
    m = rand((5, 5))
    eigvals1, eigvecs1 = tested_backend.eigh(m)
    eigvals2, eigvecs2 = target_backend.eigh(m)
    np.testing.assert_allclose(eigvals1, eigvals2)
    np.testing.assert_allclose(np.abs(eigvecs1), np.abs(eigvecs2))


def test_backend_compile(tested_backend, target_backend):
    tested_backend = K.construct_backend(tested_backend)
    target_backend = K.construct_backend(target_backend)
    func = lambda x: x + 1
    x = rand(5)
    cfunc1 = tested_backend.compile(func)
    cfunc2 = target_backend.compile(func)
    np.testing.assert_allclose(cfunc1(x), cfunc2(x))


def test_backend_gather(tested_backend, target_backend):
    tested_backend = K.construct_backend(tested_backend)
    target_backend = K.construct_backend(target_backend)
    x = rand(5)
    target_result = target_backend.gather(x, indices=[0, 1, 3])
    test_result = tested_backend.gather(x, indices=[0, 1, 3])
    np.testing.assert_allclose(test_result, target_result)
    x = rand((5, 5))
    target_result = target_backend.gather(x, indices=[0, 1, 3], axis=-1)
    test_result = tested_backend.gather(x, indices=[0, 1, 3], axis=-1)
    np.testing.assert_allclose(test_result, target_result)
    x = rand(3)
    target_result = target_backend.gather(x, condition=[True, False, True])
    test_result = tested_backend.gather(x, condition=[True, False, True])
    np.testing.assert_allclose(test_result, target_result)

    with pytest.raises(ValueError):
        result1 = target_backend.gather(x)
    with pytest.raises(ValueError):
        result2 = tested_backend.gather(x)


@pytest.mark.parametrize("return_counts", [False, True])
def test_backend_unique(tested_backend, target_backend, return_counts):
    tested_backend = K.construct_backend(tested_backend)
    target_backend = K.construct_backend(target_backend)
    x = np.random.randint(10, size=(10,))
    target_result = target_backend.unique(x, return_counts=return_counts)
    test_result = tested_backend.unique(x, return_counts=return_counts)
    if return_counts:
        idx = np.argsort(test_result[0])
        np.testing.assert_allclose(np.array(test_result[0])[idx], target_result[0])
        np.testing.assert_allclose(np.array(test_result[1])[idx], target_result[1])
    else:
        idx = np.argsort(test_result)
        np.testing.assert_allclose(np.array(test_result)[idx], target_result)
