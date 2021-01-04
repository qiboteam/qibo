import pytest
import numpy as np
import tensorflow as tf
from qibo import backends
from numpy.random import random as rand


_METHODS = [
    ("cast", [rand(4)]), ("diag", [rand(4)]),
    ("reshape", {"x": rand(4), "shape": (2, 2)}),
    ("stack", [[rand(4), rand(4)]]),
    ("concatenate", {"x": [rand((2, 3)), rand((2, 4))], "axis": 1}),
    ("expand_dims", {"x": rand((5, 5)), "axis": 1}), ("copy", [rand(5)]),
    ("range", {"start": 0, "finish": 10, "step": 2}),
    ("range", {"start": 0, "finish": 10, "step": 2, "dtype": "DTYPE"}),
    ("eye", [5]), ("zeros", [(2, 3)]), ("ones", [(2, 3)]),
    ("zeros_like", [rand((4, 4))]), ("ones_like", [rand((4, 4))]),
    ("real", [rand(5)]), ("imag", [rand(5)]), ("conj", [rand(5)]),
    ("mod", [np.random.randint(10), np.random.randint(2, 10)]),
    ("right_shift", [np.random.randint(10), np.random.randint(10)]),
    ("exp", [rand(5)]), ("sin", [rand(5)]), ("cos", [rand(5)]),
    ("pow", {"base": rand(5), "exponent": 4}),
    ("square", [rand(5)]), ("sqrt", [rand(5)]),
    ("log", [rand(5)]), ("abs", [rand(5)]),
    ("expm", [rand((4, 4))]), ("trace", [rand((6, 6))]),
    ("sum", [rand((4, 4))]), ("sum", {"x": rand((4, 4, 3)), "axis": 1}),
    ("matmul", [rand((4, 6)), rand((6, 5))]),
    ("outer", [rand((4,)), rand((3,))]),
    # ("kron", [rand((4, 4)), rand((5, 5))]),
    ("einsum", ["xy,axby->ab", rand((2, 2)), rand(4 * (2,))]),
    ("tensordot", [rand((2, 2)), rand(4 * (2,)), [[0, 1], [1, 3]]]),
    ("transpose", [rand((3, 3, 3)), [0, 2, 1]]),
    # ("inv", [rand((4, 4))]),
    ("eigvalsh", [rand((4, 4))]),
    ("unique", [np.random.randint(10, size=(10,))]),
    ("array_equal", [rand(10), rand(10)]),
    ("gather_nd", [rand((5, 3)), [0, 1]]),
    ("initial_state", [5]),
]
@pytest.mark.parametrize("names", [("numpy", "tensorflow")])
@pytest.mark.parametrize("method,kwargs", _METHODS)
def test_backend_methods(names, method, kwargs):
    backend1 = backends.construct_backend(names[0])
    backend2 = backends.construct_backend(names[1])
    if isinstance(kwargs, dict):
        result1 = getattr(backend1, method)(**kwargs)
        result2 = getattr(backend2, method)(**kwargs)
    else:
        result1 = getattr(backend1, method)(*kwargs)
        result2 = getattr(backend2, method)(*kwargs)
    np.testing.assert_allclose(result1, result2)


def test_backend_errors():
    with pytest.raises(ValueError):
        bk = backends.construct_backend("test")
    with pytest.raises(ValueError):
        backends.set_backend("a_b_c")


@pytest.mark.parametrize("names", [("numpy", "tensorflow")])
def test_backend_eigh(names):
    backend1 = backends.construct_backend(names[0])
    backend2 = backends.construct_backend(names[1])
    m = rand((5, 5))
    eigvals1, eigvecs1 = backend1.eigh(m)
    eigvals2, eigvecs2 = backend2.eigh(m)
    np.testing.assert_allclose(eigvals1, eigvals2)
    np.testing.assert_allclose(np.abs(eigvecs1), np.abs(eigvecs2))


@pytest.mark.parametrize("names", [("numpy", "tensorflow")])
def test_backend_compile(names):
    backend1 = backends.construct_backend(names[0])
    backend2 = backends.construct_backend(names[1])
    func = lambda x: x + 1
    x = rand(5)
    cfunc1 = backend1.compile(func)
    cfunc2 = backend2.compile(func)
    np.testing.assert_allclose(cfunc1(x), cfunc2(x))


def test_backend_gather():
    np_backend = backends.construct_backend("numpy")
    tf_backend = backends.construct_backend("tensorflow")
    x = rand(5)
    result1 = np_backend.gather(x, indices=[0, 1, 3])
    result2 = tf_backend.gather(x, indices=[0, 1, 3])
    np.testing.assert_allclose(result1, result2)
    x = rand((5, 5))
    result1 = np_backend.gather(x, indices=[0, 1, 3], axis=-1)
    result2 = tf_backend.gather(x, indices=[0, 1, 3], axis=-1)
    np.testing.assert_allclose(result1, result2)
    x = rand(3)
    result1 = np_backend.gather(x, condition=[True, False, True])
    result2 = tf_backend.gather(x, condition=[True, False, True])
    np.testing.assert_allclose(result1, result2[:, 0])
    with pytest.raises(ValueError):
        result1 = np_backend.gather(x)
    with pytest.raises(ValueError):
        result2 = tf_backend.gather(x)


@pytest.mark.parametrize("name,dtype",
                         [("NumpyMatrices", np.complex64),
                          ("TensorflowMatrices", tf.complex64),
                          ("TensorflowMatrices", tf.complex128)])
def test_matrix_initialization(name, dtype):
    from qibo.backends import matrices
    targetm = matrices.NumpyMatrices(np.complex128)
    m = getattr(matrices, name)(dtype)
    for g in ["I", "H", "X", "Y", "Z", "CNOT", "SWAP", "TOFFOLI"]:
        target = getattr(targetm, g)
        final = np.reshape(getattr(m, g), target.shape)
        np.testing.assert_allclose(final, target)
