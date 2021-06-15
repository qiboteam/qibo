import pytest
import numpy as np
from qibo import K
from numpy.random import random as rand


METHODS = [
    ("to_complex", [rand(3), rand(3)]),
    ("cast", [rand(4)]),
    ("diag", [rand(4)]),
    ("copy", [rand(5)]),
    ("zeros_like", [rand((4, 4))]),
    ("ones_like", [rand((4, 4))]),
    ("real", [rand(5)]),
    ("imag", [rand(5)]),
    ("conj", [rand(5)]),
    ("exp", [rand(5)]),
    ("sin", [rand(5)]),
    ("cos", [rand(5)]),
    ("square", [rand(5)]),
    ("sqrt", [rand(5)]),
    ("log", [rand(5)]),
    ("abs", [rand(5)]),
    ("trace", [rand((6, 6))]),
    ("sum", [rand((4, 4))]),
    ("matmul", [rand((4, 6)), rand((6, 5))]),
    ("outer", [rand((4,)), rand((3,))]),
    ("eigvalsh", [rand((4, 4))]),
    ("less", [rand(10), rand(10)]),
    ("array_equal", [rand(10), rand(10)]),
    ("eye", [5]),
    ("zeros", [(2, 3)]),
    ("ones", [(2, 3)]),
    ("einsum", ["xy,axby->ab", rand((2, 2)), rand(4 * (2,))]),
    ("tensordot", [rand((2, 2)), rand(4 * (2,)), [[0, 1], [1, 3]]]),
    ("transpose", [rand((3, 3, 3)), [0, 2, 1]]),
    ("gather_nd", [rand((5, 3)), [0, 1]]),
    ("expm", [rand((4, 4))]),
    ("mod", [np.random.randint(10), np.random.randint(2, 10)]),
    ("right_shift", [np.random.randint(10), np.random.randint(10)]),
    ("kron", [rand((4, 4)), rand((5, 5))]),
    ("inv", [rand((4, 4))]),
    ("initial_state", [5, True]),
    ("initial_state", [3, False])
]
@pytest.mark.parametrize("method,args", METHODS)
def test_backend_methods_list(tested_backend, target_backend, method, args):
    tested_backend = K.construct_backend(tested_backend)
    target_backend = K.construct_backend(target_backend)
    tested_func = getattr(tested_backend, method)
    target_func = getattr(target_backend, method)
    target_result = target_func(*args)
    if tested_backend.name == "qibojit" and tested_backend.op.get_backend() == "cupy":
        new_args = (tested_backend.cast(v) if isinstance(v, np.ndarray) else v
                    for v in args)
        tested_result = tested_func(*new_args)
        tested_result = tested_backend.to_numpy(tested_result)
    else:
        try:
            tested_result = tested_func(*args)
        except NotImplementedError:
            with pytest.raises(NotImplementedError):
                tested_func(*args)
            return
    np.testing.assert_allclose(tested_result, target_result)


@pytest.mark.parametrize("method,kwargs", [
    ("reshape", {"x": rand(4), "shape": (2, 2)}),
    ("expand_dims", {"x": rand((5, 5)), "axis": 1}),
    ("range", {"start": 0, "finish": 10, "step": 2}),
    ("range", {"start": 0, "finish": 10, "step": 2, "dtype": "DTYPE"}),
    ("pow", {"base": rand(5), "exponent": 4}),
    ("sum", {"x": rand((4, 4, 3)), "axis": 1}),
    ("squeeze", {"x": rand((5, 1, 2)), "axis": 1})
])
def test_backend_methods_dict(tested_backend, target_backend, method, kwargs):
    tested_backend = K.construct_backend(tested_backend)
    target_backend = K.construct_backend(target_backend)
    tested_func = getattr(tested_backend, method)
    target_func = getattr(target_backend, method)
    target_result = target_func(**kwargs)
    if tested_backend.name == "qibojit" and tested_backend.op.get_backend() == "cupy":
        new_kwargs = {k: tested_backend.cast(v) if isinstance(v, np.ndarray) else v
                      for k, v in kwargs.items()}
        tested_result = tested_func(**new_kwargs)
        tested_result = tested_backend.to_numpy(tested_result)
    else:
        tested_result = tested_func(**kwargs)
    np.testing.assert_allclose(tested_result, target_result)


def test_backend_concatenate(tested_backend, target_backend):
    tested_backend = K.construct_backend(tested_backend)
    target_backend = K.construct_backend(target_backend)
    tensors = [rand((2, 3)), rand((2, 4))]
    target_result = target_backend.concatenate(tensors, axis=1)
    if tested_backend.name == "qibojit" and tested_backend.op.get_backend() == "cupy":
        gtensors = [tested_backend.cast(x) for x in tensors]
        tested_result = tested_backend.concatenate(gtensors, axis=1)
        tested_result = tested_backend.to_numpy(tested_result)
    else:
        tested_result = tested_backend.concatenate(tensors, axis=1)
    np.testing.assert_allclose(tested_result, target_result)


def test_backend_stack(tested_backend, target_backend):
    tested_backend = K.construct_backend(tested_backend)
    target_backend = K.construct_backend(target_backend)
    tensors = [rand(4), rand(4)]
    target_result = target_backend.stack(tensors)
    if tested_backend.name == "qibojit" and tested_backend.op.get_backend() == "cupy":
        gtensors = [tested_backend.cast(x) for x in tensors]
        tested_result = tested_backend.stack(gtensors)
        tested_result = tested_backend.to_numpy(tested_result)
    else:
        tested_result = tested_backend.stack(tensors)
    np.testing.assert_allclose(tested_result, target_result)


def test_backend_eigh(tested_backend, target_backend):
    tested_backend = K.construct_backend(tested_backend)
    target_backend = K.construct_backend(target_backend)
    m = rand((5, 5))
    eigvals2, eigvecs2 = target_backend.eigh(m)
    if tested_backend.name == "qibojit" and tested_backend.op.get_backend() == "cupy":
        eigvals1, eigvecs1 = tested_backend.eigh(tested_backend.cast(m))
        eigvals1 = tested_backend.to_numpy(eigvals1)
        eigvecs1 = tested_backend.to_numpy(eigvecs1)
    else:
        eigvals1, eigvecs1 = tested_backend.eigh(m)
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


def test_hardware_backend_import():
    # TODO: Remove this test once the hardware backend is tested properly
    from qibo.backends.hardware import IcarusQBackend
