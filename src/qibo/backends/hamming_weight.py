"""Module defining the Hamming-weight-preserving backend."""

import inspect
from functools import cache
from importlib.util import find_spec, module_from_spec

from qibo.config import raise_error


def HammingWeightBackend(engine=None):
    """Dynamically create a HammingWeightBackend class based on the selected backend."""

    if engine is None:
        from qibo.backends import (  # pylint: disable=C0415
            _check_backend,
            _get_engine_name,
        )

        engine = _get_engine_name(_check_backend(engine))

    backend = None  # needed for pylint
    if engine == "numpy":
        from qibo.backends import NumpyBackend  # pylint: disable=C0415

        backend = NumpyBackend()
    elif engine in "numba":
        from qibojit.backends import NumbaBackend  # pylint: disable=C0415

        backend = NumbaBackend()
    elif engine == "cupy":
        from qibojit.backends import CupyBackend  # pylint: disable=C0415

        backend = CupyBackend()
    elif engine == "cuquantum":
        from qibojit.backends import CuQuantumBackend  # pylint: disable=C0415

        backend = CuQuantumBackend()
    elif engine == "tensorflow":  # pragma: no cover
        from qiboml.backends import TensorflowBackend  # pylint: disable=E0401

        backend = TensorflowBackend()
    elif engine == "pytorch":  # pragma: no cover
        from qiboml.backends import PyTorchBackend  # pylint: disable=E0401

        backend = PyTorchBackend()
    else:  # pragma: no cover
        raise_error(
            NotImplementedError,
            f"Backend `{engine}` is not supported for "
            + "Hamming-weight-preserving circuit simulation.",
        )

    spec = find_spec("qibo.backends._hamming_weight_operations")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    methods = {
        name: func for name, func in inspect.getmembers(module, inspect.isfunction)
    }

    for method_name in ["_get_cached_strings", "_get_lexicographical_order"]:
        if method_name in methods:
            methods[method_name] = cache(methods[method_name])

    HWBackend = type(
        "HammingWeightBackend", (backend.__class__,), methods
    )  # pylint: disable=E0606

    hw_backend = HWBackend()
    hw_backend.name = "hamming_weight"
    hw_backend.platform = engine

    hw_backend.calculate_full_probabilities = backend.calculate_probabilities

    hw_backend._dict_cached_strings_one = {}
    hw_backend._dict_cached_strings_two = {}

    hw_backend._dict_indexes = None

    return hw_backend
