"""Module defining the Hamming-weight-preserving backend."""

import inspect
from functools import cache
from importlib.util import find_spec, module_from_spec

from qibo.config import raise_error


def HammingWeightBackend(engine=None):
    """Dynamically create a HammingWeightBackend class based on the selected backend."""

    from qibo.backends import construct_backend  # pylint: disable=C415

    if engine is None:
        from qibo.backends import (  # pylint: disable=C0415
            _check_backend,
            _get_engine_name,
        )

        engine = _get_engine_name(_check_backend(engine))

    backend = None  # needed for pylint
    if engine == "numpy":
        backend = construct_backend("numpy", platform=engine)
    elif engine in ["numba", "cupy", "cuquantum"]:
        backend = construct_backend("qibojit", platform=engine)
    elif engine in ["tensorflow", "pytorch"]:  # pragma: no cover
        backend = construct_backend("qiboml", platform=engine)
    else:  # pragma: no cover
        raise_error(
            NotImplementedError,
            f"Backend `{engine}` is not supported for "
            + "Hamming-weight-preserving circuit simulation.",
        )

    import qibo.backends._hamming_weight_operations as module

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
