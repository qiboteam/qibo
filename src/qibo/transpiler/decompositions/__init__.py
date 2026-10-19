from importlib.util import find_spec

from qibo.transpiler.decompositions.abstract import GateDecompositions
from qibo.transpiler.decompositions.cz_decompositions import cz_dec, native_cz_dec
from qibo.transpiler.decompositions.direction_decompositions import direction_dec
from qibo.transpiler.decompositions.gpi2_decompositions import gpi2_dec
from qibo.transpiler.decompositions.iswap_decompositions import iswap_dec
from qibo.transpiler.decompositions.opt_decompositions import opt_dec
from qibo.transpiler.decompositions.standard_decompositions import (
    standard_decompositions,
)
from qibo.transpiler.decompositions.u3_decompositions import u3_dec

__all__ = [
    "GateDecompositions",
    "cz_dec",
    "direction_dec",
    "gpi2_dec",
    "iswap_dec",
    "native_cz_dec",
    "opt_dec",
    "standard_decompositions",
    "u3_dec",
]

# the Clifford+T decomposition needs these optional packages
if find_spec("pygridsynth") and find_spec("mpmath"):
    from qibo.transpiler.decompositions.clifford_plus_t_decompositions import (
        clifford_plus_t as clifford_plus_t,
    )

    __all__.append("clifford_plus_t")
