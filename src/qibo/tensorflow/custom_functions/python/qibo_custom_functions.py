"""Use custom function from C."""
import numpy as np
from cffi import FFI
from tensorflow.python.platform import resource_loader

ffi = FFI()
ffi.cdef("""
void binary_matrix(unsigned char *in_array, int nqubits);
""")
custom_functions = ffi.dlopen(
    resource_loader.get_path_to_datafile('_qibo_custom_functions.so'))


def binary_matrix(nqubits):
    """Computes binary mask.

    Arguments:
        nqubits (int): number of qubits

    Returns:
        A numpy matrix with binary mask indexes.
    """
    mask = np.zeros(np.int64(2**nqubits*nqubits), dtype=np.ubyte)
    pmask = ffi.cast("unsigned char *", mask.ctypes.data)
    custom_functions.binary_matrix(pmask, nqubits)
    return mask.reshape(-1, nqubits)