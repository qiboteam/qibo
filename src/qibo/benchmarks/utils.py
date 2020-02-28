"""
Various utilities used by the benchmark scripts.
"""
import h5py
import numpy as np
from typing import Dict, List


def update_file(file_path: str, logs: Dict[str, List]):
    """Updates (or creates) an `.h5` file with the current logs.

    Args:
        file_path: Full path of the file to be created/updated.
            This should end in `.h5`.
        logs: Dictionary that contains the data to be written in the file.
    """
    file = h5py.File(file_path, "w")
    for k, v in logs.items():
        file[k] = np.array(v)
    file.close()


def random_state(nqubits: int) -> np.ndarray:
    """Generates a random state."""
    x = np.random.random(2**nqubits) + 1j * np.random.random(2**nqubits)
    return x / np.sqrt((np.abs(x)**2).sum())


def parse_nqubits(nqubits_str: str) -> List[int]:
    """Transforms a string that specifies number of qubits to list.

    Supported string formats are the following:
        * 'a-b' with a and b integers.
            Then the returned list is range(a, b + 1).
        * 'a,b,c,d,...' with a, b, c, d, ... integers.
            Then the returned list is [a, b, c, d]
    """
    # TODO: Support usage of both `-` and `,` in the same string.
    if "-" in nqubits_str:
        if "," in nqubits_str:
            raise ValueError("String that specifies qubits cannot contain "
                             "both , and -.")

        nqubits_split = nqubits_str.split("-")
        if len(nqubits_split) != 2:
            raise ValueError("Invalid string that specifies nqubits "
                             "{}.".format(nqubits_str))

        n_start, n_end = nqubits_split
        return list(range(int(n_start), int(n_end) + 1))

    return [int(x) for x in nqubits_str.split(",")]