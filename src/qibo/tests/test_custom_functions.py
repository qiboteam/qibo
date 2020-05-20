"""
Testing C custom functions
"""
import pytest
import numpy as np
import itertools
from qibo.tensorflow import custom_functions as op


@pytest.mark.parametrize("nqubits", [4, 15])
def test_binary_matrix(nqubits):
  """Check that binary matrix from C is compatible with itertools."""
  op_mask = op.binary_matrix(nqubits)
  it_mask = np.array(list(itertools.product([0, 1], repeat=nqubits)))
  np.testing.assert_array_equal(np.sort(op_mask), np.sort(it_mask))
