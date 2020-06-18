import numpy as np
import pytest
from qibo.hamiltonians import XXZ


def test_hamiltonian_overloading():
    """Test basic hamiltonian overloading."""

    def transformation_a(a, b):
        return a + 0.1 * b
    def transformation_b(a, b):
        return 2 * a - b * 3.5

    H1 = XXZ(nqubits=2, delta=0.5)
    H2 = XXZ(nqubits=2, delta=1)

    hH1 = transformation_a(H1.hamiltonian, H2.hamiltonian)
    hH2 = transformation_b(H1.hamiltonian, H2.hamiltonian)

    HT1 = transformation_a(H1, H2)
    HT2 = transformation_b(H1, H2)

    np.testing.assert_allclose(hH1, HT1.hamiltonian)
    np.testing.assert_allclose(hH2, HT2.hamiltonian)


def test_hamiltonian_runtime_errors():
    """Testing hamiltonian runtime errors."""
    H1 = XXZ(nqubits=2, delta=0.5)
    H2 = XXZ(nqubits=3, delta=0.1)

    with pytest.raises(RuntimeError):
        R = H1 + H2
    with pytest.raises(RuntimeError):
        R = H1 - H2


def test_hamiltonian_notimplemented_errors():
    """Testing hamiltonian not implemented errors."""
    H1 = XXZ(nqubits=2, delta=0.5)
    H2 = XXZ(nqubits=2, delta=0.1)

    with pytest.raises(NotImplementedError):
        R = H1 * H2
    with pytest.raises(NotImplementedError):
        R = H1 - 1
    with pytest.raises(NotImplementedError):
        R = 2.5 + H2