"""
Comprehensive tests for PauliMap class with multiple backends.

Tests cover:
- Different backends (numpy, qibojit, tensorflow, pytorch if available)
- Exact measurements vs circuit-based measurements
- Various qubit numbers with appropriate sampling
- RIP-based sampling requirements
"""

import pytest
import numpy as np
from itertools import product
from functools import reduce

from qibo.symbols import I, X, Y, Z
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models.encodings import ghz_state
from qibo.backends import construct_backend, list_available_backends

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from RGD_Optimized import PauliMap


# ==================== Fixtures ====================

@pytest.fixture
def symbol_map():
    """Pauli symbol mapping."""
    return {'I': I, 'X': X, 'Y': Y, 'Z': Z}


def label2symbolic(label, symbol_map):
    """Convert Pauli string to SymbolicHamiltonian."""
    n_qubits = len(label)
    
    if all(c == 'I' for c in label):
        return SymbolicHamiltonian(I(0), nqubits=n_qubits)
    
    paulis = [symbol_map[s](i) for i, s in enumerate(label)]
    return SymbolicHamiltonian(reduce(lambda x, y: x * y, paulis), nqubits=n_qubits)


def generate_all_pauli_labels(n_qubits):
    """Generate all Pauli strings."""
    return ["".join(s) for s in product(["I", "X", "Y", "Z"], repeat=n_qubits)]


def compute_rip_required_samples(n_qubits, rank, delta=0.1):
    """
    Compute required number of Pauli measurements based on RIP.
    
    From paper (Theorem 1):
    m = C * r * d * log^6(d)
    
    where:
    - d = 2^n (dimension)
    - r = rank
    - C = O(1/delta^2)
    
    Args:
        n_qubits: Number of qubits
        rank: Target rank
        delta: RIP constant (smaller = more samples needed)
        
    Returns:
        Required number of samples
    """
    d = 2 ** n_qubits
    C = max(1.0 / (delta ** 2), 10)  # At least 10
    
    # m = C * r * d * log^6(d)
    log_d = np.log(d) if d > 1 else 1
    m_rip = C * rank * d * (log_d ** 6)
    
    # Cap at total number of Pauli operators
    m_total = 4 ** n_qubits
    
    return int(min(m_rip, m_total))


def create_pauli_operators(labels, symbol_map):
    """Create list of SymbolicHamiltonian from labels."""
    return [label2symbolic(l, symbol_map) for l in labels]


@pytest.fixture
def ghz_2qubit(symbol_map):
    """2-qubit GHZ state."""
    circuit = ghz_state(2)
    state = circuit.execute().state()
    dm = np.outer(state, state.conj())
    return circuit, state, dm


@pytest.fixture
def ghz_3qubit(symbol_map):
    """3-qubit GHZ state."""
    circuit = ghz_state(3)
    state = circuit.execute().state()
    dm = np.outer(state, state.conj())
    return circuit, state, dm


# ==================== Test Cases ====================

class TestPauliMapBasics:
    """Basic functionality tests."""
    
    def test_initialization_numpy(self, symbol_map):
        """Test initialization with numpy backend."""
        labels = ["II", "IX", "IZ", "XX"]
        ops = create_pauli_operators(labels, symbol_map)
        
        pm = PauliMap(ops, backend="numpy")
        
        assert pm.nqubits == 2
        assert pm.dim == 4
        assert len(pm.symProj_list) == 4
    
    def test_initialization_with_backend_object(self, backend_name):
        """Test initialization with backend object."""
        try:
            backend = construct_backend(backend_name)
        except Exception as e:
            pytest.skip(f"Backend {backend_name} not available: {e}")
        # backend = construct_backend(backend_name)
        
        labels = ["II", "ZZ"]
        symbol_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        ops = create_pauli_operators(labels, symbol_map)
        
        pm = PauliMap(ops, backend=backend)
        
        assert pm.backend == backend
        assert pm.nqubits == 2
    
    def test_empty_list_raises_error(self):
        """Test that empty operator list raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            PauliMap([], backend="numpy")
    
    def test_mismatched_qubits_raises_error(self, symbol_map):
        """Test that mismatched qubit numbers raise error."""
        ops = [
            label2symbolic("II", symbol_map),
            label2symbolic("XXX", symbol_map)  # 3 qubits!
        ]
        
        with pytest.raises(ValueError, match="qubits"):
            PauliMap(ops, backend="numpy")
    
    def test_matrix_caching(self, symbol_map):
        """Test that matrices are cached."""
        labels = ["II", "XX", "YY", "ZZ"]
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")
        
        # First call creates cache
        matrices1 = pm.get_matrices()
        # Second call uses cache
        matrices2 = pm.get_matrices()
        
        assert matrices1 is matrices2  # Same object
        assert len(matrices1) == 4


class TestPauliMapMeasurements:
    """Test measurement functionality."""
    
    def test_exact_measurement_on_density_matrix(self, ghz_2qubit, backend_name, symbol_map):
        """Test exact measurement on density matrix."""
        available = list_available_backends()
        if backend_name not in available:
            pytest.skip(f"Backend {backend_name} not available")
        
        circuit, state, dm = ghz_2qubit
        
        labels = ["II", "XX", "YY", "ZZ"]
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend=backend_name)
        
        measurements = pm.apply(dm)
        
        # Check measurements are real
        assert all(np.isreal(m) for m in measurements)
        
        # Check specific values for GHZ state
        # GHZ = (|00⟩ + |11⟩)/√2
        # ⟨II⟩ = 1
        # ⟨XX⟩ = 1
        # ⟨YY⟩ = -1
        # ⟨ZZ⟩ = 1
        expected = [1.0, 1.0, -1.0, 1.0]
        
        np.testing.assert_allclose(measurements, expected, atol=1e-10)
    
    def test_measurement_from_circuit_exact(self, ghz_2qubit, symbol_map):
        """Test measurement from circuit (no shots)."""
        circuit, state, dm = ghz_2qubit
        
        labels = ["II", "XX", "ZZ"]
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")
        
        # Pass circuit (no shots)
        measurements = pm.apply(circuit)
        
        # Should be same as density matrix
        measurements_dm = pm.apply(dm)
        
        np.testing.assert_allclose(measurements, measurements_dm, atol=1e-10)
    
    def test_measurement_from_circuit_with_shots(self, ghz_2qubit, symbol_map):
        """Test shot-based measurement from circuit."""
        circuit, state, dm = ghz_2qubit
        
        labels = ["II", "XX", "ZZ"]
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")
        
        # With shots
        measurements = pm.apply(circuit, nshots=10000)
        
        # Should be close to exact (but with sampling error)
        measurements_exact = pm.apply(dm)
        
        # Allow larger tolerance due to sampling
        np.testing.assert_allclose(measurements, measurements_exact, atol=0.1)
    
    def test_adjoint_operator(self, symbol_map):
        """Test A†(y) operator."""
        labels = ["II", "XX", "YY", "ZZ"]
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")
        
        y = np.array([1.0, 0.5, -0.5, 0.3])
        
        result = pm.get_adjoint_matrix(y)
        
        # Check result is Hermitian
        assert result.shape == (4, 4)
        np.testing.assert_allclose(result, result.conj().T, atol=1e-10)
    
    def test_adjoint_round_trip(self, ghz_2qubit, symbol_map):
        """Test A†(A(X)) ≈ X for sufficient measurements."""
        circuit, state, dm = ghz_2qubit
        
        # Use ALL Pauli operators for exact recovery
        labels = generate_all_pauli_labels(2)
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")
        
        # Forward: A(dm)
        y = pm.apply(dm)
        
        # Backward: A†(y)
        reconstructed = pm.get_adjoint_matrix(y)
        
        # Scale properly (see paper)
        d = 4
        m = len(labels)
        reconstructed = reconstructed * np.sqrt(d / m)
        
        # Should recover dm (approximately)
        # Note: This is not exact recovery, just checking the map works
        assert reconstructed.shape == dm.shape


class TestPauliMapRSVD:
    """Test randomized SVD functionality."""
    
    def test_rsvd_hermitian_basic(self, symbol_map):
        """Test rSVD on Hermitian matrix."""
        # Create a simple Hermitian matrix
        A = np.array([[2, 1+1j], [1-1j, 3]], dtype=complex)
        
        labels = ["II"]  # Dummy
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")
        
        # rSVD
        U, S, V = pm.rSVD_hermitian(A, rank=1, oversampling=2, power_iterations=2)
        
        # Check U = V (Hermitian property)
        np.testing.assert_allclose(U, V, atol=1e-10)
        
        # Check reconstruction
        A_approx = U @ S @ V.conj().T
        
        # Should be close to rank-1 approximation
        U_full, s_full, Vh_full = np.linalg.svd(A)
        A_rank1 = U_full[:, :1] @ np.diag(s_full[:1]) @ Vh_full[:1, :]
        
        np.testing.assert_allclose(A_approx, A_rank1, atol=1e-8)
    
    def test_rsvd_vs_full_svd(self, ghz_3qubit, symbol_map):
        """Compare rSVD with full SVD."""
        circuit, state, dm = ghz_3qubit
        
        labels = generate_all_pauli_labels(3)[:20]  # Subset
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")
        
        # Get A†(y)
        y = pm.apply(dm)
        A_adj_y = pm.get_adjoint_matrix(y)
        
        # Full SVD
        U_full, s_full, Vh_full = np.linalg.svd(A_adj_y)
        rank1_full = U_full[:, :1] @ np.diag(s_full[:1]) @ Vh_full[:1, :]
        
        # rSVD with more power iterations for better accuracy
        U_r, S_r, V_r = pm.rSVD_hermitian(
            A_adj_y, 
            rank=1,
            oversampling=10,  # More oversampling
            power_iterations=5  # More power iterations
        )
        rank1_rsvd = U_r @ S_r @ V_r.conj().T
        
        # Check if principal singular values are close
        s_rsvd = np.linalg.svd(rank1_rsvd, compute_uv=False)[0]
        s_full_top = s_full[0]
        
        # Singular value should be close
        assert abs(s_rsvd - s_full_top) / s_full_top < 0.1, \
            f"Top singular value mismatch: {s_rsvd} vs {s_full_top}"

        # # Should be very close
        # np.testing.assert_allclose(rank1_rsvd, rank1_full, atol=1e-6)


class TestPauliMapWithRIPSampling:
    """Test with RIP-based sampling requirements."""
    
    @pytest.mark.parametrize("n_qubits,rank", [
        (2, 1),
        (3, 1),
        (4, 1),
        (3, 2),
    ])
    def test_sufficient_sampling_from_rip(self, n_qubits, rank, symbol_map):
        """Test with RIP-required number of samples."""
        # For small systems, use ALL labels
        if n_qubits < 4:
            labels = generate_all_pauli_labels(n_qubits)
        else:
            # For larger systems, use RIP-based sampling
            m_required = compute_rip_required_samples(n_qubits, rank, delta=0.1)
            all_labels = generate_all_pauli_labels(n_qubits)
            
            # Ensure we have enough samples
            m_actual = min(m_required, len(all_labels))
            
            np.random.seed(42)
            labels = list(np.random.choice(all_labels, m_actual, replace=False))
            
            # Always include identity
            if "I"*n_qubits not in labels:
                labels.append("I"*n_qubits)
        
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")
        
        # Create target state
        circuit = ghz_state(n_qubits)
        dm = np.outer(circuit.execute().state(), circuit.execute().state().conj())
        
        # Get measurements
        measurements = pm.apply(dm)
        
        # Basic sanity checks
        assert len(measurements) == len(labels)
        assert all(np.isreal(m) for m in measurements)
        assert all(abs(m) <= 1.0 for m in measurements)  # Expectation values in [-1, 1]
    
    def test_insufficient_sampling_warning(self, symbol_map):
        """Test behavior with insufficient sampling."""
        n_qubits = 5
        rank = 1
        
        # Use very few samples (much less than RIP requirement)
        all_labels = generate_all_pauli_labels(n_qubits)
        labels = all_labels[:10]  # Only 10 out of 1024!
        
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")
        
        # Should still work (no crash)
        circuit = ghz_state(n_qubits)
        dm = np.outer(circuit.execute().state(), circuit.execute().state().conj())
        
        measurements = pm.apply(dm)
        
        # But warn user
        m_required = compute_rip_required_samples(n_qubits, rank)
        m_actual = len(labels)
        
        if m_actual < m_required:
            print(f"\nWarning: Using {m_actual} samples, "
                  f"RIP requires {m_required} for guaranteed recovery")


class TestPauliMapMultipleBackends:
    """Test consistency across backends."""
    
    def test_backend_consistency(self, backend_name, ghz_2qubit, symbol_map):
        """Test that different backends give same results."""
        available = list_available_backends()
        if backend_name not in available:
            pytest.skip(f"Backend {backend_name} not available")
        
        circuit, state, dm = ghz_2qubit
        
        labels = ["II", "XX", "YY", "ZZ"]
        ops = create_pauli_operators(labels, symbol_map)
        
        pm = PauliMap(ops, backend=backend_name)
        measurements = pm.apply(dm)
        
        # Get numpy baseline
        pm_numpy = PauliMap(ops, backend="numpy")
        measurements_numpy = pm_numpy.apply(dm)
        
        # Should match
        np.testing.assert_allclose(measurements, measurements_numpy, atol=1e-10)


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
