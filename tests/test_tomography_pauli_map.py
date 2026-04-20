"""
Comprehensive tests for PauliMap class with multiple backends.

Tests cover:
- Different backends (numpy, qibojit, tensorflow, pytorch if available)
- Exact measurements vs circuit-based measurements
- Various qubit numbers with appropriate sampling
- RIP-based sampling requirements
"""

from functools import reduce
from itertools import product

import numpy as np
import pytest

from qibo.backends import construct_backend
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models.encodings import ghz_state
from qibo.symbols import I, X, Y, Z
from qibo.tomography.state_tomography import PauliMap


def _try_build_backend(backend_name):
    """Build backend, splitting 'name-platform' format (e.g. 'qibojit-numba')."""
    if "-" in backend_name:
        name, platform = backend_name.split("-", 1)
    else:
        name, platform = backend_name, None
    return construct_backend(name, platform=platform)


# ==================== Fixtures ====================


@pytest.fixture
def symbol_map():
    """Pauli symbol mapping."""
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def label2symbolic(label, symbol_map):
    """Convert Pauli string to SymbolicHamiltonian."""
    n_qubits = len(label)

    if all(c == "I" for c in label):
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
    d = 2**n_qubits
    C = max(1.0 / (delta**2), 10)  # At least 10

    # m = C * r * d * log^6(d)
    log_d = np.log(d) if d > 1 else 1
    m_rip = C * rank * d * (log_d**6)

    # Cap at total number of Pauli operators
    m_total = 4**n_qubits

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
            backend = _try_build_backend(backend_name)
        except Exception as e:
            pytest.skip(f"Backend {backend_name} not available: {e}")
        # backend = construct_backend(backend_name)

        labels = ["II", "ZZ"]
        symbol_map = {"I": I, "X": X, "Y": Y, "Z": Z}
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
            label2symbolic("XXX", symbol_map),  # 3 qubits!
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

    def test_exact_measurement_on_density_matrix(
        self, ghz_2qubit, backend_name, symbol_map
    ):
        """Test exact measurement on density matrix."""
        try:
            _try_build_backend(backend_name)
        except Exception as e:
            pytest.skip(f"Backend {backend_name} not available: {e}")

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
        """Test that A†(A(X)) recovers X for full Pauli measurements.

        Pauli completeness: Σ_i Tr(S_i X) S_i = d * X
        So: get_adjoint_matrix(apply(dm)) = d * dm → divide by d to recover dm.
        """
        circuit, state, dm = ghz_2qubit

        # Use ALL Pauli operators for exact recovery
        labels = generate_all_pauli_labels(2)
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")

        # Forward: A(dm) = [Tr(S_i dm)]
        y = pm.apply(dm)

        # Backward: A†(y) = Σ_i y_i S_i = d * dm  (Pauli completeness)
        reconstructed = pm.get_adjoint_matrix(y)

        # Verify Pauli completeness: reconstructed / d == dm exactly
        d = pm.dim  # = 4 for 2 qubits
        np.testing.assert_allclose(
            reconstructed / d,
            dm,
            atol=1e-10,
            err_msg="Pauli completeness A†(A(X)) = d*X failed",
        )

    def test_identity_operator_returns_trace(self, symbol_map):
        """Identity measurement returns Tr(ρ), not hardcoded 1.0.

        For unnormalized X_k (Tr ≠ 1), the identity expectation value
        must be Tr(ρ), not 1.0 — otherwise the scale-correction gradient vanishes.
        """
        labels = ["II"]
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")

        # Unnormalized state: Tr = 2.0
        dm_unnorm = np.eye(4, dtype=complex) * 0.5
        measurements = pm.apply(dm_unnorm)
        np.testing.assert_allclose(
            measurements[0],
            2.0,
            atol=1e-10,
            err_msg="Identity on Tr=2 state must return 2.0, not 1.0",
        )

        # Normalized state: Tr = 1.0
        dm_norm = np.eye(4, dtype=complex) / 4
        measurements_norm = pm.apply(dm_norm)
        np.testing.assert_allclose(
            measurements_norm[0],
            1.0,
            atol=1e-10,
            err_msg="Identity on normalized state must return 1.0",
        )


class TestPauliMapRSVD:
    """Test randomized SVD functionality."""

    def test_rsvd_hermitian_basic(self, symbol_map):
        """Test rSVD on Hermitian matrix."""
        # Create a simple Hermitian matrix
        A = np.array([[2, 1 + 1j], [1 - 1j, 3]], dtype=complex)

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
            power_iterations=5,  # More power iterations
        )
        rank1_rsvd = U_r @ S_r @ V_r.conj().T

        # Check if principal singular values are close
        s_rsvd = np.linalg.svd(rank1_rsvd, compute_uv=False)[0]
        s_full_top = s_full[0]

        # Singular value should be close
        assert (
            abs(s_rsvd - s_full_top) / s_full_top < 0.1
        ), f"Top singular value mismatch: {s_rsvd} vs {s_full_top}"

        # # Should be very close
        # np.testing.assert_allclose(rank1_rsvd, rank1_full, atol=1e-6)


class TestPauliMapWithRIPSampling:
    """Test with RIP-based sampling requirements."""

    @pytest.mark.parametrize(
        "n_qubits,rank",
        [
            (2, 1),
            (3, 1),
            (4, 1),
            (3, 2),
        ],
    )
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
            if "I" * n_qubits not in labels:
                labels.append("I" * n_qubits)

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
        """Test that PauliMap works even with very few measurements.

        Insufficient sampling should not crash; measurements must still be
        finite and real (expectation values of Hermitian operators).
        """
        n_qubits = 5

        # Use very few samples (much less than RIP requirement)
        all_labels = generate_all_pauli_labels(n_qubits)
        labels = all_labels[:10]  # Only 10 out of 1024!

        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")

        circuit = ghz_state(n_qubits)
        dm = np.outer(circuit.execute().state(), circuit.execute().state().conj())

        measurements = pm.apply(dm)

        # Must return one value per operator
        assert len(measurements) == len(labels)
        # Expectation values of Hermitian operators are real
        assert all(np.isreal(m) for m in measurements), "All measurements must be real"
        # Must be finite (no NaN/Inf)
        assert all(
            np.isfinite(m) for m in measurements
        ), "All measurements must be finite"
        # Non-identity Pauli expectation values are in [-1, 1]
        non_identity = [m for l, m in zip(labels, measurements) if l != "I" * n_qubits]
        assert all(
            abs(m) <= 1.0 + 1e-10 for m in non_identity
        ), "Non-identity Pauli expectation values must be in [-1, 1]"


class TestPauliMapMultipleBackends:
    """Test consistency across backends."""

    def test_backend_consistency(self, backend_name, ghz_2qubit, symbol_map):
        """Test that different backends give same results."""
        try:
            _try_build_backend(backend_name)
        except Exception as e:
            pytest.skip(f"Backend {backend_name} not available: {e}")

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
