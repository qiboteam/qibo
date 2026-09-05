"""
Comprehensive tests for RGDOptimizer class.

Tests cover:
- Multiple backends
- All initialization methods (algorithm, random, random_rank)
- Exact vs shot-based measurements
- Various qubit numbers with RIP-based sampling
- Fast SVD vs standard SVD
- Numerical stability
"""

from functools import reduce
from itertools import product

import numpy as np
import pytest

from qibo.backends import construct_backend
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models.encodings import ghz_state
from qibo.symbols import I, X, Y, Z
from qibo.tomography.state_tomography import PauliMap, RGDOptimizer


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


def create_pauli_operators(labels, symbol_map):
    """Create list of SymbolicHamiltonian from labels."""
    return [label2symbolic(l, symbol_map) for l in labels]


def compute_rip_samples(n_qubits, rank, delta=0.1):
    """Compute RIP-required samples."""
    d = 2**n_qubits
    C = max(1.0 / (delta**2), 10)
    log_d = np.log(d) if d > 1 else 1
    m_rip = C * rank * d * (log_d**6)
    m_total = 4**n_qubits
    return int(min(m_rip, m_total))


# ==================== Test Cases ====================


class TestRGDOptimizerBasics:
    """Basic functionality tests."""

    def test_initialization_numpy(self, symbol_map):
        """Test optimizer initialization."""
        labels = ["II", "XX", "ZZ"]
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")

        optimizer = RGDOptimizer(pauli_map=pm, rank=1, use_fast_svd=True)

        assert optimizer.rank == 1
        assert optimizer.use_fast_svd == True
        assert optimizer.iteration == 0
        assert optimizer.converged == False

    def test_initialization_with_target(self, symbol_map):
        """Test initialization with target density matrix."""
        labels = ["II", "XX"]
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")

        target_dm = np.eye(4, dtype=complex) / 4  # Maximally mixed

        optimizer = RGDOptimizer(pauli_map=pm, rank=1, target_dm=target_dm)

        assert optimizer.target_dm is not None


class TestRGDOptimizerInitializationMethods:
    """Test different initialization methods."""

    @pytest.mark.parametrize("n_qubits", [2, 3])
    @pytest.mark.parametrize("init_method", ["algorithm", "random", "random_rank"])
    def test_initialization_methods(self, n_qubits, init_method, symbol_map):
        """Test all initialization methods produce valid X_0."""
        # Use full labels for small systems
        labels = generate_all_pauli_labels(n_qubits)
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")

        # Create target
        circuit = ghz_state(n_qubits)
        dm = np.outer(circuit.execute().state(), circuit.execute().state().conj())
        measurements = pm.apply(dm)

        optimizer = RGDOptimizer(pauli_map=pm, rank=1, target_dm=dm, max_iterations=10)

        # Initialize
        coef = optimizer.initialize(measurements, method=init_method)

        # Check X_0 properties
        assert optimizer.Xk is not None
        assert optimizer.Uk is not None
        assert optimizer.Vk is not None

        # Check Hermiticity
        X0 = optimizer.Xk
        diff = X0 - X0.conj().T
        hermiticity_error = np.linalg.norm(diff, "fro")
        assert hermiticity_error < 1e-10, f"X_0 not Hermitian for {init_method}"

        # Check trace (should be positive for normalized state)
        trace = np.trace(X0)
        assert np.isreal(trace), f"Trace not real for {init_method}"
        assert trace > 0, f"Trace not positive for {init_method}"

        # For random_rank, check rank is correct
        if init_method == "random_rank":
            s_vals = np.linalg.svd(X0, compute_uv=False)
            rank_actual = np.sum(s_vals > 1e-10)
            assert (
                rank_actual == optimizer.rank
            ), f"random_rank produced rank {rank_actual}, expected {optimizer.rank}"

    def test_algorithm_initialization_quality(self, symbol_map):
        """Test that algorithm init gives good starting point."""
        n_qubits = 2
        labels = generate_all_pauli_labels(n_qubits)
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")

        circuit = ghz_state(n_qubits)
        dm = np.outer(circuit.execute().state(), circuit.execute().state().conj())
        measurements = pm.apply(dm)

        optimizer = RGDOptimizer(pauli_map=pm, rank=1, target_dm=dm)

        optimizer.initialize(measurements, method="algorithm")

        # Check initial fidelity
        # For full measurements + coef-corrected algorithm init, X_0 ≈ target → fidelity ≈ 1
        trace_X0 = np.real(np.trace(optimizer.Xk))
        X0_norm = optimizer.Xk / trace_X0 if abs(trace_X0) > 1e-12 else optimizer.Xk
        init_fid = np.real(np.trace(X0_norm @ dm))

        assert (
            init_fid > 0.99
        ), f"Algorithm init fidelity too low for full measurements: {init_fid:.6f}"


class TestRGDOptimizerConvergence:
    """Test convergence properties."""

    @pytest.mark.parametrize("n_qubits", [2, 3])
    @pytest.mark.parametrize("use_fast_svd", [True, False])
    def test_convergence_pure_state_full_measurements(
        self, n_qubits, use_fast_svd, symbol_map
    ):
        """Test convergence with full Pauli measurements."""
        labels = generate_all_pauli_labels(n_qubits)
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")

        circuit = ghz_state(n_qubits)
        dm = np.outer(circuit.execute().state(), circuit.execute().state().conj())
        measurements = pm.apply(dm)

        optimizer = RGDOptimizer(
            pauli_map=pm,
            rank=1,
            target_dm=dm,
            tol=1e-4,
            max_iterations=50,
            use_fast_svd=use_fast_svd,
        )

        result = optimizer.run(measurements, init_method="algorithm", verbose=False)

        # Should converge
        assert optimizer.converged, f"Failed to converge (fast_svd={use_fast_svd})"
        assert optimizer.fidelity_history[-1] > 0.99

    @pytest.mark.parametrize("n_qubits", [2, 3])
    def test_convergence_with_rip_sampling(self, n_qubits, symbol_map):
        """Test convergence with RIP-sufficient sampling."""
        rank = 1

        # Use full labels for n < 4
        if n_qubits < 4:
            labels = generate_all_pauli_labels(n_qubits)
        else:
            m_required = compute_rip_samples(n_qubits, rank)
            all_labels = generate_all_pauli_labels(n_qubits)
            m_actual = min(m_required, len(all_labels))

            np.random.seed(42)
            labels = list(np.random.choice(all_labels, m_actual, replace=False))
            if "I" * n_qubits not in labels:
                labels.append("I" * n_qubits)

        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")

        circuit = ghz_state(n_qubits)
        dm = np.outer(circuit.execute().state(), circuit.execute().state().conj())
        measurements = pm.apply(dm)

        optimizer = RGDOptimizer(
            pauli_map=pm, rank=rank, target_dm=dm, tol=1e-3, max_iterations=100
        )

        result = optimizer.run(measurements, init_method="algorithm", verbose=False)

        # For n=2,3 full measurements → should converge to fidelity ≈ 1.0
        assert (
            optimizer.fidelity_history[-1] > 0.99
        ), f"Low fidelity with full Pauli measurements (n={n_qubits}): {optimizer.fidelity_history[-1]:.6f}"

    def test_convergence_with_shot_noise(self, symbol_map):
        """Test convergence with shot-based measurements."""
        n_qubits = 2
        labels = generate_all_pauli_labels(n_qubits)
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")

        circuit = ghz_state(n_qubits)
        dm = np.outer(circuit.execute().state(), circuit.execute().state().conj())

        # Shot-based measurements
        measurements = pm.apply(circuit, nshots=5000)

        optimizer = RGDOptimizer(
            pauli_map=pm,
            rank=1,
            target_dm=dm,
            tol=1e-2,  # Relax tolerance for noisy measurements
            max_iterations=100,
        )

        result = optimizer.run(measurements, init_method="algorithm", verbose=False)

        # Should still get reasonable fidelity
        assert optimizer.fidelity_history[-1] > 0.9


class TestRGDOptimizerNumericalStability:
    """Test numerical stability."""

    def test_hermiticity_preservation(self, symbol_map):
        """Test that Hermiticity is preserved throughout."""
        n_qubits = 2
        labels = generate_all_pauli_labels(n_qubits)
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")

        circuit = ghz_state(n_qubits)
        dm = np.outer(circuit.execute().state(), circuit.execute().state().conj())
        measurements = pm.apply(dm)

        optimizer = RGDOptimizer(pauli_map=pm, rank=1, target_dm=dm, max_iterations=20)

        result = optimizer.run(measurements, init_method="algorithm", verbose=False)

        # Check final X_k is Hermitian
        diff = optimizer.Xk - optimizer.Xk.conj().T
        hermiticity_error = np.linalg.norm(diff, "fro")

        assert hermiticity_error < 1e-8, f"Lost Hermiticity: {hermiticity_error}"

    def test_gradient_descent_monotonic(self, symbol_map):
        """Test that error decreases (mostly) monotonically."""
        n_qubits = 2
        labels = generate_all_pauli_labels(n_qubits)
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")

        circuit = ghz_state(n_qubits)
        dm = np.outer(circuit.execute().state(), circuit.execute().state().conj())
        measurements = pm.apply(dm)

        optimizer = RGDOptimizer(pauli_map=pm, rank=1, target_dm=dm, max_iterations=30)

        result = optimizer.run(measurements, init_method="algorithm", verbose=False)

        errors = optimizer.history

        # Count non-monotonic steps
        non_monotonic = sum(
            1 for i in range(1, len(errors)) if errors[i] > errors[i - 1] * (1 + 1e-10)
        )

        # Full measurements + exact line search → Frobenius error must strictly decrease
        assert non_monotonic == 0, (
            f"Full measurements should be strictly monotonic: "
            f"{non_monotonic}/{len(errors)} non-monotonic steps. "
            f"Errors: {[f'{e:.4e}' for e in errors]}"
        )


class TestRGDOptimizerBackendConsistency:
    """Test consistency across backends."""

    def test_backend_gives_same_result(self, backend_name, symbol_map):
        """Test that different backends give consistent results."""
        try:
            _try_build_backend(backend_name)
        except Exception as e:
            pytest.skip(f"Backend {backend_name} not available: {e}")

        n_qubits = 2
        labels = generate_all_pauli_labels(n_qubits)
        ops = create_pauli_operators(labels, symbol_map)

        circuit = ghz_state(n_qubits)
        dm = np.outer(circuit.execute().state(), circuit.execute().state().conj())

        # Test with current backend
        pm = PauliMap(ops, backend=backend_name)
        measurements = pm.apply(dm)

        optimizer = RGDOptimizer(pauli_map=pm, rank=1, target_dm=dm, max_iterations=30)

        result = optimizer.run(measurements, init_method="algorithm", verbose=False)

        # Get numpy baseline
        pm_numpy = PauliMap(ops, backend="numpy")
        measurements_numpy = pm_numpy.apply(dm)

        optimizer_numpy = RGDOptimizer(
            pauli_map=pm_numpy, rank=1, target_dm=dm, max_iterations=30
        )

        result_numpy = optimizer_numpy.run(
            measurements_numpy, init_method="algorithm", verbose=False
        )

        # Results should be very close
        fid = optimizer.fidelity_history[-1]
        fid_numpy = optimizer_numpy.fidelity_history[-1]

        assert (
            abs(fid - fid_numpy) < 0.01
        ), f"Backend inconsistency: {backend_name}={fid}, numpy={fid_numpy}"


class TestRGDOptimizerFastVsStandard:
    """Test fast SVD vs standard SVD."""

    def test_fast_svd_gives_same_result(self, symbol_map):
        """Test that fast SVD gives same result as standard."""
        n_qubits = 3
        labels = generate_all_pauli_labels(n_qubits)[:30]
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")

        circuit = ghz_state(n_qubits)
        dm = np.outer(circuit.execute().state(), circuit.execute().state().conj())
        measurements = pm.apply(dm)

        # Fast SVD
        opt_fast = RGDOptimizer(
            pauli_map=pm, rank=1, target_dm=dm, max_iterations=20, use_fast_svd=True
        )

        result_fast = opt_fast.run(measurements, init_method="algorithm", verbose=False)

        # Standard SVD
        opt_std = RGDOptimizer(
            pauli_map=pm, rank=1, target_dm=dm, max_iterations=20, use_fast_svd=False
        )

        result_std = opt_std.run(measurements, init_method="algorithm", verbose=False)

        # Results should be very close
        fid_fast = opt_fast.fidelity_history[-1]
        fid_std = opt_std.fidelity_history[-1]

        assert (
            abs(fid_fast - fid_std) < 0.01
        ), f"Fast vs Standard mismatch: fast={fid_fast}, std={fid_std}"


class TestRGDOptimizerEdgeCases:
    """Test edge cases and failure modes."""

    def test_insufficient_measurements(self, symbol_map):
        """Test behavior with very few measurements."""
        n_qubits = 3
        # Use only 5 measurements (way too few!)
        labels = ["III", "XXX", "YYY", "ZZZ", "XYZ"]
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")

        circuit = ghz_state(n_qubits)
        dm = np.outer(circuit.execute().state(), circuit.execute().state().conj())
        measurements = pm.apply(dm)

        optimizer = RGDOptimizer(pauli_map=pm, rank=1, target_dm=dm, max_iterations=50)

        # Should not crash, but may not converge well
        result = optimizer.run(measurements, init_method="random", verbose=False)

        # Just check it doesn't crash
        assert result is not None

    def test_random_initialization_fallback(self, symbol_map):
        """Test that random init works when algorithm fails."""
        n_qubits = 3
        # Deliberately poor sampling
        labels = ["III", "IIX", "IIY", "IIZ"]  # Only measure last qubit
        ops = create_pauli_operators(labels, symbol_map)
        pm = PauliMap(ops, backend="numpy")

        circuit = ghz_state(n_qubits)
        dm = np.outer(circuit.execute().state(), circuit.execute().state().conj())
        measurements = pm.apply(dm)

        # Try algorithm (may fail)
        opt_alg = RGDOptimizer(pauli_map=pm, rank=1, target_dm=dm, max_iterations=20)

        result_alg = opt_alg.run(measurements, init_method="algorithm", verbose=False)
        fid_alg = opt_alg.fidelity_history[-1] if opt_alg.fidelity_history else 0

        # Try random (should be more robust)
        opt_rand = RGDOptimizer(
            pauli_map=pm,
            rank=1,
            target_dm=dm,
            max_iterations=100,  # More iterations for difficult case
            regularization=1e-9,  # Slightly more regularization
        )

        result_rand = opt_rand.run(measurements, init_method="random", verbose=False)
        fid_rand = opt_rand.fidelity_history[-1]

        # With only 4 measurements, even random may struggle
        # Just check it doesn't crash and produces some valid result
        assert (
            0 <= fid_rand <= 1.0
        ), f"Random init produced out-of-range fidelity: {fid_rand:.6f}"

        # Document the behavior
        print(f"\nPoor sampling (4 measurements) results:")
        print(f"  Algorithm init: fidelity = {fid_alg:.4f}")
        print(f"  Random init: fidelity = {fid_rand:.4f}")
        print(f"  (Both expected to be low with insufficient measurements)")


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
