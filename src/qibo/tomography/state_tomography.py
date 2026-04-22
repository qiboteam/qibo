"""
Quantum State Tomography via Riemannian Gradient Descent (RGD) - Version 2

This module implements the Riemannian Gradient Descent algorithm for quantum
state tomography as described in Hsu et al. (2024), with optimizations from
the Supplementary Information.

Key Features:
    1. Fast tangent-space SVD algorithm (SI Section 8.7) - O(d²r) vs O(d³)
    2. Multiple initialization methods (algorithm-based, random, random_rank)
    3. Randomized SVD (rSVD) for large matrices
    4. Improved numerical stability for Hermitian matrices
    5. Comprehensive convergence tracking and diagnostics

Main Classes:
    PauliMap: Implements measurement operator map A and its adjoint A†
    RGDOptimizer: Main optimization algorithm with fast SVD support

Mathematical Background:
    Given Pauli measurements y = A(ρ) + noise, recover low-rank ρ by:

    1. Project gradient to tangent space: P_T(G_k)
    2. Compute optimal step size: α_k
    3. Hard threshold update: X_{k+1} = H_r(X_k + α_k P_T(G_k))

    where H_r denotes rank-r truncation via SVD.

Example Usage:
    >>> from qibo.symbols import I, X, Y, Z
    >>> from qibo.hamiltonians import SymbolicHamiltonian
    >>> from qibo.models.encodings import ghz_state
    >>> import numpy as np
    >>>
    >>> # 1. Create quantum state
    >>> n_qubits = 3
    >>> circuit = ghz_state(n_qubits)
    >>> state = circuit.execute().state()
    >>> rho_true = np.outer(state, state.conj())
    >>>
    >>> # 2. Define Pauli operators
    >>> ops = [SymbolicHamiltonian(Z(0)*Z(1), nqubits=n_qubits),
    ...        SymbolicHamiltonian(X(0)*X(1), nqubits=n_qubits),
    ...        ...]  # Add more operators
    >>>
    >>> # 3. Create PauliMap and get measurements
    >>> pauli_map = PauliMap(ops, backend="numpy")
    >>> measurements = pauli_map.apply(rho_true)
    >>>
    >>> # 4. Reconstruct density matrix
    >>> optimizer = RGDOptimizer(
    ...     pauli_map=pauli_map,
    ...     rank=1,  # Pure state
    ...     target_dm=rho_true,  # For validation
    ...     max_iterations=100,
    ...     use_fast_svd=True  # Use fast algorithm
    ... )
    >>>
    >>> # 5. Run optimization
    >>> rho_reconstructed = optimizer.run(
    ...     measurements,
    ...     init_method='algorithm',
    ...     verbose=True
    ... )
    >>>
    >>> # 6. Get detailed results
    >>> results = optimizer.get_results()
    >>> print(f"Final fidelity: {results['final_fidelity']:.6f}")
    >>> print(f"Converged: {results['converged']}")
    >>> print(f"Iterations: {results['iterations']}")

Performance Notes:
    - Fast SVD is O(d²r + r³) vs O(d³) for standard SVD
    - Most efficient when rank r << dimension d
    - Recommended for systems with n_qubits >= 4 and rank <= d/4
    - For very large systems (n_qubits > 6), use randomized SVD

Reference:
    Hsu, M.-C., et al. (2024). "Quantum state tomography via non-convex
    Riemannian gradient descent." Supplementary Information Section 8.7.

Author:
    Implementation based on paper by Hsu et al. (2024)
    Optimized version with fast SVD and multiple initialization methods
"""

from typing import List, Optional, Tuple, Union

import numpy as np

from qibo.backends import construct_backend
from qibo.config import raise_error
from qibo.hamiltonians import SymbolicHamiltonian


class PauliMap:
    """Pauli measurement operator map and its adjoint.

    Implements the linear map A: H_d(C) → R^m and its adjoint A†: R^m → H_d(C)
    as defined in the paper.
    """

    def __init__(
        self,
        symProj_list: List[SymbolicHamiltonian],
        backend: Optional[Union[str, object]] = None,
    ):
        """Initialize Pauli measurement map.

        Args:
            symProj_list (list[SymbolicHamiltonian]): List of Pauli operators
                as SymbolicHamiltonians.
            backend (str or Backend, optional): Qibo backend name or instance.
                Defaults to ``'numpy'``.

        Raises:
            ValueError: If ``symProj_list`` is empty or operators have inconsistent
                qubit counts.
        """
        if isinstance(backend, str) or backend is None:
            name = backend if backend else "numpy"
            if "-" in name:
                bname, platform = name.split("-", 1)
            else:
                bname, platform = name, None
            self.backend = construct_backend(bname, platform=platform)
        else:
            self.backend = backend

        if not symProj_list:
            raise_error(ValueError, "symProj_list cannot be empty")

        self.symProj_list = symProj_list
        self.nqubits = symProj_list[0].nqubits
        self.dim = 2**self.nqubits
        self._cached_matrices = None

        for i, op in enumerate(symProj_list):
            if op.nqubits != self.nqubits:
                raise_error(
                    ValueError,
                    f"Operator {i} has {op.nqubits} qubits, expected {self.nqubits}",
                )

    def get_matrices(self) -> List[np.ndarray]:
        """Get cached Pauli operator matrices.

        Returns:
            list[ndarray]: Dense matrix representations of Pauli operators.
        """
        if self._cached_matrices is None:
            self._cached_matrices = [proj.matrix for proj in self.symProj_list]
        return self._cached_matrices

    def apply(
        self, state: Union[np.ndarray, object], nshots: Optional[int] = None
    ) -> np.ndarray:
        """Apply measurement operator A(X) = [Tr(P_1 X), ..., Tr(P_m X)].

        Args:
            state (ndarray or Circuit): Density matrix (2-D array) or quantum circuit.
            nshots (int, optional): Number of measurement shots. If ``None``,
                exact expectation values are used.

        Returns:
            ndarray: Measurement vector y of shape (m,) where m is the number
                of operators.
        """
        results = []

        # Density matrix path: compute Tr(P_i ρ) directly from cached matrices.
        # This avoids the deprecated SymbolicHamiltonian.expectation(2D-array) call.
        if isinstance(state, np.ndarray) and state.ndim == 2:
            matrices = self.get_matrices()
            for i, proj in enumerate(self.symProj_list):
                if len(proj.terms) == 0:
                    # Identity operator: Tr(I^⊗n ρ) = Tr(ρ).
                    # Must NOT hardcode 1.0 here — when state is the optimization
                    # iterate X_k, Tr(X_k) ≠ 1 in general. Hardcoding 1.0 would
                    # zero out the identity residual and kill the scale-correction signal.
                    val = float(self.backend.real(self.backend.trace(state)))
                    results.append(val)
                else:
                    val = self.backend.real(self.backend.trace(matrices[i] @ state))
                    results.append(float(val))
            return self.backend.cast(results)

        # State-vector or circuit path
        actual_state = state
        if hasattr(state, "execute") and nshots is None:
            actual_state = state.execute().state()

        for proj in self.symProj_list:
            if len(proj.terms) == 0:
                results.append(1.0)
                continue

            if nshots is not None and hasattr(state, "execute"):
                try:
                    val = proj.expectation_from_circuit(state, nshots=nshots)
                except (TypeError, AttributeError):
                    val = proj.expectation(state.execute().state())
            else:
                val = proj.expectation_from_state(actual_state)

            results.append(float(self.backend.real(val)))

        return self.backend.cast(results)

    def get_adjoint_matrix(self, y: np.ndarray) -> np.ndarray:
        """Compute adjoint operator A†(y) = Σ_i y_i P_i.

        Args:
            y (ndarray): Measurement vector of shape (m,).

        Returns:
            ndarray: Hermitian matrix A†(y) of shape (d, d) where d = 2^n_qubits.
        """
        matrices = self.get_matrices()
        return sum(y_i * mat for y_i, mat in zip(y, matrices))

    def rSVD_hermitian(
        self, A: np.ndarray, rank: int, oversampling: int = 5, power_iterations: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Randomized SVD for Hermitian matrices (Algorithm 1 from SI).

        This is more efficient than full SVD for large matrices.

        Args:
            A (ndarray): Hermitian matrix to decompose.
            rank (int): Target rank.
            oversampling (int): Oversampling parameter. Defaults to 5.
            power_iterations (int): Number of power iterations. Defaults to 2.

        Returns:
            tuple[ndarray, ndarray, ndarray]: ``(U, S, V)`` where ``U = V``
                for the Hermitian case.
        """
        d = A.shape[0]
        r_oversample = min(rank + oversampling, d)

        # Random test matrix
        Omega = np.random.randn(d, r_oversample) + 1j * np.random.randn(d, r_oversample)

        # Range finding with power iteration
        Y = A @ Omega
        Q, _ = np.linalg.qr(Y)

        for _ in range(power_iterations):
            Y = A @ Q
            Q, _ = np.linalg.qr(Y)

        # Project to smaller subspace
        B = Q.conj().T @ A @ Q

        # SVD of smaller matrix
        U_B, s_vals, Vh_B = np.linalg.svd(B, hermitian=True)

        # Recover full factorization
        U = Q @ U_B[:, :rank]
        S = np.diag(s_vals[:rank])
        V = U  # For Hermitian: V = U

        return U, S, V


class RGDOptimizer:
    """Riemannian Gradient Descent optimizer for quantum state tomography.

    Implements Algorithm 1 with optimizations from SI Section 8.7.
    """

    def __init__(
        self,
        pauli_map: PauliMap,
        rank: int = 1,
        target_dm: Optional[np.ndarray] = None,
        tol: float = 1e-5,
        max_iterations: int = 150,
        regularization: float = 1e-10,
        use_fast_svd: bool = True,
    ):
        """Initialize RGD optimizer.

        Args:
            pauli_map (PauliMap): PauliMap instance containing measurement operators.
            rank (int): Rank of the low-rank approximation (r in paper). Defaults to 1.
            target_dm (ndarray, optional): Target density matrix for computing fidelity.
            tol (float): Convergence tolerance on relative Frobenius change.
                Defaults to 1e-5.
            max_iterations (int): Maximum number of RGD iterations. Defaults to 150.
            regularization (float): Regularization parameter for step size computation.
                Defaults to 1e-10.
            use_fast_svd (bool): Use fast tangent-space SVD (SI Section 8.7).
                Defaults to ``True``.
        """
        self.pauli_map = pauli_map
        self.backend = pauli_map.backend
        self.rank = rank
        self.tol = tol
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.use_fast_svd = use_fast_svd  # NEW

        self.target_dm = self.backend.cast(target_dm) if target_dm is not None else None

        # State variables
        self.Xk = None
        self.Uk = None  # Left singular vectors
        self.Vk = None  # Right singular vectors (= Uk for Hermitian)
        self.Sigma_k = None  # Singular values
        self.coef = None

        # Tracking
        self.history = []
        self.fidelity_history = []
        self.gradient_norm_history = []
        self.step_size_history = []
        self.iteration = 0
        self.converged = False

    def initialize(
        self,
        measurements: np.ndarray,
        method: str = "algorithm",  # NEW: 'algorithm', 'random', 'random_rank'
        use_rsvd: bool = True,
        svd_threshold: int = 32,
    ) -> float:
        """Initialize X_0.

        Args:
            measurements (ndarray): Measurement vector y.
            method (str): Initialization method. ``'algorithm'`` uses the paper's
                X_0 = H_r(A†(y)); ``'random'`` uses a random full-rank Hermitian
                matrix; ``'random_rank'`` uses a random rank-r Hermitian matrix.
                Defaults to ``'algorithm'``.
            use_rsvd (bool): Use randomized SVD for large matrices. Defaults to ``True``.
            svd_threshold (int): Use rSVD only when dim > threshold. Defaults to 32.

        Returns:
            float: Scaling coefficient sqrt(d/m).
        """
        m = len(measurements)
        dim = self.pauli_map.dim

        # Scaling coefficient
        self.coef = np.sqrt(dim / m)
        y_scaled = self.backend.cast(measurements) * self.coef

        if method == "algorithm":
            # Paper's initialization: X_0 = H_r(A†(y))
            # A†(y) = coef * sum_i y_i * S_i  (paper Eq.; the coef factor appears twice:
            # once in y_scaled = coef * measurements, and once as A†'s own coef)
            adj_matrix = self.pauli_map.get_adjoint_matrix(y_scaled) * self.coef
            adj_matrix = (adj_matrix + adj_matrix.conj().T) / 2  # Ensure Hermitian

            should_use_rsvd = use_rsvd and (dim > svd_threshold)

            if should_use_rsvd:
                U, S, V = self.pauli_map.rSVD_hermitian(adj_matrix, self.rank)
            else:
                adj_np = self.backend.to_numpy(adj_matrix)
                U, s_vals, Vh = np.linalg.svd(adj_np, hermitian=True)
                U = U[:, : self.rank]
                s_vals = s_vals[: self.rank]
                S = np.diag(s_vals)
                V = U  # Hermitian

            self.Uk = self.backend.cast(U)
            self.Vk = self.backend.cast(V)
            self.Sigma_k = self.backend.cast(S)

            # Construct and ensure Hermiticity (do in numpy first)
            temp_Xk = U @ S @ V.conj().T
            temp_Xk = (temp_Xk + temp_Xk.conj().T) / 2
            # No normalization: paper Algorithm 1 says "No further normalization is needed"

            self.Xk = self.backend.cast(temp_Xk)

        elif method == "random":
            # Random Hermitian initialization (full rank then truncate)
            np.random.seed()
            A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
            A = (A + A.conj().T) / 2  # Make Hermitian
            A = A / np.linalg.norm(A, "fro")  # Normalize

            # Truncate to rank r
            U, s_vals, Vh = np.linalg.svd(A, hermitian=True)
            U = U[:, : self.rank]
            s_vals = s_vals[: self.rank]
            s_vals = s_vals / np.sum(s_vals)  # Normalize
            S = np.diag(s_vals)
            V = U  # Hermitian

            self.Uk = self.backend.cast(U)
            self.Vk = self.backend.cast(V)
            self.Sigma_k = self.backend.cast(S)

            # Construct and ensure Hermiticity
            temp_Xk = U @ S @ V.conj().T
            temp_Xk = (temp_Xk + temp_Xk.conj().T) / 2
            self.Xk = self.backend.cast(temp_Xk)

        elif method == "random_rank":
            # Random rank-r Hermitian matrix directly
            # Key fix: Ensure proper normalization and positive semidefiniteness
            np.random.seed()

            # Generate random orthonormal basis
            U = np.random.randn(dim, self.rank) + 1j * np.random.randn(dim, self.rank)
            U, _ = np.linalg.qr(U)  # Orthonormalize

            # Generate reasonable singular values (decreasing, positive)
            # Use exponential decay to ensure well-conditioned matrix
            s_vals = np.exp(-np.arange(self.rank))  # e^0, e^-1, e^-2, ...
            s_vals = s_vals / np.sum(s_vals)  # Normalize so trace = 1

            S = np.diag(s_vals)
            V = U  # Hermitian

            self.Uk = self.backend.cast(U)
            self.Vk = self.backend.cast(V)
            self.Sigma_k = self.backend.cast(S)

            # Construct and ensure Hermiticity
            temp_Xk = U @ S @ V.conj().T
            temp_Xk = (temp_Xk + temp_Xk.conj().T) / 2

            # Final normalization to ensure Tr(X_0) = 1
            trace = np.trace(temp_Xk)
            if abs(trace) > 1e-10:
                temp_Xk = temp_Xk / trace

            self.Xk = self.backend.cast(temp_Xk)
        else:
            raise_error(ValueError, f"Unknown initialization method: {method}")

        # Ensure Hermiticity
        self.Xk = (self.Xk + self.backend.conj(self.Xk.T)) / 2

        return self.coef

    def _fast_tangent_svd(
        self,
        Jk: np.ndarray,
        Uk: np.ndarray,
        Vk: np.ndarray,
        Sigma_k: np.ndarray,
        alpha_k: float,
        Gk: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fast SVD using tangent space structure (SI Section 8.7).

        This avoids O(d³) SVD by exploiting the structure:
        J_k = X_k + α_k P_T(G_k)
            = U_k Ξ V_k† + U_k Y_1† + Y_2 V_k†
            = [U_k Q_2] M_k [V_k† Q_1†]ᵀ

        where M_k is only 2r × 2r, so SVD is much faster!

        Args:
            Jk (ndarray): Not used directly; kept for interface compatibility.
            Uk (ndarray): Left singular vectors of X_k, shape (d, r).
            Vk (ndarray): Right singular vectors of X_k, shape (d, r).
            Sigma_k (ndarray): Diagonal singular value matrix, shape (r, r).
            alpha_k (float): Step size scalar.
            Gk (ndarray): Gradient matrix, shape (d, d), Hermitian.

        Returns:
            tuple[ndarray, ndarray, ndarray]: Updated ``(U_new, Sigma_new, V_new)``
                with shapes (d, r), (r, r), (d, r) respectively.

        Complexity:
            O(d²r + r³) vs O(d³) for full SVD
        """
        r = self.rank
        Fk = alpha_k * Gk

        # Compute components (SI Eq. in Section 8.7)
        # Ξ = Σ_k + U_k† F_k V_k
        Xi = Sigma_k + Uk.conj().T @ Fk @ Vk

        # Y_1 = (I - V_k V_k†) F_k U_k
        VVh = Vk @ Vk.conj().T
        Y1 = (np.eye(Vk.shape[0]) - VVh) @ Fk @ Uk

        # Y_2 = (I - U_k U_k†) F_k V_k
        UUh = Uk @ Uk.conj().T
        Y2 = (np.eye(Uk.shape[0]) - UUh) @ Fk @ Vk

        # QR factorizations
        Q1, R1 = np.linalg.qr(Y1)
        Q2, R2 = np.linalg.qr(Y2)

        # Build small matrix M_k (2r × 2r)
        M_top = np.hstack([Xi, R1.conj().T])
        M_bottom = np.hstack([R2, np.zeros((r, r))])
        M_k = np.vstack([M_top, M_bottom])

        # SVD of small matrix (O(r³) instead of O(d³)!)
        U_M, s_M, Vh_M = np.linalg.svd(M_k, full_matrices=False)

        # Recover full factorization
        basis_left = np.hstack([Uk, Q2])
        basis_right = np.hstack([Vk, Q1])

        U_new = basis_left @ U_M[:, :r]
        s_new = s_M[:r]
        V_new = basis_right @ Vh_M[:r, :].conj().T

        return U_new, np.diag(s_new), V_new

    def step(self, coef: float, y_scaled: np.ndarray) -> dict:
        """Perform one RGD iteration (Algorithm 1, main loop).

        Implements one iteration of:
        1. Compute gradient: G_k = A†(y - A(X_k))
        2. Project to tangent space: P_T(G_k)
        3. Compute step size: α_k
        4. Update via hard thresholding: X_{k+1} = H_r(X_k + α_k P_T(G_k))

        Args:
            coef (float): Scaling coefficient sqrt(d/m).
            y_scaled (ndarray): Scaled measurements (y * coef).

        Returns:
            dict: Iteration statistics with keys ``iteration``, ``step_size``,
                ``gradient_norm``, ``numerator``, ``denominator``.
        """
        # Step 1: Compute gradient G_k = A†(y - A(X_k))
        A_Xk = self.pauli_map.apply(self.Xk) * coef
        residual = y_scaled - A_Xk
        Gk = self.pauli_map.get_adjoint_matrix(residual) * coef
        Gk = (Gk + Gk.conj().T) / 2  # Ensure Hermitian

        # Step 2: Project onto tangent space
        # For Hermitian: U = V, so P_T(G) = UU† G + G UU† - UU† G UU†
        U = self.Uk
        UUh = U @ U.conj().T
        PTk_Gk = UUh @ Gk + Gk @ UUh - UUh @ Gk @ UUh
        PTk_Gk = (PTk_Gk + PTk_Gk.conj().T) / 2

        # Step 3: Compute step size
        numerator = np.linalg.norm(PTk_Gk, "fro") ** 2
        A_PTk_Gk = self.pauli_map.apply(PTk_Gk) * coef
        denominator = np.linalg.norm(A_PTk_Gk) ** 2

        denominator_reg = denominator + self.regularization
        alpha_k = float(numerator / denominator_reg)
        alpha_k = min(alpha_k, 2.0)  # Clip for stability

        gradient_norm = float(np.linalg.norm(Gk, "fro"))

        # Step 4: Update with hard thresholding
        if self.use_fast_svd and self.rank < self.pauli_map.dim // 4:
            # Use fast tangent-space SVD
            U_new, Sigma_new, V_new = self._fast_tangent_svd(
                None,  # Don't need to form J_k explicitly!
                self.Uk,
                self.Vk,
                self.Sigma_k,
                alpha_k,
                PTk_Gk,
            )
        else:
            # Standard approach
            Jk = self.Xk + alpha_k * PTk_Gk
            Jk = (Jk + Jk.conj().T) / 2

            U_new, s_new, Vh_new = np.linalg.svd(Jk, hermitian=True)
            U_new = U_new[:, : self.rank]
            s_new = s_new[: self.rank]
            Sigma_new = np.diag(s_new)
            V_new = U_new  # Hermitian

        # Update state
        self.Uk = self.backend.cast(U_new)
        self.Vk = self.backend.cast(V_new)
        self.Sigma_k = self.backend.cast(Sigma_new)

        # Compute X_k in numpy first, then cast (safer for Hermiticity enforcement)
        temp_Xk = U_new @ Sigma_new @ V_new.conj().T
        temp_Xk = (temp_Xk + temp_Xk.conj().T) / 2  # Enforce Hermiticity in numpy
        self.Xk = self.backend.cast(temp_Xk)  # Final cast to backend

        self.iteration += 1
        self.gradient_norm_history.append(gradient_norm)
        self.step_size_history.append(alpha_k)

        return {
            "iteration": self.iteration,
            "step_size": alpha_k,
            "gradient_norm": gradient_norm,
            "numerator": float(numerator),
            "denominator": float(denominator),
        }

    def _compute_metrics(self) -> Tuple[float, float]:
        """Compute error metrics against target density matrix.

        Returns:
            tuple[float, float]: ``(frobenius_error, fidelity)`` where
                ``frobenius_error`` is ``||X_k - ρ_target||_F`` and ``fidelity``
                is ``Tr((X_k / Tr(X_k)) ρ_target)``. Returns ``(0.0, 0.0)``
                if ``target_dm`` is ``None``.
        """
        if self.target_dm is None:
            return 0.0, 0.0

        diff = self.Xk - self.target_dm
        fro_error = float(self.backend.matrix_norm(diff, "fro"))

        # Normalize X_k before computing fidelity so the metric is always in [0,1].
        # For full measurements X_k already has Tr=1 (no-op); for partial measurements
        # the trace may deviate, so we normalize to get the correct state fidelity.
        trace_Xk = float(self.backend.real(self.backend.trace(self.Xk)))
        if abs(trace_Xk) > 1e-12:
            Xk_normalized = self.Xk / trace_Xk
        else:
            Xk_normalized = self.Xk
        fidelity = float(
            self.backend.real(self.backend.trace(Xk_normalized @ self.target_dm))
        )

        return fro_error, fidelity

    def run(
        self,
        measurements: np.ndarray,
        init_method: str = "algorithm",
        verbose: bool = True,
    ) -> np.ndarray:
        """Run complete RGD algorithm.

        Args:
            measurements (ndarray): Measurement vector y.
            init_method (str): Initialization method — ``'algorithm'``, ``'random'``,
                or ``'random_rank'``. Defaults to ``'algorithm'``.
            verbose (bool): Print progress to stdout. Defaults to ``True``.

        Returns:
            ndarray: Reconstructed density matrix X_k.
        """
        # Initialize
        coef = self.initialize(measurements, method=init_method)
        y_scaled = self.backend.cast(measurements) * coef

        if verbose:
            print(f"\n{'='*70}")
            print(f"Riemannian Gradient Descent for Quantum State Tomography")
            print(f"{'='*70}")
            print(f"System: {self.pauli_map.nqubits} qubits, rank {self.rank}")
            print(f"Measurements: {len(measurements)}")
            print(f"Initialization: {init_method}")
            print(f"Fast SVD: {self.use_fast_svd}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Tolerance: {self.tol}")
            print(f"Regularization: {self.regularization}")
            print(f"{'='*70}\n")

            if self.target_dm is not None:
                print(
                    f"{'Iter':>6} | {'Fro Error':>12} | {'Fidelity':>10} | "
                    f"{'Step Size':>12} | {'Grad Norm':>12}"
                )
                print(f"{'-'*70}")

        # Log initial quality metrics (for benchmarking only, not convergence)
        init_fro_error, init_fidelity = self._compute_metrics()
        if verbose and self.target_dm is not None:
            print(
                f"{'INIT':>6} | {init_fro_error:>12.6e} | {init_fidelity:>10.6f} | "
                f"{'---':>12} | {'---':>12}"
            )

        # Main loop
        for k in range(self.max_iterations):
            # Save current iterate before step (for convergence criterion)
            Xk_prev = np.array(self.Xk)

            stats = self.step(coef, y_scaled)
            fro_error, fidelity = self._compute_metrics()

            self.history.append(fro_error)
            self.fidelity_history.append(fidelity)

            if verbose and (k % 1 == 0) and self.target_dm is not None:
                print(
                    f"{stats['iteration']:6d} | {fro_error:12.6e} | "
                    f"{fidelity:10.6f} | {stats['step_size']:12.6e} | "
                    f"{stats['gradient_norm']:12.6e}"
                )

            # Convergence criterion: relative change between consecutive iterates
            # ||X_{k+1} - X_k||_F / ||X_k||_F < tol
            # (target_dm is unknown in real tomography — cannot use fro_error for convergence)
            Xk_np = np.array(self.Xk)
            norm_diff = np.linalg.norm(Xk_np - Xk_prev, "fro")
            norm_prev = np.linalg.norm(Xk_prev, "fro")
            rel_change = norm_diff / norm_prev if norm_prev > 1e-12 else norm_diff
            if rel_change < self.tol:
                self.converged = True
                if verbose:
                    print(f"\n{'='*70}")
                    print(f"Converged at iteration {self.iteration}!")
                    print(f"Relative change: {rel_change:.6e}")
                    if self.target_dm is not None:
                        print(f"Final Frobenius error: {fro_error:.6e}")
                        print(f"Final fidelity: {fidelity:.6f}")
                    print(f"{'='*70}\n")
                break

            # Check for stagnation
            if stats["gradient_norm"] < 1e-12:
                if verbose:
                    print(
                        f"\nWarning: Gradient norm too small ({stats['gradient_norm']:.2e})"
                    )
                    print(f"Algorithm stagnated at iteration {self.iteration}")
                break

        if not self.converged and verbose:
            print(f"\nReached maximum iterations ({self.max_iterations})")
            if self.target_dm is not None:
                print(f"Final Frobenius error: {self.history[-1]:.6e}")
                print(f"Final fidelity: {self.fidelity_history[-1]:.6f}\n")

        return self.Xk

    def get_results(self) -> dict:
        """Get comprehensive optimization results and statistics.

        Returns:
            dict: Keys are ``density_matrix`` (final X_k), ``converged`` (bool),
                ``iterations`` (int), ``frobenius_errors`` (ndarray),
                ``fidelities`` (ndarray), ``gradient_norms`` (ndarray),
                ``step_sizes`` (ndarray), ``final_error`` (float or ``None``),
                ``final_fidelity`` (float or ``None``).
        """
        return {
            "density_matrix": self.Xk,
            "converged": self.converged,
            "iterations": self.iteration,
            "frobenius_errors": np.array(self.history),
            "fidelities": np.array(self.fidelity_history),
            "gradient_norms": np.array(self.gradient_norm_history),
            "step_sizes": np.array(self.step_size_history),
            "final_error": self.history[-1] if self.history else None,
            "final_fidelity": (
                self.fidelity_history[-1] if self.fidelity_history else None
            ),
        }
