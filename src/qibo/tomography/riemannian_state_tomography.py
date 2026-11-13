"""Low-rank state tomography with Riemannian gradient descent."""

import math

from numpy.typing import ArrayLike

from qibo import Circuit
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.quantum_info._superoperator_transformations import (
    _operator_to_pauli_vectors_fht,
    _pauli_vectors_to_operator_fht,
)
from qibo.tomography.abstract import Tomography


class RiemannianStateTomography(Tomography):
    """Low-rank state tomography with non-convex Riemannian gradient descent.

    The density matrix :math:`\\rho` of :math:`n` qubits, with :math:`d = 2^{n}`, is assumed
    to have rank at most :math:`r`, which is the case of the states of interest in most
    applications. It is then determined by far fewer than :math:`d^{2}` parameters, and
    :math:`m = \\mathcal{O}(r \\, d)` Pauli expectation values, up to logarithmic factors,
    are enough to recover it [1, 2]. The protocol draws :math:`m` distinct Pauli strings
    :math:`S_{i}` uniformly at random, and estimates each expectation value
    :math:`\\text{tr}(S_{i} \\, \\rho)` from ``nshots`` shots, which gives the data
    :math:`y_{i}` of the linear sensing map

    .. math::
        \\mathcal{A}(X)_{i} = \\sqrt{\\frac{d}{m}} \\, \\text{tr}(S_{i} \\, X) \\, , \\qquad
        \\mathcal{A}^{\\dagger}(y) = \\sqrt{\\frac{d}{m}} \\sum_{i=1}^{m} y_{i} \\, S_{i} \\, .

    The estimate seeks the Hermitian matrix of rank at most :math:`r` that minimizes
    :math:`\\|y - \\mathcal{A}(X)\\|_{2}^{2} / 2`, with the Riemannian gradient descent
    of Ref. [3], as applied to tomography in Ref. [4]. The first iterate is the best
    rank-:math:`r` approximation of :math:`\\mathcal{A}^{\\dagger}(y)`. At each iteration,
    the gradient :math:`G = \\mathcal{A}^{\\dagger}(y - \\mathcal{A}(X))` is projected onto
    the tangent space :math:`T` of the rank-:math:`r` matrices at :math:`X`, and
    :math:`X` is moved along the projection :math:`\\mathcal{P}_{T}(G)` with the step size
    that minimizes the objective exactly,

    .. math::
        \\alpha = \\frac{\\|\\mathcal{P}_{T}(G)\\|_{F}^{2}}
        {\\|\\mathcal{A}(\\mathcal{P}_{T}(G))\\|_{2}^{2}} \\, ,

    followed by the truncation to rank :math:`r`. The contraction factor of the error is a
    constant that does not depend on the condition number of :math:`\\rho`, so the number
    of iterations grows only logarithmically with the target error [4], while the
    first-order methods based on a factorization of the state converge more slowly for
    states with a large condition number. The update involves :math:`d \\times r` and
    :math:`2r \\times 2r` matrices, apart from the sensing maps and the first iterate.

    The truncation keeps the :math:`r` largest eigenvalues and sets the negative ones to
    zero, *i.e.* it projects onto the positive semidefinite matrices of rank at most
    :math:`r`, which contain :math:`\\rho`. The truncation of Ref. [4] keeps instead the
    singular values of largest magnitude, so that a negative eigenvalue can survive, and in
    our tests it trapped the iterations in spurious minima for states of rank larger than
    one. The unit trace is relaxed, as in Ref. [4], so the estimate is positive
    semidefinite with rank at most :math:`r`, but its trace is not exactly one when the
    expectation values have statistical errors.

    The Pauli strings must determine the state. Under the conditions of Ref. [4], which
    require more strings than the number of parameters of the state, and enough shots, the
    error in the Frobenius norm is bounded by a constant times :math:`\\sqrt{r}` times the
    noise level. In the numerical experiments of Ref. [4], between 20% and 40% of the
    :math:`4^{n}` strings were measured. With fewer strings, other low-rank states may fit
    the data as well as the true one, particularly if only few Pauli expectation values of
    the state are not zero, as in GHZ states, and the estimate can be any of them. Each
    Pauli string is measured with its own circuit, so ``nobservables`` circuits are
    executed.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.tomography import RiemannianStateTomography

            circuit = Circuit(4)
            circuit.add(gates.RY(qubit, 0.3 * (qubit + 1)) for qubit in range(4))
            circuit.add(gates.CNOT(qubit, qubit + 1) for qubit in range(3))

            # pure state, which has rank one, from 128 of the 256 Pauli strings
            tomography = RiemannianStateTomography()
            estimate = tomography(
                circuit, rank=1, nobservables=128, nshots=2000, seed=1
            )

    References:
        1. D. Gross, Y.-K. Liu, S. T. Flammia, S. Becker and J. Eisert,
           *Quantum state tomography via compressed sensing*,
           `Phys. Rev. Lett. 105, 150401 (2010) <https://doi.org/10.1103/PhysRevLett.105.150401>`_.
        2. S. T. Flammia, D. Gross, Y.-K. Liu and J. Eisert,
           *Quantum tomography via compressed sensing: error bounds, sample complexity and
           efficient estimators*,
           `New J. Phys. 14, 095022 (2012) <https://doi.org/10.1088/1367-2630/14/9/095022>`_.
        3. K. Wei, J.-F. Cai, T. F. Chan and S. Leung,
           *Guarantees of Riemannian optimization for low rank matrix recovery*,
           `SIAM J. Matrix Anal. Appl. 37, 1198 (2016) <https://doi.org/10.1137/15M1050525>`_.
        4. M.-C. Hsu, E.-J. Kuo, W.-H. Yu, J.-F. Cai and M.-H. Hsieh,
           *Quantum state tomography via non-convex Riemannian gradient descent*,
           `Phys. Rev. Lett. 132, 240804 (2024) <https://doi.org/10.1103/PhysRevLett.132.240804>`_.
    """

    def __call__(
        self,
        circuit: Circuit,
        rank: int,
        nobservables: int,
        nshots: int = 1000,
        iterations: int = 100,
        tolerance: float = 1e-8,
        seed: int | None = None,
        backend: Backend = None,
    ) -> ArrayLike:
        """Estimates the low-rank density matrix of the state prepared by ``circuit``.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit preparing the
                state, without measurements.
            rank (int): rank :math:`r` of the estimate, which must be at least the rank
                of the state.
            nobservables (int): number :math:`m` of Pauli strings that are measured,
                between :math:`1` and :math:`4^{n}`. It is also the number of circuits.
            nshots (int, optional): number of shots per Pauli string.
                Defaults to :math:`1000`.
            iterations (int, optional): maximum number of Riemannian gradient descent
                iterations, after the initialization. Defaults to :math:`100`.
            tolerance (float, optional): the iterations stop when the largest change of
                an element of the estimate is smaller than this value.
                Defaults to :math:`10^{-8}`.
            seed (int, optional): seed of the sampling of the Pauli strings and of their
                outcomes. If ``None``, the generator of the backend is not reseeded for the
                outcomes. Defaults to ``None``.
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend
                to be used in the execution. If ``None``, it uses the current
                backend. Defaults to ``None``.

        Returns:
            ArrayLike: :math:`2^{n} \\times 2^{n}` positive semidefinite estimate of the density
            matrix, with rank at most ``rank``. Its trace is not constrained.
        """
        nqubits = circuit.nqubits
        dim = 2**nqubits

        if isinstance(nshots, bool) or not isinstance(nshots, int) or nshots < 1:
            raise_error(
                ValueError,
                f"``nshots`` must be a positive int, but it is {nshots}.",
            )

        if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= dim:
            raise_error(
                ValueError,
                f"``rank`` must be an int between 1 and {dim}, but it is {rank}.",
            )

        if (
            isinstance(nobservables, bool)
            or not isinstance(nobservables, int)
            or not 1 <= nobservables <= dim**2
        ):
            raise_error(
                ValueError,
                f"``nobservables`` must be an int between 1 and {dim**2}, "
                + f"but it is {nobservables}.",
            )

        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or iterations < 1
        ):
            raise_error(
                ValueError,
                f"``iterations`` must be a positive int, but it is {iterations}.",
            )

        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, int | float)
            or tolerance < 0
        ):
            raise_error(
                ValueError,
                f"``tolerance`` must be a non-negative number, but it is {tolerance}.",
            )

        backend = _check_backend(backend)

        # distinct Pauli strings, as positions in the Pauli basis of base-4 digits, with
        # I, X, Y, Z as 0, 1, 2, 3 and the first qubit as the most significant digit
        indices = backend.sort(
            backend.random_choice(
                dim**2,
                size=nobservables,
                replace=False,
                seed=seed,
                dtype=backend.int64,
            )
        )
        weights = backend.cast(
            [4 ** (nqubits - 1 - qubit) for qubit in range(nqubits)],
            dtype=backend.int64,
        )
        digits = (indices[:, None] // weights[None, :]) % 4
        observables = [
            "".join("IXYZ"[digit] for digit in row)
            for row in backend.to_numpy(digits).tolist()
        ]

        circuits = self.circuits(circuit, observables)

        if seed is not None:
            backend.set_seed(seed)

        results = self.execute(circuits, nshots, backend)

        frequencies = backend.zeros((nobservables, dim), dtype=backend.float64)
        for index, result in enumerate(results):
            for outcome, frequency in result.frequencies(binary=False).items():
                frequencies[index, outcome] = frequency / nshots

        # the eigenvalue of a Pauli string is -1 for each qubit of its support whose
        # outcome is 1, so the expectation values are the averages of these parities
        outcomes = backend.arange(dim)
        signs = backend.ones((nobservables, dim), dtype=backend.float64)
        for qubit in range(nqubits):
            bits = backend.cast(
                (outcomes // 2 ** (nqubits - 1 - qubit)) % 2, dtype=backend.float64
            )
            supports = backend.cast(digits[:, qubit] > 0, dtype=backend.float64)
            signs = signs * (1.0 - 2.0 * supports[:, None] * bits[None, :])

        scale = math.sqrt(dim / nobservables)
        data = scale * backend.sum(frequencies * signs, axis=1)

        # the state and its eigenvectors are set by the initialization, which is the
        # first iteration, and used by the following ones
        state, vectors = None, None
        residuals = data
        for iteration in range(iterations + 1):
            # the adjoint of the sensing map, which is the gradient of the objective
            # up to its sign
            coefficients = backend.zeros(dim**2, dtype=backend.float64)
            coefficients[indices] = scale * residuals
            gradient = _pauli_vectors_to_operator_fht(
                coefficients, nqubits, dim, backend=backend
            )

            if iteration == 0:
                # initialization, with the truncation of the adjoint of the data
                basis = None
                matrix = gradient
            else:
                # projection of the gradient onto the tangent space at the state
                directions = gradient @ vectors
                tangent = (
                    vectors @ backend.dagger(directions)
                    + directions @ backend.dagger(vectors)
                    - vectors
                    @ (backend.dagger(vectors) @ directions)
                    @ backend.dagger(vectors)
                )
                sensed = (
                    scale
                    * backend.real(
                        _operator_to_pauli_vectors_fht(
                            tangent, nqubits, dim, backend=backend
                        )
                    )[indices]
                )

                # the exact line search stops when there is no more descent direction
                denominator = backend.sum(sensed**2)
                if float(denominator) == 0.0:
                    break
                step = backend.sum(backend.abs(tangent) ** 2) / denominator

                # the matrix on the tangent space has its range in the span of
                # the eigenvectors and the gradient applied to them
                basis, _ = backend.qr(
                    backend.concatenate((vectors, directions), axis=1)
                )
                matrix = backend.dagger(basis) @ (state + step * tangent) @ basis

            # truncation to the largest eigenvalues, with the negative ones set to zero
            eigenvalues, eigenvectors = backend.eigh(matrix)
            kept = backend.argsort(eigenvalues)[-rank:]
            vectors = eigenvectors[:, kept]
            if basis is not None:
                vectors = basis @ vectors
            new_state = (
                vectors * backend.maximum(eigenvalues[kept], 0.0)
            ) @ backend.dagger(vectors)

            converged = (
                iteration > 0
                and backend.max(backend.abs(new_state - state)) < tolerance
            )
            state = new_state
            if converged:
                break

            residuals = (
                data
                - scale
                * backend.real(
                    _operator_to_pauli_vectors_fht(state, nqubits, dim, backend=backend)
                )[indices]
            )

        return state

    def circuits(self, circuit: Circuit, observables: list[str]) -> list[Circuit]:
        """Appends the basis changes and the measurements of each Pauli string to ``circuit``.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit preparing the
                state, without measurements.
            observables (list): Pauli strings to be measured, each one a ``str`` of
                ``"I"``, ``"X"``, ``"Y"`` and ``"Z"``, one per qubit, *e.g.* ``"XIZ"``.

        Returns:
            list: one circuit for each Pauli string, in the same order. Each circuit
            measures all qubits, after rotating the Pauli :math:`X` with
            :class:`qibo.gates.H`, and the Pauli :math:`Y` with :class:`qibo.gates.SDG`
            followed by :class:`qibo.gates.H`. The identity and the Pauli :math:`Z` need
            no rotation. The outcomes of the qubits of the identity are not used.
        """
        return self._pauli_circuits(circuit, observables)
