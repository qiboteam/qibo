"""Quantum state tomography with Pauli measurements."""

import math
from functools import reduce
from itertools import product

from numpy.typing import ArrayLike

from qibo import Circuit, gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.quantum_info._superoperator_transformations import (
    _pauli_vectors_to_operator_fht,
)
from qibo.quantum_info.utils import hadamard_transform
from qibo.tomography.abstract import Tomography

_METHODS = (
    "linear-inversion",
    "physical-linear-inversion",
    "least-squares",
    "mle",
)


class StateTomography(Tomography):
    """Quantum state tomography with Pauli measurements.

    The state :math:`\\rho` of :math:`n` qubits is expanded in the Pauli basis,

    .. math::
        \\rho = \\frac{1}{2^{n}} \\sum_{P} \\text{tr}(P \\, \\rho) \\, P \\, ,

    where the sum runs over the :math:`4^{n}` Pauli strings :math:`P`. Each qubit is measured
    in one of the bases :math:`X`, :math:`Y`, or :math:`Z`, which gives :math:`3^{n}`
    measurement settings. The expectation value of a Pauli string with :math:`w` non-identity
    factors is estimated from the outcomes of any setting that measures those factors in the
    right bases, so there are :math:`3^{n - w}` such settings, whose estimates are averaged.
    The number of circuits, :math:`3^{n}`, and the cost of the reconstruction grow
    exponentially with :math:`n`, so this protocol is meant for few qubits.

    The estimate of the state, :math:`\\hat{\\rho}`, is obtained from the outcomes with one of
    the following estimators, selected with the argument ``method`` of :meth:`__call__`.

    * ``"linear-inversion"``: the estimated expectation values are inserted in the expansion
      above [1, 2]. The estimate is Hermitian, has unit trace, and is unbiased, but it is
      **not** positive semidefinite in general, since the statistical errors can make some of
      its eigenvalues negative.
    * ``"physical-linear-inversion"``: the density matrix that is the closest, in the Frobenius
      norm, to the linear inversion estimate. It is obtained by projecting the eigenvalues of
      the latter onto the probability simplex, which sets the negative ones to zero and shifts
      the others to keep the trace [3].
    * ``"least-squares"``: the density matrix that minimizes the sum of the squared
      differences between the observed frequencies and the probabilities that it predicts for
      all outcomes of all settings. The minimization is constrained to physical states, and it
      is carried out by an accelerated projected gradient method [3, 6].
    * ``"mle"``: the density matrix that maximizes the likelihood of the observed outcomes,
      found by the diluted iterative algorithm of Refs. [4, 5]. It is always physical, but it
      is biased towards states with low rank.

    Readout errors can be mitigated, in all estimators, with the response matrix of the
    measurements [7]. In the linear estimators, the probabilities of the outcomes are
    multiplied by the inverse of the response matrix. In ``"least-squares"`` and ``"mle"``,
    the response matrix is included in the model of the probabilities, so that the observed
    frequencies are used as they are.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.tomography import StateTomography

            circuit = Circuit(2)
            circuit.add(gates.H(0))
            circuit.add(gates.CNOT(0, 1))

            # reconstruction of the Bell state from 9 circuits of 1000 shots
            estimate = StateTomography()(circuit, nshots=1000, seed=1)

            # the same data, with a maximum likelihood estimate
            estimate = StateTomography()(circuit, nshots=1000, seed=1, method="mle")

    References:
        1. M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*,
           Cambridge University Press (2010), Sec. 8.4.2.
        2. D. F. V. James, P. G. Kwiat, W. J. Munro and A. G. White,
           *Measurement of qubits*,
           `Phys. Rev. A 64, 052312 (2001) <https://doi.org/10.1103/PhysRevA.64.052312>`_.
        3. J. A. Smolin, J. M. Gambetta and G. Smith,
           *Efficient method for computing the maximum-likelihood quantum state from
           measurements with additive Gaussian noise*,
           `Phys. Rev. Lett. 108, 070502 (2012) <https://doi.org/10.1103/PhysRevLett.108.070502>`_.
        4. Z. Hradil, *Quantum-state estimation*,
           `Phys. Rev. A 55, R1561 (1997) <https://doi.org/10.1103/PhysRevA.55.R1561>`_.
        5. A. I. Lvovsky, *Iterative maximum-likelihood reconstruction in quantum homodyne
           tomography*, `J. Opt. B: Quantum Semiclass. Opt. 6, S556 (2004)
           <https://doi.org/10.1088/1464-4266/6/6/014>`_.
        6. A. Beck and M. Teboulle, *A fast iterative shrinkage-thresholding algorithm for
           linear inverse problems*,
           `SIAM J. Imaging Sci. 2, 183 (2009) <https://doi.org/10.1137/080716542>`_.
        7. S. Bravyi, S. Sheldon, A. Kandala, D. C. Mckay and J. M. Gambetta,
           *Mitigating measurement errors in multiqubit experiments*,
           `Phys. Rev. A 103, 042605 (2021) <https://doi.org/10.1103/PhysRevA.103.042605>`_.
    """

    def __call__(
        self,
        circuit: Circuit,
        nshots: int = 1000,
        seed: int | None = None,
        method: str = "linear-inversion",
        response_matrix: ArrayLike | None = None,
        iterations: int = 1000,
        tolerance: float = 1e-8,
        backend: Backend | None = None,
    ) -> ArrayLike:
        """Estimates the density matrix of the state prepared by ``circuit``.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit preparing the
                state, without measurements.
            nshots (int, optional): number of shots per measurement setting.
                Defaults to :math:`1000`.
            seed (int, optional): seed of the sampling of the outcomes. If ``None``, the
                generator of the backend is not reseeded. Defaults to ``None``.
            method (str, optional): estimator of the state. It is one of
                ``"linear-inversion"``, ``"physical-linear-inversion"``,
                ``"least-squares"``, or ``"mle"``. Only the last two are iterative, and
                use ``iterations`` and ``tolerance``. Defaults to ``"linear-inversion"``.
            response_matrix (ArrayLike, optional): :math:`2^{n} \\times 2^{n}` response
                matrix of the readout of the :math:`n` qubits, whose element
                :math:`(i, j)` is the probability of reading the outcome :math:`i` when the
                computational basis state :math:`j` is prepared. It is the matrix returned by
                :func:`qibo.models.error_mitigation.get_response_matrix`. If ``None``, the
                readout is assumed to be ideal. Defaults to ``None``.
            iterations (int, optional): maximum number of iterations of ``"least-squares"``
                and ``"mle"``. Defaults to :math:`1000`.
            tolerance (float, optional): the iterations of ``"least-squares"`` and ``"mle"``
                stop when the largest change of an element of the estimate is smaller than
                this value. Defaults to :math:`10^{-8}`.
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend
                to be used in the execution. If ``None``, it uses the current
                backend. Defaults to ``None``.

        Returns:
            ArrayLike: :math:`2^{n} \\times 2^{n}` estimated density matrix.
        """
        if isinstance(nshots, bool) or not isinstance(nshots, int) or nshots < 1:
            raise_error(
                ValueError,
                f"``nshots`` must be a positive int, but it is {nshots}.",
            )

        if method not in _METHODS:
            raise_error(
                ValueError,
                f"``method`` must be one of {_METHODS}, but it is {method}.",
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

        nqubits = circuit.nqubits
        dim = 2**nqubits

        if response_matrix is not None:
            response_matrix = backend.cast(response_matrix, dtype=backend.float64)
            if response_matrix.shape != (dim, dim):
                raise_error(
                    ValueError,
                    f"``response_matrix`` must have shape ({dim}, {dim}), "
                    + f"but it has shape {tuple(response_matrix.shape)}.",
                )

        circuits = self.circuits(circuit)

        if seed is not None:
            backend.set_seed(seed)

        results = self.execute(circuits, nshots, backend)

        settings = list(product("XYZ", repeat=nqubits))

        # frequencies of the outcomes of each setting, one setting for each row
        frequencies = backend.zeros((len(settings), dim), dtype=backend.float64)
        for index, result in enumerate(results):
            for outcome, frequency in result.frequencies(binary=False).items():
                frequencies[index, outcome] = frequency / nshots

        if method in ("linear-inversion", "physical-linear-inversion"):
            if response_matrix is not None:
                # the readout errors act on the probabilities of each setting
                frequencies = backend.matmul(
                    frequencies, backend.transpose(backend.inv(response_matrix))
                )

            # supports of the Pauli strings, as the bits of the Walsh-Hadamard transform,
            # and the weights of the qubits in the base-4 position of a Pauli string
            supports = list(product((0, 1), repeat=nqubits))
            weights = [4 ** (nqubits - 1 - qubit) for qubit in range(nqubits)]

            coefficients = backend.zeros(4**nqubits, dtype=backend.float64)
            counts = backend.zeros(4**nqubits, dtype=backend.float64)
            for setting, probabilities in zip(settings, frequencies):
                # the Walsh-Hadamard transform of the probabilities, which the Hadamard
                # transform divides by 2^n, has for each support the expectation value of
                # the Pauli string with the measured Pauli on the qubits of the support
                # and the identity on the others
                expectations = dim * backend.real(
                    hadamard_transform(probabilities, backend=backend)
                )

                # base-4 digits of the Pauli strings, with I, X, Y, Z as 0, 1, 2, 3
                digits = ["XYZ".index(pauli) + 1 for pauli in setting]
                indices = backend.cast(
                    [
                        sum(
                            digit * bit * weight
                            for digit, bit, weight in zip(digits, bits, weights)
                        )
                        for bits in supports
                    ],
                    dtype=backend.int64,
                )

                coefficients[indices] += expectations
                counts[indices] += 1.0

            # each Pauli string is averaged over all the settings that measure it
            state = _pauli_vectors_to_operator_fht(
                coefficients / counts, nqubits, dim, backend=backend
            )
            state = state / dim

            if method == "physical-linear-inversion":
                state = _project_physical(state, backend)

            return state

        if response_matrix is None:
            response_matrix = backend.identity(dim, dtype=backend.float64)

        # the rows of the unitaries that change the bases of the settings are the
        # vectors of the measurement bases, so that the probabilities of the outcomes
        # of all settings, for a state rho, are the diagonals of U rho U^{dagger}
        hadamard = backend.cast([[1, 1], [1, -1]], dtype=backend.complex128)
        hadamard = hadamard / math.sqrt(2)
        rotations = {
            "X": hadamard,
            "Y": hadamard @ backend.cast([[1, 0], [0, -1j]], dtype=backend.complex128),
            "Z": backend.identity(2, dtype=backend.complex128),
        }
        unitaries = backend.cast(
            [
                reduce(backend.kron, [rotations[pauli] for pauli in setting])
                for setting in settings
            ],
            dtype=backend.complex128,
        )
        conjugates = backend.conj(unitaries)

        identity = backend.identity(dim, dtype=backend.complex128)
        state = identity / dim

        if method == "mle":
            for _ in range(iterations):
                probabilities = backend.matmul(
                    backend.real(
                        backend.einsum(
                            "sbj,sbj->sb", backend.matmul(unitaries, state), conjugates
                        )
                    ),
                    backend.transpose(response_matrix),
                )

                # the operator R, which is the identity at the maximum of the likelihood
                ratios = backend.matmul(
                    frequencies / backend.maximum(probabilities, 1e-12),
                    response_matrix,
                ) / len(settings)
                operator = backend.einsum(
                    "sbi,sb,sbj->ij",
                    conjugates,
                    backend.cast(ratios, dtype=backend.complex128),
                    unitaries,
                )

                # diluted iteration, which increases the likelihood at every step
                update = identity + operator
                new_state = update @ state @ update
                new_state = new_state / backend.trace(new_state)

                converged = backend.max(backend.abs(new_state - state)) < tolerance
                state = new_state
                if converged:
                    break

            return state

        # the largest eigenvalue of the Hessian of the sum of squares, in the space of
        # the traceless matrices, which is that of the differences of two states
        lipschitz = 3 ** (nqubits - 1) * float(
            backend.eigvalsh(backend.transpose(response_matrix) @ response_matrix)[-1]
        )

        extrapolated = state
        momentum = 1.0
        for _ in range(iterations):
            probabilities = backend.matmul(
                backend.real(
                    backend.einsum(
                        "sbj,sbj->sb",
                        backend.matmul(unitaries, extrapolated),
                        conjugates,
                    )
                ),
                backend.transpose(response_matrix),
            )

            residuals = backend.matmul(probabilities - frequencies, response_matrix)
            gradient = backend.einsum(
                "sbi,sb,sbj->ij",
                conjugates,
                backend.cast(residuals, dtype=backend.complex128),
                unitaries,
            )

            new_state = _project_physical(extrapolated - gradient / lipschitz, backend)

            # accelerated projected gradient
            new_momentum = (1 + math.sqrt(1 + 4 * momentum**2)) / 2
            extrapolated = new_state + (momentum - 1) / new_momentum * (
                new_state - state
            )

            converged = backend.max(backend.abs(new_state - state)) < tolerance
            state, momentum = new_state, new_momentum
            if converged:
                break

        return state

    def circuits(self, circuit: Circuit) -> list[Circuit]:
        """Appends the basis changes and the measurements of each setting to ``circuit``.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit preparing the
                state, without measurements.

        Returns:
            list: :math:`3^{n}` circuits, one for each setting, in the order of
            ``itertools.product("XYZ", repeat=n)``, i.e. the Pauli of the first qubit
            changes the slowest. Each circuit measures all qubits, after rotating the
            Pauli :math:`X` with :class:`qibo.gates.H`, and the Pauli :math:`Y` with
            :class:`qibo.gates.SDG` followed by :class:`qibo.gates.H`.
        """
        self._check_circuit(circuit)

        nqubits = circuit.nqubits

        circuits = []
        for setting in product("XYZ", repeat=nqubits):
            tomography_circuit = circuit.copy()
            for qubit, pauli in enumerate(setting):
                if pauli == "Y":
                    tomography_circuit.add(gates.SDG(qubit))
                if pauli != "Z":
                    tomography_circuit.add(gates.H(qubit))
            tomography_circuit.add(gates.M(*range(nqubits)))
            circuits.append(tomography_circuit)

        return circuits


def _project_physical(state: ArrayLike, backend: Backend) -> ArrayLike:
    """Projects a Hermitian matrix onto the closest density matrix in the Frobenius norm.

    The eigenvalues are projected onto the probability simplex, and the eigenvectors
    are kept. It gives the same result as the algorithm of Ref. [1], which sets the
    smallest eigenvalues to zero and distributes their weight among the others.

    Args:
        state (ArrayLike): Hermitian matrix.
        backend (:class:`qibo.backends.abstract.Backend`): backend to be used in the
            calculation.

    Returns:
        ArrayLike: positive semidefinite matrix with unit trace.

    References:
        1. J. A. Smolin, J. M. Gambetta and G. Smith,
           *Efficient method for computing the maximum-likelihood quantum state from
           measurements with additive Gaussian noise*,
           `Phys. Rev. Lett. 108, 070502 (2012) <https://doi.org/10.1103/PhysRevLett.108.070502>`_.
    """
    eigenvalues, eigenvectors = backend.eigh((state + backend.dagger(state)) / 2)

    # the eigenvalues in decreasing order, and the shift that makes the positive ones
    # sum up to one
    ordered = backend.flip(eigenvalues, axis=0)
    shifts = (backend.cumsum(ordered) - 1.0) / backend.arange(1, len(ordered) + 1)
    nonzero = int(backend.sum(ordered - shifts > 0))
    eigenvalues = backend.maximum(eigenvalues - shifts[nonzero - 1], 0.0)

    eigenvectors_scaled = eigenvectors * backend.cast(eigenvalues, eigenvectors.dtype)

    return eigenvectors_scaled @ backend.dagger(eigenvectors)
