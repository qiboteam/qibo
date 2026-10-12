"""Double-bracket iterations for the diagonalization of Hamiltonians."""

import math

from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.hamiltonians.abstract import AbstractHamiltonian
from qibo.models.circuit import Circuit


class DoubleBracketIteration:
    """Diagonalization double-bracket iteration (DBI) of a Hamiltonian.

    Given a Hamiltonian :math:`H_{0}`, the :math:`k`-th recursion step of the DBI is

    .. math::
        H_{k+1} = e^{s_{k} \\, W_{k}} \\, H_{k} \\, e^{-s_{k} \\, W_{k}} \\, ,
        \\quad W_{k} = [D_{k}, H_{k}] \\, ,

    where :math:`D_{k}` is a diagonal operator and :math:`s_{k}` is the step duration.
    If :math:`D_{k}` is not specified, the dephasing :math:`\\Delta(H_{k})`, i.e. the diagonal
    restriction of :math:`H_{k}`, is used. This choice, called canonical bracket, gives rise to
    the Głazek-Wilson-Wegner (GWW) DBI, which decreases the Hilbert-Schmidt norm of the
    off-diagonal restriction of :math:`H_{k}` for sufficiently short durations.

    Two types of rotation are available. In the ``"double_bracket"`` rotation, the exact unitary
    :math:`e^{-s_{k} \\, W_{k}}` is applied. In the ``"group_commutator"`` rotation, the unitary
    :math:`e^{i \\sqrt{s_{k}} \\, D_{k}} \\, e^{i \\sqrt{s_{k}} \\, H_{k}} \\,
    e^{-i \\sqrt{s_{k}} \\, D_{k}}` is applied instead, which is the group commutator iteration
    (GCI) that can be implemented in a quantum computer through :meth:`circuit`.
    A negative ``step_duration`` corresponds to flipping the sign of :math:`D_{k}`.

    Args:
        hamiltonian (ArrayLike or :class:`qibo.hamiltonians.abstract.AbstractHamiltonian`):
            Hamiltonian :math:`H_{0}` to be diagonalized. Must be hermitian.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in the
            execution. If ``None``, it uses the current backend. Defaults to ``None``.

    Example:
        .. testcode::

            from qibo.backends import NumpyBackend
            from qibo.hamiltonians import TFIM
            from qibo.models.double_bracket import DoubleBracketIteration

            backend = NumpyBackend()

            hamiltonian = TFIM(3, h=1.0, backend=backend)
            dbi = DoubleBracketIteration(hamiltonian, backend=backend)
            norms = dbi.run(nsteps=5)
            print(norms[0] > norms[-1])

        .. testoutput::

            True

    References:
        1.  M. Gluza, *Double-bracket quantum algorithms for diagonalization*,
        `Quantum (2024) <https://doi.org/10.22331/q-2024-04-08-1316>`_,
        `arXiv:2206.11772 <https://arxiv.org/abs/2206.11772>`_.
    """

    def __init__(
        self,
        hamiltonian: ArrayLike | AbstractHamiltonian,
        backend: Backend | None = None,
    ):
        backend = _check_backend(backend)

        if isinstance(hamiltonian, AbstractHamiltonian):
            hamiltonian = hamiltonian.matrix

        hamiltonian = backend.cast(hamiltonian, dtype=backend.complex128)

        if (
            len(hamiltonian.shape) != 2
            or hamiltonian.shape[0] != hamiltonian.shape[1]
            or backend.max(backend.abs(hamiltonian - backend.conj(hamiltonian.T)))
            > 1e-10
        ):
            raise_error(ValueError, "``hamiltonian`` must be a hermitian matrix.")

        self.backend = backend
        self.history = []
        self.initial_hamiltonian = hamiltonian
        self.hamiltonian = hamiltonian
        self.unitary = backend.identity(hamiltonian.shape[0], dtype=backend.complex128)

    @property
    def energy_fluctuations(self) -> ArrayLike:
        r"""Energy fluctuations :math:`\sqrt{\langle H_{k}^{2} \rangle - \langle H_{k}
        \rangle^{2}}` of the current Hamiltonian :math:`H_{k}` in all computational basis states.

        Fluctuations vanish for eigenstates of :math:`H_{k}`.
        """
        backend = self.backend
        energies = backend.real(backend.diag(self.hamiltonian))
        squares = backend.real(backend.diag(self.hamiltonian @ self.hamiltonian))

        return backend.sqrt(backend.abs(squares - energies**2))

    @property
    def off_diagonal_norm(self) -> float:
        r"""Hilbert-Schmidt norm :math:`\|\sigma(H_{k})\|_{\text{HS}}` of the off-diagonal
        restriction of the current Hamiltonian :math:`H_{k}`."""
        return self._off_diagonal_norm(self.hamiltonian)

    def circuit(self, backend: Backend | None = None) -> Circuit:
        r"""Circuit :math:`V_{K}` of the group commutator iteration (GCI) after :math:`K` steps.

        The circuit is built recursively from the recorded step durations :math:`s_{k}` and
        diagonal operators :math:`D_{k}` as

        .. math::
            V_{k+1} = V_{k} \, e^{i \sqrt{s_{k}} \, D_{k}} \, V_{k}^{\dagger} \,
            e^{i \sqrt{s_{k}} \, H_{0}} \, V_{k} \, e^{-i \sqrt{s_{k}} \, D_{k}} \, ,

        with :math:`V_{0} = \mathbb{1}`, so that :math:`H_{K} = V_{K}^{\dagger} \, H_{0} \,
        V_{K}` and :math:`V_{K} |\mu\rangle` approximates an eigenstate of :math:`H_{0}`
        for a computational basis state :math:`|\mu\rangle`. It only contains evolutions
        under :math:`H_{0}` and under the diagonal operators, and its number of evolutions
        under :math:`H_{0}` is :math:`(3^{K} - 1) / 2`. It reproduces the iteration exactly when
        the ``"group_commutator"`` rotation was used.

        Args:
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend used to
                build the evolution gates. If ``None``, it uses the backend of the iteration.
                Defaults to ``None``.

        Returns:
            :class:`qibo.models.circuit.Circuit`: Circuit implementing :math:`V_{K}`.
        """
        from qibo.quantum_info.linalg_operations import (  # circular import
            matrix_exponentiation,
        )

        backend = self.backend if backend is None else _check_backend(backend)
        nqubits = int(math.log2(self.initial_hamiltonian.shape[0]))
        qubits = list(range(nqubits))

        circuit = Circuit(nqubits)
        for duration, diagonal in self.history:
            time = math.sqrt(abs(duration))
            sign = 1 if duration >= 0 else -1
            phases = backend.exp(1j * sign * time * diagonal)

            new_circuit = Circuit(nqubits)
            new_circuit.add(gates.Unitary(backend.diag(backend.conj(phases)), *qubits))
            new_circuit.add(circuit.copy(deep=True).queue)
            new_circuit.add(
                gates.Unitary(
                    matrix_exponentiation(
                        self.initial_hamiltonian, 1j * time, backend=backend
                    ),
                    *qubits,
                )
            )
            new_circuit.add(circuit.invert().queue)
            new_circuit.add(gates.Unitary(backend.diag(phases), *qubits))
            new_circuit.add(circuit.copy(deep=True).queue)
            circuit = new_circuit

        return circuit

    def optimize_step(
        self,
        diagonal: ArrayLike | None = None,
        rotation: str = "double_bracket",
        step_max: float | None = None,
        nsamples: int = 100,
    ) -> float:
        r"""Step duration that maximally decreases the off-diagonal norm in one step.

        A grid search over :math:`[-s_{\max}, s_{\max}]` (:math:`[0, s_{\max}]` for the
        canonical bracket) is followed by a bounded local refinement around the best point.

        Args:
            diagonal (ArrayLike, optional): diagonal operator :math:`D_{k}`, given either as the
                vector of its diagonal entries or as a diagonal matrix. If ``None``, the
                dephasing of the current Hamiltonian is used. Defaults to ``None``.
            rotation (str, optional): either ``"double_bracket"`` or ``"group_commutator"``.
                Defaults to ``"double_bracket"``.
            step_max (float, optional): upper limit of the search. If ``None``, it is set to
                :math:`\pi / \|[D_{k}, H_{k}]\|`, with the operator norm. Defaults to ``None``.
            nsamples (int, optional): number of points of the grid search. Defaults to
                :math:`100`.

        Returns:
            float: Optimal step duration.
        """
        from qibo.quantum_info.linalg_operations import commutator  # circular import

        canonical = diagonal is None
        diagonal = self._diagonal(diagonal)

        if step_max is None:
            bracket = commutator(self.backend.diag(diagonal), self.hamiltonian)
            norm = float(self.backend.matrix_norm(bracket, order=2))
            if norm < 1e-12:
                return 0.0
            step_max = math.pi / norm

        lower = 0.0 if canonical else -step_max
        delta = (step_max - lower) / (nsamples - 1)
        grid = [lower + j * delta for j in range(nsamples)]
        costs = [self._cost(step, diagonal, rotation) for step in grid]
        best = costs.index(min(costs))

        result = minimize_scalar(
            lambda step: self._cost(step, diagonal, rotation),
            bounds=(max(lower, grid[best] - delta), min(step_max, grid[best] + delta)),
            method="bounded",
        )

        return float(result.x) if result.fun < costs[best] else grid[best]

    def run(
        self,
        nsteps: int,
        diagonals: list[ArrayLike] | None = None,
        rotation: str = "double_bracket",
        step_max: float | None = None,
        nsamples: int = 100,
    ) -> list[float]:
        r"""Runs ``nsteps`` greedy recursion steps of the iteration.

        In each step, the step duration is optimized for the canonical bracket and for each
        diagonal operator in ``diagonals``, and the choice yielding the smallest off-diagonal
        norm is applied.

        Args:
            nsteps (int): number of recursion steps.
            diagonals (list[ArrayLike], optional): candidate diagonal operators :math:`D_{k}`,
                e.g. products of Pauli-:math:`Z` operators, to be considered in addition to the
                canonical bracket in every step. Defaults to ``None``.
            rotation (str, optional): either ``"double_bracket"`` or ``"group_commutator"``.
                Defaults to ``"double_bracket"``.
            step_max (float, optional): see :meth:`optimize_step`. Defaults to ``None``.
            nsamples (int, optional): see :meth:`optimize_step`. Defaults to :math:`100`.

        Returns:
            list[float]: Off-diagonal norms before the first step and after each step.
        """
        norms = [self.off_diagonal_norm]
        for _ in range(nsteps):
            candidates = [None] + (diagonals if diagonals is not None else [])
            options = []
            for diagonal in candidates:
                step = self.optimize_step(diagonal, rotation, step_max, nsamples)
                cost = self._cost(step, self._diagonal(diagonal), rotation)
                options.append((cost, step, diagonal))
            _, step, diagonal = min(options, key=lambda option: option[0])
            self.step(step, diagonal, rotation)
            norms.append(self.off_diagonal_norm)

        return norms

    def step(
        self,
        step_duration: float,
        diagonal: ArrayLike | None = None,
        rotation: str = "double_bracket",
    ) -> ArrayLike:
        """Performs one recursion step and updates the iterated Hamiltonian.

        Args:
            step_duration (float): step duration :math:`s_{k}`.
            diagonal (ArrayLike, optional): diagonal operator :math:`D_{k}`, given either as the
                vector of its diagonal entries or as a diagonal matrix. If ``None``, the
                dephasing of the current Hamiltonian is used. Defaults to ``None``.
            rotation (str, optional): either ``"double_bracket"`` or ``"group_commutator"``.
                Defaults to ``"double_bracket"``.

        Returns:
            ArrayLike: Iterated Hamiltonian :math:`H_{k+1}`.
        """
        diagonal = self._diagonal(diagonal)
        rotation_unitary = self._rotation(step_duration, diagonal, rotation)

        self.hamiltonian = (
            self.backend.conj(rotation_unitary.T) @ self.hamiltonian @ rotation_unitary
        )
        self.unitary = self.unitary @ rotation_unitary
        self.history.append((step_duration, diagonal))

        return self.hamiltonian

    def _cost(self, step_duration: float, diagonal: ArrayLike, rotation: str) -> float:
        """Off-diagonal norm of the Hamiltonian after a rotation, without applying it."""
        rotation_unitary = self._rotation(step_duration, diagonal, rotation)
        rotated = (
            self.backend.conj(rotation_unitary.T) @ self.hamiltonian @ rotation_unitary
        )

        return self._off_diagonal_norm(rotated)

    def _diagonal(self, diagonal: ArrayLike | None) -> ArrayLike:
        """Vector of diagonal entries of :math:`D_{k}`, using dephasing if ``None``."""
        if diagonal is None:
            return self.backend.real(self.backend.diag(self.hamiltonian))

        diagonal = self.backend.cast(diagonal, dtype=self.backend.complex128)
        dim = self.hamiltonian.shape[0]

        if len(diagonal.shape) == 2:
            vector = self.backend.diag(diagonal)
            if (
                self.backend.max(self.backend.abs(diagonal - self.backend.diag(vector)))
                > 1e-10
            ):
                raise_error(ValueError, "``diagonal`` must be a diagonal matrix.")
            diagonal = vector

        if diagonal.shape != (dim,):
            raise_error(
                ValueError,
                f"``diagonal`` must have {dim} entries, but has shape {diagonal.shape}.",
            )

        return self.backend.real(diagonal)

    def _off_diagonal_norm(self, matrix: ArrayLike) -> float:
        """Hilbert-Schmidt norm of the off-diagonal restriction of ``matrix``."""
        off_diagonal = matrix - self.backend.diag(self.backend.diag(matrix))

        return float(self.backend.matrix_norm(off_diagonal, order="fro"))

    def _rotation(
        self, step_duration: float, diagonal: ArrayLike, rotation: str
    ) -> ArrayLike:
        """Unitary implementing one rotation of the iteration."""
        from qibo.quantum_info.linalg_operations import (  # circular import
            commutator,
            matrix_exponentiation,
        )

        if rotation not in ("double_bracket", "group_commutator"):
            raise_error(
                ValueError,
                "``rotation`` must be either 'double_bracket' or 'group_commutator', "
                + f"but is {rotation}.",
            )

        diag_matrix = self.backend.diag(diagonal)

        if rotation == "double_bracket":
            bracket = commutator(diag_matrix, self.hamiltonian)

            return matrix_exponentiation(bracket, -step_duration, backend=self.backend)

        time = math.sqrt(abs(step_duration))
        if step_duration < 0:
            diag_matrix = -diag_matrix

        forward = matrix_exponentiation(diag_matrix, 1j * time, backend=self.backend)
        evolution = matrix_exponentiation(
            self.hamiltonian, 1j * time, backend=self.backend
        )

        return forward @ evolution @ self.backend.conj(forward)
