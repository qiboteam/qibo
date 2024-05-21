"""Testing DoubleBracketIteration model"""

from copy import deepcopy

import numpy as np
import pytest

from qibo import symbols
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.models.dbi.double_bracket import (
    DoubleBracketCostFunction,
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
    DoubleBracketScheduling,
)
from qibo.models.dbi.double_bracket_evolution_oracles import EvolutionOracle
from qibo.models.dbi.group_commutator_iteration_transpiler import (
    DoubleBracketRotationType,
    EvolutionOracleType,
    GroupCommutatorIterationWithEvolutionOracles,
)
from qibo.quantum_info import random_hermitian

NSTEPS = 1
seed = 10
"""Number of steps for evolution."""


@pytest.mark.parametrize("nqubits", [1, 2])
def test_double_bracket_iteration_canonical(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.canonical,
    )
    initial_off_diagonal_norm = dbi.off_diagonal_norm
    for _ in range(NSTEPS):
        dbi(step=np.sqrt(0.001))

    assert initial_off_diagonal_norm > dbi.off_diagonal_norm


@pytest.mark.parametrize("nqubits", [1, 2])
def test_double_bracket_iteration_group_commutator(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.group_commutator,
    )
    initial_off_diagonal_norm = dbi.off_diagonal_norm

    # test first iteration with default d
    dbi(mode=DoubleBracketGeneratorType.group_commutator, step=0.01)
    for _ in range(NSTEPS):
        dbi(step=0.01, d=d)

    assert initial_off_diagonal_norm > dbi.off_diagonal_norm


@pytest.mark.parametrize("nqubits", [3])
def test_double_bracket_iteration_eval_dbr_unitary(backend, nqubits):
    r"""The bound is $$||e^{-[D,H]}-GC||\le s^{3/2}(||[H,[D,H]||+||[D,[D,H]]||$$ which we check by a loglog fit."""
    h0 = random_hermitian(2**nqubits, backend=backend)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.group_commutator,
    )

    times = np.linspace(0.001, 0.01, 10)
    norms = []
    norms_bound = []
    for s in times:
        u = dbi.eval_dbr_unitary(
            s, d=d, mode=DoubleBracketGeneratorType.single_commutator
        )
        v = dbi.eval_dbr_unitary(
            s, d=d, mode=DoubleBracketGeneratorType.group_commutator
        )

        norms.append(np.linalg.norm(u - v))
        w = dbi.commutator(h0, d)
        norms_bound.append(
            0.5
            * s**1.48
            * (
                np.linalg.norm(dbi.commutator(h0, w))
                + np.linalg.norm(dbi.commutator(d, w))
            )
        )
        assert np.linalg.norm(u - v) < 10 * s**1.49 * (
            np.linalg.norm(h0) + np.linalg.norm(d)
        ) * np.linalg.norm(h0) * np.linalg.norm(d)


@pytest.mark.parametrize("nqubits", [3])
def test_dbi_evolution_oracle(backend, nqubits, t_step=0.1, eps=0.001):
    """We test the basic functionality provided by `EvolutionOracle`:
    - hamiltonian_simulation: will use `SymbolicHamiltonian.circuit()` and should match with the corresponding evolution unitary up to the discretization error threshold
    - numerical is just exponential $e^{-1jt_{step} H}$
    - text_strings will have strings which just have the name of the evolution oracle
    """
    from numpy.linalg import norm

    from qibo import symbols
    from qibo.hamiltonians import SymbolicHamiltonian

    h_x = SymbolicHamiltonian(
        symbols.X(0)
        + symbols.Z(0) * symbols.X(1)
        + symbols.Y(2)
        + symbols.Y(1) * symbols.Y(2),
        nqubits=3,
    )
    d_0 = SymbolicHamiltonian(symbols.Z(0), nqubits=3)
    h_input = h_x + d_0

    evolution_oracle = EvolutionOracle(
        h_input, "ZX", mode_evolution_oracle=EvolutionOracleType.hamiltonian_simulation
    )
    evolution_oracle.eps_trottersuzuki = eps

    U_hamiltonian_simulation = evolution_oracle.circuit(t_step).unitary()
    V_target = h_input.exp(t_step)

    assert norm(U_hamiltonian_simulation - V_target) < eps

    evolution_oracle_np = EvolutionOracle(
        h_input, "ZX numpy", mode_evolution_oracle=EvolutionOracleType.numerical
    )
    U_np = evolution_oracle_np.circuit(t_step)
    assert norm(U_np - V_target) < 1e-10

    evolution_oracle_txt = EvolutionOracle(
        h_input, "ZX test", mode_evolution_oracle=EvolutionOracleType.text_strings
    )
    U_txt = evolution_oracle_txt.circuit(t_step)
    assert isinstance(U_txt, str)


def test_gci_evolution_oracles_types_numerical(
    backend, nqubits=3, t_step=1e-3, eps=1e-3
):
    r"""

    This is testing the following:

    `dbi_exact` runs $V_{exact} = e^{-sW}$ and rotates $H_1 = V_{exact}^\dagger H_0 V_{exact}$.

    `dbi_GC` runs $V_{GC} = GC$ and rotates $K_1 = V_{GC}^\dagger H_0 V_{GC}$.

    We assert that dbi_exact and dbi_GC should be within the approximation bound of the GC
    $$||J_1-H_1||\le2 ||H_0||\,||R-V||\le C ||H_0|| s^{3/2}$$

    `gci` runs $V_{EO,GC} = GC$ and rotates $J_1 = V_{EO,GC}^\dagger H_0 V_{EO,GC}$.

    We assert that gci and dbi2 should be within machine precision for the correct sorting.
    $$||J_1-K_1||\le2 ||H_0||\,||R-Q||\le \epsilon$$
    """
    from numpy.linalg import norm

    h_x = SymbolicHamiltonian(
        symbols.X(0)
        + symbols.Z(0) * symbols.X(1)
        + symbols.Y(2)
        + symbols.Y(1) * symbols.Y(2),
        nqubits=3,
    )
    d_0 = SymbolicHamiltonian(symbols.Z(0), nqubits=3)
    h_input = h_x + d_0

    dbi = DoubleBracketIteration(deepcopy(h_input.dense))

    w = dbi.commutator(h_input.dense.matrix, d_0.dense.matrix)
    norms_bound = (
        0.5
        * t_step**1.48
        * (
            np.linalg.norm(dbi.commutator(h_input.dense.matrix, w))
            + np.linalg.norm(dbi.commutator(d_0.matrix, w))
        )
    )

    v_exact = dbi.eval_dbr_unitary(
        t_step, d=d_0.dense.matrix, mode=DoubleBracketGeneratorType.single_commutator
    )
    v_gc = dbi.eval_dbr_unitary(
        t_step, d=d_0.dense.matrix, mode=DoubleBracketGeneratorType.group_commutator
    )
    assert norm(v_exact - v_gc) < norms_bound

    dbi(t_step, d=d_0.dense.matrix)
    h_1 = dbi.h.matrix

    dbi.h = deepcopy(h_input.dense)
    dbi(t_step, d=d_0.dense.matrix, mode=DoubleBracketGeneratorType.group_commutator)
    k_1 = dbi.h.matrix

    assert norm(h_1 - k_1) < 2 * norm(h_input.dense.matrix) * norms_bound

    evolution_oracle = EvolutionOracle(
        h_input, "ZX", mode_evolution_oracle=EvolutionOracleType.numerical
    )

    evolution_oracle_diagonal_target = EvolutionOracle(
        d_0, "D0", mode_evolution_oracle=EvolutionOracleType.numerical
    )

    gci = GroupCommutatorIterationWithEvolutionOracles(deepcopy(evolution_oracle))

    u_gc_from_oracles = gci.group_commutator(t_step, evolution_oracle_diagonal_target)
    assert (
        norm(u_gc_from_oracles["forwards"].conj().T - u_gc_from_oracles["backwards"])
        < 1e-10
    )

    assert (
        norm(v_exact - gci.eval_gcr_unitary(t_step, evolution_oracle_diagonal_target))
        < norms_bound
    )

    gci(t_step, diagonal_association=evolution_oracle_diagonal_target)
    j_1 = gci.h.matrix

    assert norm(h_1 - j_1) < 2 * norm(h_input.dense.matrix) * norms_bound

    assert norm(j_1 - k_1) < 1e-10

    # test single_commutator case
    gci = GroupCommutatorIterationWithEvolutionOracles(deepcopy(evolution_oracle))
    gci.mode_double_bracket_rotation = DoubleBracketRotationType.single_commutator
    u_gc_from_oracles = gci.group_commutator(t_step, evolution_oracle_diagonal_target)
    assert (
        norm(u_gc_from_oracles["forwards"].conj().T - u_gc_from_oracles["backwards"])
        < 1e-10
    )

    # make 2 new objects with DoubleBracketRotationType.group_commutator_reduced and group_commutator, compare h matrix rotation


@pytest.mark.parametrize("nqubits", [3])
@pytest.mark.parametrize(
    "mode_evolution_oracle",
    [EvolutionOracleType.hamiltonian_simulation, EvolutionOracleType.numerical],
)
def test_gci_frame_shifted_oracles(backend, nqubits, mode_evolution_oracle):
    r"""In a group commutator iteration (GCI) we have
    $$J_{k+1}= U_k^\dagger J_k U_k$$
    which is obtained by a product formula for $U_k$.
    We will use two examples
    $$A_k = e^{is D} e^{is J_k} e^{-isD}$$

    This means that $A_k$ and $B_k$ schemes should give the same `Groupand
    $$B_k = e^{-is J_k}e^{is D} e^{is J_k} e^{-isD}$$
    In both cases $D$ is fixed, which amounts to a product formula approximation of the BHMM scheme.

    For $B_k$ we have the group commutator bound, see below. For $A_k$ we will have that
    $$J_{k+1}= A_k^\dagger J_k A_k= B_k^\dagger J_k B_k$$
    because of a reduction by means of a commutator vanishing (the ordering was chosen on purpose).
    CommutatorIterationWithEvolutionOracles.h`. Additionally that should be also `DoubleBracketIteration.h` as long as the ordering is correct.

    If we operate in the `EvolutionOracleType.hamiltonian_simulation` there will be deviations based on the `EvolutionOracle.eps_trottersuzuki` threshold.
    """

    h_x = SymbolicHamiltonian(
        symbols.X(0)
        + symbols.Z(0) * symbols.X(1)
        + symbols.Y(2)
        + symbols.Y(1) * symbols.Y(2),
        nqubits=3,
    )
    d_0 = SymbolicHamiltonian(symbols.Z(0), nqubits=3)
    h_input = h_x + d_0

    dbi = DoubleBracketIteration(deepcopy(h_input.dense))

    evolution_oracle = EvolutionOracle(h_input, "ZX", mode_evolution_oracle)
    gci = GroupCommutatorIterationWithEvolutionOracles(deepcopy(evolution_oracle))

    evolution_oracle_diagonal_target = EvolutionOracle(d_0, "D0", mode_evolution_oracle)

    from numpy.linalg import norm

    if mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation:
        threshold = evolution_oracle.eps_trottersuzuki * 10
    else:
        threshold = 1e-10

    r = 0.01
    for _ in range(3):
        a = dbi.h.exp(r)
        b = gci.iterated_hamiltonian_evolution_oracle.eval_unitary(r)

        assert norm(a - b) < threshold
        a = dbi.eval_dbr_unitary(
            r, d=d_0.dense.matrix, mode=DoubleBracketGeneratorType.group_commutator
        )
        b = gci.eval_gcr_unitary(r, evolution_oracle_diagonal_target)

        assert norm(a - b) < threshold
        dbi(r, d=d_0.dense.matrix, mode=DoubleBracketGeneratorType.group_commutator)
        gci(r, diagonal_association=evolution_oracle_diagonal_target)

        k_r = dbi.h.matrix
        j_r = gci.h.matrix

        assert norm(a - b) < threshold

        assert norm(norm(dbi.sigma(k_r)) - norm(dbi.sigma(j_r))) < threshold
    gci.iterated_hamiltonian_evolution_oracle.get_composed_circuit()


@pytest.mark.parametrize("nqubits", [3])
@pytest.mark.parametrize("mode_evolution_oracle", [EvolutionOracleType.numerical])
def test_gci_implementation_normal_and_oracles(backend, nqubits, mode_evolution_oracle):
    """The regular implementation in `DoubleBracketIteration` should be viewed as using classical dynamic programming: memoization of the updated Hamiltonian is used explicitly. Instead, using `FrameShiftedEvolutionOracle` we can store a sequence of recursive rotations such that eventually the iteration gives the same result.
    This function tests numerical agreement using the numpy backend.
    """
    h_x = SymbolicHamiltonian(
        symbols.X(0)
        + symbols.Z(0) * symbols.X(1)
        + symbols.Y(2)
        + symbols.Y(1) * symbols.Y(2),
        nqubits=3,
    )
    d_0 = SymbolicHamiltonian(symbols.Z(0), nqubits=3)
    h_input = h_x + d_0

    dbi = DoubleBracketIteration(deepcopy(h_input.dense))

    evolution_oracle = EvolutionOracle(h_input, "ZX", mode_evolution_oracle)
    gci = GroupCommutatorIterationWithEvolutionOracles(deepcopy(evolution_oracle))

    evolution_oracle_diagonal_target = EvolutionOracle(d_0, "D0", mode_evolution_oracle)

    from numpy.linalg import norm

    if mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation:
        threshold = evolution_oracle.eps_trottersuzuki * 10
    else:
        threshold = 1e-10

    times = np.linspace(1e-5, 1, 5)
    for r in times:

        dbi(r, d=d_0.dense.matrix, mode=DoubleBracketGeneratorType.group_commutator)
        k_r = dbi.h.matrix
        dbi.h = deepcopy(h_input.dense)
        gci(r, diagonal_association=evolution_oracle_diagonal_target)
        j_r = gci.h.matrix
        gci.h = deepcopy(h_input.dense)
        gci.iterated_hamiltonian_evolution_oracle = deepcopy(evolution_oracle)

        assert norm(k_r - j_r) < threshold

    r = 1

    for _ in range(3):
        dbi(r, d=d_0.dense.matrix, mode=DoubleBracketGeneratorType.group_commutator)
        gci(r, diagonal_association=evolution_oracle_diagonal_target)
        k_r = dbi.h.matrix
        j_r = gci.h.matrix

        assert norm(k_r - j_r) < threshold


@pytest.mark.parametrize("nqubits", [1, 2])
def test_double_bracket_iteration_single_commutator(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.single_commutator,
    )
    initial_off_diagonal_norm = dbi.off_diagonal_norm

    # test first iteration with default d
    dbi(mode=DoubleBracketGeneratorType.single_commutator, step=0.01)
    for _ in range(NSTEPS):
        dbi(step=0.01, d=d)

    assert initial_off_diagonal_norm > dbi.off_diagonal_norm


@pytest.mark.parametrize("nqubits", [3, 4])
def test_hyperopt_step(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbi = DoubleBracketIteration(Hamiltonian(nqubits, h0, backend=backend))
    dbi.scheduling = DoubleBracketScheduling.hyperopt
    # find initial best step with look_ahead = 1
    initial_step = 0.01
    delta = 0.02
    step = dbi.choose_step(
        step_min=initial_step - delta, step_max=initial_step + delta, max_evals=100
    )

    assert step != initial_step

    # evolve following with optimized first step
    for generator in DoubleBracketGeneratorType:
        dbi(mode=generator, step=step, d=d)

    # find the following step size with look_ahead
    look_ahead = 3

    step = dbi.choose_step(
        step_min=initial_step - delta,
        step_max=initial_step + delta,
        max_evals=10,
        look_ahead=look_ahead,
    )

    # evolve following the optimized first step
    for gentype in range(look_ahead):
        dbi(mode=DoubleBracketGeneratorType(gentype + 1), step=step, d=d)


def test_energy_fluctuations(backend):
    h0 = np.array([[1, 0], [0, -1]])
    h0 = backend.cast(h0, dtype=backend.dtype)

    state = np.array([1, 0])
    state = backend.cast(state, dtype=backend.dtype)

    dbi = DoubleBracketIteration(Hamiltonian(1, matrix=h0, backend=backend))
    energy_fluctuation = dbi.energy_fluctuation(state=state)
    assert energy_fluctuation == 1.0


@pytest.mark.parametrize(
    "scheduling",
    [
        DoubleBracketScheduling.grid_search,
        DoubleBracketScheduling.hyperopt,
        #        DoubleBracketScheduling.simulated_annealing,
    ],
)
@pytest.mark.parametrize("nqubits", [3, 4, 5])
def test_double_bracket_iteration_scheduling_grid_hyperopt_annealing(
    backend, nqubits, scheduling
):
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.single_commutator,
    )
    initial_off_diagonal_norm = dbi.off_diagonal_norm
    for _ in range(NSTEPS):
        step1 = dbi.choose_step(d=d, scheduling=scheduling)
        dbi(d=d, step=step1)
    step2 = dbi.choose_step()
    dbi(step=step2)
    assert initial_off_diagonal_norm > dbi.off_diagonal_norm


@pytest.mark.parametrize("nqubits", [3, 4, 6])
@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize(
    "cost",
    [
        DoubleBracketCostFunction.least_squares,
        DoubleBracketCostFunction.off_diagonal_norm,
    ],
)
def test_double_bracket_iteration_scheduling_polynomial(backend, nqubits, n, cost):
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.single_commutator,
        scheduling=DoubleBracketScheduling.polynomial_approximation,
        cost=cost,
    )
    initial_off_diagonal_norm = dbi.off_diagonal_norm
    for _ in range(NSTEPS):
        step1 = dbi.choose_step(d=d, n=n)
        dbi(d=d, step=step1)
    assert initial_off_diagonal_norm > dbi.off_diagonal_norm
