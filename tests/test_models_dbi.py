"""Testing DoubleBracketIteration model"""

import numpy as np
import pytest

from qibo.hamiltonians import Hamiltonian
from qibo.models.dbi.double_bracket import (
    DoubleBracketCostFunction,
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
    DoubleBracketScheduling,
)
from qibo.models.dbi.double_bracket_evolution_oracles import EvolutionOracle
from qibo.models.dbi.group_commutator_iteration_transpiler import *
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

        norms.append(np.linalg.norm(u - v) )
        w = dbi.commutator(h0,d)
        norms_bound.append(0.5*s**1.48 * (
            np.linalg.norm(dbi.commutator(h0,w)) + np.linalg.norm(dbi.commutator(d,w))
        ))
        assert np.linalg.norm(u - v) < 10 * s**1.49 * (
                np.linalg.norm(h0) + np.linalg.norm(d)
            ) * np.linalg.norm(h0) * np.linalg.norm(d)



@pytest.mark.parametrize("nqubits", [3])
def test_dbi_evolution_oracle(backend, nqubits, t_step = 0.1, eps = 0.001 ):
    """ We test the basic functionality provided by `EvolutionOracle`:
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
    assert norm(U_np - V_target) < 1e-12

    evolution_oracle_txt = EvolutionOracle(
        h_input, "ZX test", mode_evolution_oracle=EvolutionOracleType.text_strings
    )
    U_txt = evolution_oracle_txt.circuit(t_step)   
    assert isinstance(U_txt, str)



def test_gci_evolution_oracles_types_numerical(nqubits,backend,t_step, eps):
    """

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

    v_exact = dbi.eval_dbr_unitary(t_step, d=d_0.dense.matrix, mode=DoubleBracketGeneratorType.single_commutator)
    v_gc = dbi.eval_dbr_unitary(t_step, d=d_0.dense.matrix, mode=DoubleBracketGeneratorType.group_commutator)

    dbi(t_step, d = d_0.dense.matrix )
    h_1 = dbi.h.matrix

    dbi.h = deepcopy(h_input.dense)
    dbi(t_step, d = d_0.dense.matrix, mode = DoubleBracketGeneratorType.group_commutator )
    k_1 = dbi.h.matrix

    w = dbi.commutator(h_input.dense.matrix,d_0.dense.matrix)
    norms_bound = 0.5*t_step**1.48 * (
        np.linalg.norm(dbi.commutator(h_input.dense.matrix,w)) + np.linalg.norm(dbi.commutator(d_0.matrix,w))
    )
    assert norm(v_exact - v_gc) < norms_bound
    assert norm(h_1-k_1) < 2 * norm(h_input.dense.matrix) * norms_bound   

    evolution_oracle = EvolutionOracle(h_input, "ZX",
                        mode_evolution_oracle = EvolutionOracleType.numerical)    
    d_02 = SymbolicHamiltonian(symbols.Z(0), nqubits=3)
    evolution_oracle_diagonal_target =  EvolutionOracle(d_02, "D0",
               mode_evolution_oracle = EvolutionOracleType.numerical)

    gci = GroupCommutatorIterationWithEvolutionOracles( deepcopy(evolution_oracle ))
    #gci.mode_double_bracket_rotation = DoubleBracketRotationType.group_commutator

    u_gc_from_oracles = gci.group_commutator( t_step, evolution_oracle_diagonal_target )   
    u_gci = u_gc_from_oracles['forwards']

    assert norm(u_gci.conj().T - u_gc_from_oracles['backwards']) < 1e-12


    v_exact = dbi.eval_dbr_unitary(t_step, d=d_0.dense.matrix, mode=DoubleBracketGeneratorType.single_commutator)
    w = dbi.commutator(h_input.dense.matrix,d_0.dense.matrix)
    norms_bound = 0.5*t_step**1.48 * (
        np.linalg.norm(dbi.commutator(h_input.dense.matrix,w)) + np.linalg.norm(dbi.commutator(d_0.matrix,w))
    )
    assert norm(v_exact - u_gci) < norms_bound

    gci(t_step, diagonal_association= evolution_oracle_diagonal_target )
    j_1 = gci.iterated_hamiltonian_evolution_oracle.h.matrix
    assert norm(h_1-j_1) < 2 * norm(h_input.dense.matrix) * norms_bound   
    assert norm(j_1-k_1) < 1e-12 
    
    
    



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
