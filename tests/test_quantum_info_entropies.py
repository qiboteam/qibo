import math

import numpy as np
import pytest

from qibo.config import PRECISION_TOL
from qibo.quantum_info.basis import pauli_basis
from qibo.quantum_info.entropies import (
    classical_mutual_information,
    classical_relative_entropy,
    classical_relative_renyi_entropy,
    classical_relative_tsallis_entropy,
    classical_renyi_entropy,
    classical_tsallis_entropy,
    conditional_entropy,
    entanglement_entropy,
    linear_entropy,
    mutual_information,
    relative_entropy_of_coherence,
    relative_renyi_entropy,
    relative_tsallis_entropy,
    relative_von_neumann_entropy,
    renyi_entropy,
    shannon_entropy,
    stabilizer_renyi_entropy,
    tsallis_entropy,
    von_neumann_entropy,
)
from qibo.quantum_info.linalg_operations import matrix_power
from qibo.quantum_info.random_ensembles import (
    random_clifford,
    random_density_matrix,
    random_statevector,
)


def test_shannon_entropy_errors(backend):
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        shannon_entropy(prob, -2, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([[1.0], [0.0]])
        prob = backend.cast(prob, dtype=prob.dtype)
        shannon_entropy(prob, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([])
        prob = backend.cast(prob, dtype=prob.dtype)
        shannon_entropy(prob, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, -1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        shannon_entropy(prob, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.1, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        shannon_entropy(prob, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([0.5, 0.4999999])
        prob = backend.cast(prob, dtype=prob.dtype)
        shannon_entropy(prob, backend=backend)


@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_shannon_entropy(backend, base):
    prob_array = [1.0, 0.0]
    prob_array = backend.cast(prob_array, dtype=np.float64)
    result = shannon_entropy(prob_array, base, backend=backend)
    backend.assert_allclose(result, 0.0)

    if base == 2:
        prob_array = np.array([0.5, 0.5])
        prob_array = backend.cast(prob_array, dtype=prob_array.dtype)
        result = shannon_entropy(prob_array, base, backend=backend)
        backend.assert_allclose(result, 1.0)


@pytest.mark.parametrize("kind", [None, list])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_classical_relative_entropy(backend, base, kind):
    with pytest.raises(TypeError):
        prob = np.random.rand(1, 2)
        prob_q = np.random.rand(1, 5)
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        classical_relative_entropy(prob, prob_q, backend=backend)
    with pytest.raises(TypeError):
        prob = np.random.rand(1, 2)[0]
        prob_q = np.array([])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        classical_relative_entropy(prob, prob_q, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([-1, 2.0])
        prob_q = np.random.rand(1, 5)[0]
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        classical_relative_entropy(prob, prob_q, backend=backend)
    with pytest.raises(ValueError):
        prob = np.random.rand(1, 2)[0]
        prob_q = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        classical_relative_entropy(prob, prob_q, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob_q = np.random.rand(1, 2)[0]
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        classical_relative_entropy(prob, prob_q, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob_q = np.array([0.0, 1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        classical_relative_entropy(prob, prob_q, base=-2, backend=backend)

    prob_p = np.random.rand(10)
    prob_q = np.random.rand(10)
    prob_p /= np.sum(prob_p)
    prob_q /= np.sum(prob_q)

    target = np.sum(prob_p * np.log(prob_p) / np.log(base)) - np.sum(
        prob_p * np.log(prob_q) / np.log(base)
    )

    if kind is not None:
        prob_p, prob_q = kind(prob_p), kind(prob_q)
    else:
        prob_p = np.real(backend.cast(prob_p))
        prob_q = np.real(backend.cast(prob_q))

    divergence = classical_relative_entropy(prob_p, prob_q, base=base, backend=backend)

    backend.assert_allclose(divergence, target, atol=1e-5)


@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_classical_mutual_information(backend, base):
    prob_p = np.random.rand(10)
    prob_q = np.random.rand(10)
    prob_p /= np.sum(prob_p)
    prob_q /= np.sum(prob_q)

    joint_dist = np.kron(prob_p, prob_q)
    joint_dist /= np.sum(joint_dist)

    prob_p = backend.cast(prob_p, dtype=prob_p.dtype)
    prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
    joint_dist = backend.cast(joint_dist, dtype=joint_dist.dtype)

    backend.assert_allclose(
        classical_mutual_information(joint_dist, prob_p, prob_q, base, backend),
        0.0,
        atol=1e-10,
    )


@pytest.mark.parametrize("kind", [None, list])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("alpha", [0, 1, 2, 3, np.inf])
def test_classical_renyi_entropy(backend, alpha, base, kind):
    with pytest.raises(TypeError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        classical_renyi_entropy(prob, alpha="2", backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        classical_renyi_entropy(prob, alpha=-2, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        classical_renyi_entropy(prob, alpha, base="2", backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        classical_renyi_entropy(prob, alpha, base=-2, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([[1.0], [0.0]])
        prob = backend.cast(prob, dtype=prob.dtype)
        classical_renyi_entropy(prob, alpha, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([])
        prob = backend.cast(prob, dtype=prob.dtype)
        classical_renyi_entropy(prob, alpha, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, -1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        classical_renyi_entropy(prob, alpha, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.1, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        classical_renyi_entropy(prob, alpha, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([0.5, 0.4999999])
        prob = backend.cast(prob, dtype=prob.dtype)
        classical_renyi_entropy(prob, alpha, backend=backend)

    prob_dist = np.random.rand(10)
    prob_dist /= np.sum(prob_dist)

    if alpha == 0.0:
        target = np.log2(len(prob_dist)) / np.log2(base)
    elif alpha == 1:
        target = shannon_entropy(
            backend.cast(prob_dist, dtype=np.float64), base=base, backend=backend
        )
    elif alpha == 2:
        target = -1 * np.log2(np.sum(prob_dist**2)) / np.log2(base)
    elif alpha == np.inf:
        target = -1 * np.log2(max(prob_dist)) / np.log2(base)
    else:
        target = (1 / (1 - alpha)) * np.log2(np.sum(prob_dist**alpha)) / np.log2(base)

    if kind is not None:
        prob_dist = kind(prob_dist)
    else:
        prob_dist = np.real(backend.cast(prob_dist))

    renyi_ent = classical_renyi_entropy(prob_dist, alpha, base=base, backend=backend)

    backend.assert_allclose(renyi_ent, target, atol=1e-5)


@pytest.mark.parametrize("kind", [None, list])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("alpha", [0, 1 / 2, 1, 2, 3, np.inf])
def test_classical_relative_renyi_entropy(backend, alpha, base, kind):
    with pytest.raises(TypeError):
        prob = np.random.rand(1, 2)
        prob_q = np.random.rand(1, 5)
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        classical_relative_renyi_entropy(prob, prob_q, alpha, base, backend=backend)
    with pytest.raises(TypeError):
        prob = np.random.rand(1, 2)[0]
        prob_q = np.array([])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        classical_relative_renyi_entropy(prob, prob_q, alpha, base, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([-1, 2.0])
        prob_q = np.random.rand(1, 5)[0]
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        classical_relative_renyi_entropy(prob, prob_q, alpha, base, backend=backend)
    with pytest.raises(ValueError):
        prob = np.random.rand(1, 2)[0]
        prob_q = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        classical_relative_renyi_entropy(prob, prob_q, alpha, base, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob_q = np.random.rand(1, 2)[0]
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        classical_relative_renyi_entropy(prob, prob_q, alpha, base, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob_q = np.array([0.0, 1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        classical_relative_renyi_entropy(prob, prob_q, alpha, base=-2, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([1.0, 0.0])
        prob_q = np.array([0.0, 1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        classical_relative_renyi_entropy(
            prob, prob_q, alpha="1", base=base, backend=backend
        )
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob_q = np.array([0.0, 1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        classical_relative_renyi_entropy(
            prob, prob_q, alpha=-2, base=base, backend=backend
        )

    prob_p = np.random.rand(10)
    prob_q = np.random.rand(10)
    prob_p /= np.sum(prob_p)
    prob_q /= np.sum(prob_q)

    if alpha == 0.5:
        target = -2 * np.log2(np.sum(np.sqrt(prob_p * prob_q))) / np.log2(base)
    elif alpha == 1.0:
        target = classical_relative_entropy(
            np.real(backend.cast(prob_p)),
            np.real(backend.cast(prob_q)),
            base=base,
            backend=backend,
        )
    elif alpha == np.inf:
        target = np.log2(max(prob_p / prob_q)) / np.log2(base)
    else:
        target = (
            (1 / (alpha - 1))
            * np.log2(np.sum(prob_p**alpha * prob_q ** (1 - alpha)))
            / np.log2(base)
        )

    if kind is not None:
        prob_p, prob_q = kind(prob_p), kind(prob_q)
    else:
        prob_p = np.real(backend.cast(prob_p))
        prob_q = np.real(backend.cast(prob_q))

    divergence = classical_relative_renyi_entropy(
        prob_p, prob_q, alpha=alpha, base=base, backend=backend
    )

    backend.assert_allclose(divergence, target, atol=1e-5)


@pytest.mark.parametrize("kind", [None, list])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("alpha", [0, 1, 2, 3])
def test_classical_tsallis_entropy(backend, alpha, base, kind):
    with pytest.raises(TypeError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        classical_tsallis_entropy(prob, alpha="2", backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        classical_tsallis_entropy(prob, alpha=-2, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        classical_tsallis_entropy(prob, alpha, base="2", backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        classical_tsallis_entropy(prob, alpha, base=-2, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([[1.0], [0.0]])
        prob = backend.cast(prob, dtype=prob.dtype)
        classical_tsallis_entropy(prob, alpha, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([])
        prob = backend.cast(prob, dtype=prob.dtype)
        classical_tsallis_entropy(prob, alpha, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, -1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        classical_tsallis_entropy(prob, alpha, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.1, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        classical_tsallis_entropy(prob, alpha, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([0.5, 0.4999999])
        prob = backend.cast(prob, dtype=prob.dtype)
        classical_tsallis_entropy(prob, alpha, backend=backend)

    prob_dist = np.random.rand(10)
    prob_dist /= np.sum(prob_dist)

    if alpha == 1.0:
        target = shannon_entropy(
            np.real(backend.cast(prob_dist)), base=base, backend=backend
        )
    else:
        target = (1 / (1 - alpha)) * (np.sum(prob_dist**alpha) - 1)

    if kind is not None:
        prob_dist = kind(prob_dist)
    else:
        prob_dist = np.real(backend.cast(prob_dist))

    backend.assert_allclose(
        classical_tsallis_entropy(prob_dist, alpha=alpha, base=base, backend=backend),
        target,
        atol=1e-5,
    )


@pytest.mark.parametrize("kind", [None, list])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("alpha", [0, 1, 2, 3])
def test_classical_relative_tsallis_entropy(backend, alpha, base, kind):
    prob_dist_p = np.random.rand(10)
    prob_dist_p /= np.sum(prob_dist_p)

    prob_dist_q = np.random.rand(10)
    prob_dist_q /= np.sum(prob_dist_q)

    prob_dist_p = backend.cast(prob_dist_p, dtype=np.float64)
    prob_dist_q = backend.cast(prob_dist_q, dtype=np.float64)

    if alpha == 1.0:
        target = classical_relative_entropy(prob_dist_p, prob_dist_q, base, backend)
    else:
        target = ((prob_dist_p / prob_dist_q) ** (1 - alpha) - 1) / (1 - alpha)
        target = backend.sum(prob_dist_p**alpha * target)

    if kind is not None:
        prob_dist_p = kind(prob_dist_p)
        prob_dist_q = kind(prob_dist_q)

    value = classical_relative_tsallis_entropy(
        prob_dist_p, prob_dist_q, alpha, base, backend
    )

    backend.assert_allclose(value, target)


@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_von_neumann_entropy(backend, base):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        test = von_neumann_entropy(state, base=base, backend=backend)
    with pytest.raises(ValueError):
        state = np.array([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        test = von_neumann_entropy(state, base=0, backend=backend)

    state = np.array([1.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(
        von_neumann_entropy(state, backend=backend), 0.0, atol=PRECISION_TOL
    )

    state = backend.zero_state(2, density_matrix=True)

    nqubits = 2
    state = backend.maximally_mixed_state(nqubits)
    if base == 2:
        test = 2.0
    elif base == 10:
        test = 0.6020599913279624
    elif base == np.e:
        test = 1.3862943611198906
    else:
        test = 0.8613531161467861

    backend.assert_allclose(
        von_neumann_entropy(state, base, backend=backend),
        test,
    )


@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_relative_von_neumann_entropy(backend, base):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        target = random_density_matrix(2, pure=True, backend=backend)
        relative_von_neumann_entropy(state, target, base=base, backend=backend)
    with pytest.raises(TypeError):
        target = np.random.rand(2, 3)
        target = backend.cast(target, dtype=target.dtype)
        state = random_density_matrix(2, pure=True, backend=backend)
        relative_von_neumann_entropy(state, target, base=base, backend=backend)
    with pytest.raises(ValueError):
        state = np.array([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        target = np.array([0.0, 1.0])
        target = backend.cast(target, dtype=target.dtype)
        relative_von_neumann_entropy(state, target, base=0, backend=backend)

    nqubits = 2
    dims = 2**nqubits

    target = backend.maximally_mixed_state(nqubits)

    state = random_statevector(dims, backend=backend)
    state = backend.outer(state, backend.conj(state.T))

    rel_entropy = relative_von_neumann_entropy(
        state, target, base=base, backend=backend
    )

    backend.assert_allclose(rel_entropy, 2 / float(np.log2(base)), atol=1e-5)


@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_mutual_information(backend, base):
    with pytest.raises(ValueError):
        state = np.ones((3, 3))
        state = backend.cast(state, dtype=state.dtype)
        mutual_information(state, [0], backend)

    state_a = random_density_matrix(4, backend=backend)
    state_b = random_density_matrix(4, backend=backend)
    state = backend.kron(state_a, state_b)

    backend.assert_allclose(
        mutual_information(state, [0, 1], base, backend),
        0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("alpha", [0, 1, 2, 3, np.inf])
def test_renyi_entropy(backend, alpha, base):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        renyi_entropy(state, alpha=alpha, base=base, backend=backend)
    with pytest.raises(TypeError):
        state = random_statevector(4, backend=backend)
        renyi_entropy(state, alpha="2", base=base, backend=backend)
    with pytest.raises(ValueError):
        state = random_statevector(4, backend=backend)
        renyi_entropy(state, alpha=-1, base=base, backend=backend)
    with pytest.raises(ValueError):
        state = random_statevector(4, backend=backend)
        renyi_entropy(state, alpha=alpha, base=0, backend=backend)

    state = random_density_matrix(4, backend=backend)

    if alpha == 0.0:
        target = np.log2(len(state)) / np.log2(base)
    elif alpha == 1.0:
        target = von_neumann_entropy(state, base=base, backend=backend)
    elif alpha == np.inf:
        target = backend.matrix_norm(state, order=2)
        target = -1 * backend.log2(target) / np.log2(base)
    else:
        target = np.log2(
            np.trace(np.linalg.matrix_power(backend.to_numpy(state), alpha))
        )
        target = (1 / (1 - alpha)) * target / np.log2(base)

    backend.assert_allclose(
        renyi_entropy(state, alpha=alpha, base=base, backend=backend), target, atol=1e-5
    )

    # test pure state
    state = random_density_matrix(4, pure=True, backend=backend)
    backend.assert_allclose(
        renyi_entropy(state, alpha=alpha, base=base, backend=backend), 0.0, atol=1e-8
    )


@pytest.mark.parametrize(
    ["state_flag", "target_flag"], [[True, True], [False, True], [True, False]]
)
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("alpha", [0, 1, 2, 3, 5.4, np.inf])
def test_relative_renyi_entropy(backend, alpha, base, state_flag, target_flag):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        target = random_density_matrix(4, backend=backend)
        relative_renyi_entropy(state, target, alpha=alpha, base=base, backend=backend)
    with pytest.raises(TypeError):
        target = np.random.rand(2, 3)
        target = backend.cast(target, dtype=target.dtype)
        state = random_density_matrix(4, backend=backend)
        relative_renyi_entropy(state, target, alpha=alpha, base=base, backend=backend)
    with pytest.raises(TypeError):
        state = random_statevector(4, backend=backend)
        target = random_statevector(4, backend=backend)
        relative_renyi_entropy(state, target, alpha="2", base=base, backend=backend)
    with pytest.raises(ValueError):
        state = random_statevector(4, backend=backend)
        target = random_statevector(4, backend=backend)
        relative_renyi_entropy(state, target, alpha=-1, base=base, backend=backend)
    with pytest.raises(ValueError):
        state = random_statevector(4, backend=backend)
        target = random_statevector(4, backend=backend)
        relative_renyi_entropy(state, target, alpha=alpha, base=0, backend=backend)

    state = (
        random_statevector(4, backend=backend)
        if state_flag
        else random_density_matrix(4, backend=backend)
    )
    target = (
        random_statevector(4, backend=backend)
        if target_flag
        else random_density_matrix(4, backend=backend)
    )

    if state_flag and target_flag:
        backend.assert_allclose(
            relative_renyi_entropy(state, target, alpha, base, backend), 0.0, atol=1e-5
        )
    else:
        if target_flag and alpha > 1:
            with pytest.raises(NotImplementedError):
                relative_renyi_entropy(state, target, alpha, base, backend)
        else:
            if alpha == 1.0:
                log = relative_von_neumann_entropy(
                    state, target, base=base, backend=backend
                )
            elif alpha == np.inf:
                state_outer = (
                    backend.outer(state, backend.conj(state.T)) if state_flag else state
                )
                target_outer = (
                    backend.outer(target, backend.conj(target.T))
                    if target_flag
                    else target
                )
                new_state = matrix_power(state_outer, 0.5, backend=backend)
                new_target = matrix_power(target_outer, 0.5, backend=backend)

                log = backend.log2(backend.matrix_norm(new_state @ new_target, order=1))

                log = -2 * log / np.log2(base)
            else:
                if len(state.shape) == 1:
                    state = backend.outer(state, backend.conj(state))

                if len(target.shape) == 1:
                    target = backend.outer(target, backend.conj(target))

                log = matrix_power(state, alpha, backend=backend)
                log = log @ matrix_power(target, 1 - alpha, backend=backend)
                log = backend.log2(backend.trace(log))

                log = (1 / (alpha - 1)) * log / np.log2(base)

            backend.assert_allclose(
                relative_renyi_entropy(
                    state, target, alpha=alpha, base=base, backend=backend
                ),
                log,
                atol=1e-5,
            )

    # test pure states
    state = random_density_matrix(4, pure=True, backend=backend)
    target = random_density_matrix(4, pure=True, backend=backend)
    backend.assert_allclose(
        relative_renyi_entropy(state, target, alpha=alpha, base=base, backend=backend),
        0.0,
        atol=1e-7,
    )


@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("alpha", [0, 1, 2, 3, 5.4])
def test_tsallis_entropy(backend, alpha, base):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        tsallis_entropy(state, alpha=alpha, base=base, backend=backend)
    with pytest.raises(TypeError):
        state = random_statevector(4, backend=backend)
        tsallis_entropy(state, alpha="2", base=base, backend=backend)
    with pytest.raises(ValueError):
        state = random_statevector(4, backend=backend)
        tsallis_entropy(state, alpha=-1, base=base, backend=backend)
    with pytest.raises(ValueError):
        state = random_statevector(4, backend=backend)
        tsallis_entropy(state, alpha=alpha, base=0, backend=backend)

    state = random_density_matrix(4, backend=backend)

    if alpha == 1.0:
        target = von_neumann_entropy(state, base=base, backend=backend)
    else:
        target = (1 / (1 - alpha)) * (
            backend.trace(matrix_power(state, alpha, backend=backend)) - 1
        )

    backend.assert_allclose(
        tsallis_entropy(state, alpha=alpha, base=base, backend=backend),
        target,
        atol=1e-5,
    )

    # test pure state
    state = random_density_matrix(4, pure=True, backend=backend)
    backend.assert_allclose(
        tsallis_entropy(state, alpha=alpha, base=base, backend=backend), 0.0, atol=1e-5
    )


@pytest.mark.parametrize(
    ["state_flag", "target_flag"], [[True, True], [False, True], [True, False]]
)
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("alpha", [0, 0.5, 1, 1.9])
def test_relative_tsallis_entropy(backend, alpha, base, state_flag, target_flag):
    state = random_statevector(4, backend=backend)
    target = random_statevector(4, backend=backend)

    with pytest.raises(TypeError):
        relative_tsallis_entropy(state, target, alpha=1j, backend=backend)

    with pytest.raises(ValueError):
        relative_tsallis_entropy(state, target, alpha=3, backend=backend)

    with pytest.raises(ValueError):
        relative_tsallis_entropy(state, target, alpha=-1.0, backend=backend)

    state = (
        random_statevector(4, seed=10, backend=backend)
        if state_flag
        else random_density_matrix(4, seed=10, backend=backend)
    )
    target = (
        random_statevector(4, seed=11, backend=backend)
        if target_flag
        else random_density_matrix(4, seed=11, backend=backend)
    )

    value = relative_tsallis_entropy(state, target, alpha, base, backend)

    if alpha == 1.0:
        target_value = relative_von_neumann_entropy(
            state, target, base=base, backend=backend
        )
    else:
        if alpha < 1.0:
            alpha = 2 - alpha

        if state_flag:
            state = backend.outer(state, backend.conj(state.T))

        if target_flag:
            target = backend.outer(target, backend.conj(target.T))

        target_value = matrix_power(state, alpha, backend=backend)
        target_value = target_value @ matrix_power(target, 1 - alpha, backend=backend)
        target_value = (1 - backend.trace(target_value)) / (1 - alpha)

    backend.assert_allclose(value, target_value, atol=1e-10)


@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("bipartition", [[0], [1]])
def test_entanglement_entropy(backend, bipartition, base):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        test = entanglement_entropy(
            state,
            bipartition=bipartition,
            base=base,
            backend=backend,
        )
    with pytest.raises(ValueError):
        state = np.array([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        test = entanglement_entropy(
            state,
            bipartition=bipartition,
            base=0,
            backend=backend,
        )

    # Bell state
    state = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)
    state = backend.cast(state, dtype=state.dtype)

    entang_entrop = entanglement_entropy(
        state,
        bipartition=bipartition,
        base=base,
        backend=backend,
    )

    if base == 2:
        test = 1.0
    elif base == 10:
        test = 0.30102999566398125
    elif base == np.e:
        test = 0.6931471805599454
    else:
        test = 0.4306765580733931

    backend.assert_allclose(entang_entrop, test, atol=PRECISION_TOL)

    # Product state
    state = backend.kron(
        random_statevector(2, backend=backend), random_statevector(2, backend=backend)
    )

    entang_entrop = entanglement_entropy(
        state,
        bipartition=bipartition,
        base=base,
        backend=backend,
    )
    backend.assert_allclose(entang_entrop, 0.0, atol=PRECISION_TOL)


@pytest.mark.parametrize("base", [2, 10, math.e, 5])
@pytest.mark.parametrize("partition", [[0], [1]])
def test_conditional_entropy(backend, partition, base):
    with pytest.raises(ValueError):
        state = np.ones((3, 3))
        state = backend.cast(state, dtype=state.dtype)
        conditional_entropy(state, partition, base=base, backend=backend)
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        conditional_entropy(state, partition, base=base, backend=backend)

    # Bell state: conditional entropy is negative
    state = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)
    state = backend.cast(state, dtype=state.dtype)
    value = conditional_entropy(state, partition, base=base, backend=backend)
    backend.assert_allclose(value, -1 / np.log2(base), atol=PRECISION_TOL)

    # product pure state
    state = backend.zero_state(2)
    value = conditional_entropy(state, partition, base=base, backend=backend)
    backend.assert_allclose(value, 0.0, atol=PRECISION_TOL)

    # maximally mixed state
    state = backend.maximally_mixed_state(2)
    value = conditional_entropy(state, partition, base=base, backend=backend)
    backend.assert_allclose(value, 1 / np.log2(base), atol=PRECISION_TOL)


@pytest.mark.parametrize("base", [2, 10, math.e, 5])
def test_conditional_entropy_random_states(backend, base):
    # for product states, S(A|B) = S(A)
    state_a = random_density_matrix(4, backend=backend)
    state_b = random_density_matrix(4, backend=backend)
    state = backend.kron(state_a, state_b)
    value = conditional_entropy(state, [0, 1], base=base, backend=backend)
    target = von_neumann_entropy(state_a, base=base, backend=backend)
    backend.assert_allclose(value, target, atol=1e-6)

    # for pure states, S(A|B) = - S(A)
    state = random_statevector(8, backend=backend)
    value = conditional_entropy(state, [0], base=base, backend=backend)
    target = -entanglement_entropy(state, [1, 2], base=base, backend=backend)
    backend.assert_allclose(value, target, atol=1e-6)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_linear_entropy(backend, nqubits, normalize):
    with pytest.raises(TypeError):
        state = backend.zero_state(nqubits)
        linear_entropy(state, normalize="True", backend=backend)
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        linear_entropy(state, normalize=normalize, backend=backend)

    dims = 2**nqubits

    # pure states
    for density_matrix in (False, True):
        state = backend.zero_state(nqubits, density_matrix=density_matrix)
        value = linear_entropy(state, normalize=normalize, backend=backend)
        backend.assert_allclose(value, 0.0, atol=PRECISION_TOL)

    # maximally mixed state
    state = backend.maximally_mixed_state(nqubits)
    value = linear_entropy(state, normalize=normalize, backend=backend)
    target = 1.0 if normalize else 1 - 1 / dims
    backend.assert_allclose(value, target, atol=PRECISION_TOL)

    # random state
    state = random_density_matrix(dims, backend=backend)
    value = linear_entropy(state, normalize=normalize, backend=backend)
    state = backend.to_numpy(state)
    target = 1 - np.real(np.trace(state @ state))
    target = target * dims / (dims - 1) if normalize else target
    backend.assert_allclose(value, target, atol=PRECISION_TOL)


@pytest.mark.parametrize("base", [2, 10, math.e, 5])
@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_relative_entropy_of_coherence(backend, nqubits, base):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        relative_entropy_of_coherence(state, base=base, backend=backend)
    with pytest.raises(ValueError):
        state = backend.zero_state(nqubits)
        relative_entropy_of_coherence(state, base=0, backend=backend)

    dims = 2**nqubits

    for density_matrix in (False, True):
        # incoherent state
        state = backend.zero_state(nqubits, density_matrix=density_matrix)
        value = relative_entropy_of_coherence(state, base=base, backend=backend)
        backend.assert_allclose(value, 0.0, atol=PRECISION_TOL)

        # maximally coherent state
        state = backend.plus_state(nqubits, density_matrix=density_matrix)
        value = relative_entropy_of_coherence(state, base=base, backend=backend)
        backend.assert_allclose(value, nqubits / np.log2(base), atol=PRECISION_TOL)

    # maximally mixed state is diagonal
    state = backend.maximally_mixed_state(nqubits)
    value = relative_entropy_of_coherence(state, base=base, backend=backend)
    backend.assert_allclose(value, 0.0, atol=PRECISION_TOL)

    # random state: S(diag(rho)) - S(rho)
    state = random_density_matrix(dims, backend=backend)
    value = relative_entropy_of_coherence(state, base=base, backend=backend)
    state = backend.to_numpy(state)
    eigenvalues = np.linalg.eigvalsh(state)
    eigenvalues = eigenvalues[eigenvalues > 0.0]
    diagonal = np.real(np.diag(state))
    diagonal = diagonal[diagonal > 0.0]
    target = np.sum(eigenvalues * np.log2(eigenvalues))
    target -= np.sum(diagonal * np.log2(diagonal))
    backend.assert_allclose(value, target / np.log2(base), atol=1e-6)

    # single-qubit state (I + 0.6 X) / 2, for which diag(rho) = I / 2
    # and the eigenvalues are 0.8 and 0.2
    state = np.array([[0.5, 0.3], [0.3, 0.5]])
    state = backend.cast(state, dtype=state.dtype)
    value = relative_entropy_of_coherence(state, base=base, backend=backend)
    target = 1 + 0.8 * np.log2(0.8) + 0.2 * np.log2(0.2)
    backend.assert_allclose(value, target / np.log2(base), atol=PRECISION_TOL)


def test_stabilizer_renyi_entropy_errors(backend):
    state = backend.zero_state(2)

    with pytest.raises(TypeError):
        wrong_state = np.random.rand(2, 3)
        wrong_state = backend.cast(wrong_state, dtype=wrong_state.dtype)
        stabilizer_renyi_entropy(wrong_state, 2, backend=backend)
    with pytest.raises(TypeError):
        wrong_state = np.array([])
        wrong_state = backend.cast(wrong_state, dtype=wrong_state.dtype)
        stabilizer_renyi_entropy(wrong_state, 2, backend=backend)
    with pytest.raises(ValueError):
        wrong_state = np.ones(3) / math.sqrt(3)
        wrong_state = backend.cast(wrong_state, dtype=wrong_state.dtype)
        stabilizer_renyi_entropy(wrong_state, 2, backend=backend)
    with pytest.raises(TypeError):
        stabilizer_renyi_entropy(state, "2", backend=backend)
    for alpha in (0, -1, math.inf, math.nan):
        with pytest.raises(ValueError):
            stabilizer_renyi_entropy(state, alpha, backend=backend)
    with pytest.raises(ValueError):
        stabilizer_renyi_entropy(state, 2, base=0, backend=backend)
    with pytest.raises(TypeError):
        stabilizer_renyi_entropy(state, 2, check_purity="True", backend=backend)
    with pytest.raises(NotImplementedError):
        mixed_state = backend.maximally_mixed_state(2)
        stabilizer_renyi_entropy(mixed_state, 2, backend=backend)

    # purity is not verified if ``check_purity=False``
    mixed_state = backend.maximally_mixed_state(2)
    value = stabilizer_renyi_entropy(
        mixed_state, 2, check_purity=False, backend=backend
    )
    assert math.isfinite(float(value))


@pytest.mark.parametrize("base", [2, 10, math.e, 5])
@pytest.mark.parametrize("alpha", [0.5, 1, 2, 3, 4.5])
def test_stabilizer_renyi_entropy_single_qubit_magic_state(backend, alpha, base):
    # (|0> + exp(i * pi / 4) |1>) / sqrt(2) has Pauli expectation values
    # <I> = 1, <X> = <Y> = 1 / sqrt(2), and <Z> = 0.
    state = np.array([1.0, np.exp(1j * math.pi / 4)]) / math.sqrt(2)
    state = backend.cast(state, dtype=state.dtype)

    if alpha == 1:
        target = 1 / 2
    else:
        target = math.log2((1 + 2 ** (1 - alpha)) / 2) / (1 - alpha)

    for kind in (state, backend.outer(state, backend.conj(state))):
        value = stabilizer_renyi_entropy(kind, alpha, base=base, backend=backend)
        backend.assert_allclose(value, target / math.log2(base), atol=1e-10)


@pytest.mark.parametrize("alpha", [0.5, 1, 2, 3])
@pytest.mark.parametrize("nqubits", [1, 2, 3, 4])
def test_stabilizer_renyi_entropy_stabilizer_states(backend, nqubits, alpha):
    dims = 2**nqubits

    ghz = np.zeros(dims)
    ghz[0] = ghz[-1] = 1 / math.sqrt(2)
    ghz = backend.cast(ghz, dtype=ghz.dtype)

    clifford = random_clifford(nqubits, seed=10, backend=backend)
    random_stabilizer = backend.execute_circuit(clifford).state()

    states = [
        backend.zero_state(nqubits),
        backend.plus_state(nqubits),
        ghz,
        random_stabilizer,
    ]

    for state in states:
        for kind in (state, backend.outer(state, backend.conj(state))):
            value = stabilizer_renyi_entropy(kind, alpha, backend=backend)
            backend.assert_allclose(value, 0.0, atol=PRECISION_TOL)


@pytest.mark.parametrize("base", [2, 10, math.e])
@pytest.mark.parametrize("alpha", [0.3, 0.5, 1, 2, 3, 4.5])
@pytest.mark.parametrize("nqubits", [1, 2, 3, 4])
def test_stabilizer_renyi_entropy_random_states(backend, nqubits, alpha, base):
    dims = 2**nqubits

    state = random_statevector(dims, backend=backend)

    # brute-force calculation over all Pauli strings
    paulis = backend.to_numpy(pauli_basis(nqubits, backend=backend))
    vector = backend.to_numpy(state)
    expectations = np.real(np.einsum("a,iab,b->i", np.conj(vector), paulis, vector))
    probabilities = expectations**2

    if alpha == 1:
        logs = np.log2(np.where(probabilities > 0.0, probabilities, 1.0))
        target = -np.sum(probabilities * logs) / dims
    else:
        target = np.log2(np.sum(probabilities**alpha) / dims) / (1 - alpha)

    for kind in (state, backend.outer(state, backend.conj(state))):
        value = stabilizer_renyi_entropy(kind, alpha, base=base, backend=backend)
        backend.assert_allclose(value, target / math.log2(base), atol=1e-8)


@pytest.mark.parametrize("alpha", [0.5, 1, 2, 3])
def test_stabilizer_renyi_entropy_properties(backend, alpha):
    state_a = random_statevector(2, backend=backend)
    state_b = random_statevector(4, backend=backend)
    entropy_a = stabilizer_renyi_entropy(state_a, alpha, backend=backend)
    entropy_b = stabilizer_renyi_entropy(state_b, alpha, backend=backend)

    # additivity for product states
    entropy = stabilizer_renyi_entropy(
        backend.kron(state_a, state_b), alpha, backend=backend
    )
    backend.assert_allclose(entropy, entropy_a + entropy_b, atol=1e-8)

    state = random_statevector(8, backend=backend)
    entropy = stabilizer_renyi_entropy(state, alpha, backend=backend)

    # invariance under Clifford operations
    clifford = random_clifford(3, seed=2, backend=backend)
    unitary = clifford.unitary(backend=backend)
    transformed = stabilizer_renyi_entropy(unitary @ state, alpha, backend=backend)
    backend.assert_allclose(transformed, entropy, atol=1e-8)

    # invariance under global phase
    phased = stabilizer_renyi_entropy(state * np.exp(1j * 0.3), alpha, backend=backend)
    backend.assert_allclose(phased, entropy, atol=1e-8)

    # upper bound for pure states
    assert entropy <= math.log2(8) + PRECISION_TOL
