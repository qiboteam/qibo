import numpy as np
import pytest

from qibo.config import PRECISION_TOL
from qibo.quantum_info.entropies import (
    classical_mutual_information,
    classical_relative_entropy,
    classical_relative_renyi_entropy,
    classical_relative_tsallis_entropy,
    classical_renyi_entropy,
    classical_tsallis_entropy,
    entanglement_entropy,
    mutual_information,
    relative_renyi_entropy,
    relative_tsallis_entropy,
    relative_von_neumann_entropy,
    renyi_entropy,
    shannon_entropy,
    tsallis_entropy,
    von_neumann_entropy,
)
from qibo.quantum_info.linalg_operations import matrix_power
from qibo.quantum_info.random_ensembles import (
    random_density_matrix,
    random_statevector,
    random_unitary,
)


def test_shannon_entropy_errors(backend):
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = shannon_entropy(prob, -2, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([[1.0], [0.0]])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = shannon_entropy(prob, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = shannon_entropy(prob, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, -1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = shannon_entropy(prob, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.1, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = shannon_entropy(prob, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([0.5, 0.4999999])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = shannon_entropy(prob, backend=backend)


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
        test = classical_relative_entropy(prob, prob_q, backend=backend)
    with pytest.raises(TypeError):
        prob = np.random.rand(1, 2)[0]
        prob_q = np.array([])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_entropy(prob, prob_q, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([-1, 2.0])
        prob_q = np.random.rand(1, 5)[0]
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_entropy(prob, prob_q, backend=backend)
    with pytest.raises(ValueError):
        prob = np.random.rand(1, 2)[0]
        prob_q = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_entropy(prob, prob_q, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob_q = np.random.rand(1, 2)[0]
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_entropy(prob, prob_q, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob_q = np.array([0.0, 1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_entropy(prob, prob_q, base=-2, backend=backend)

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
        test = classical_renyi_entropy(prob, alpha="2", backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_renyi_entropy(prob, alpha=-2, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_renyi_entropy(prob, alpha, base="2", backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_renyi_entropy(prob, alpha, base=-2, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([[1.0], [0.0]])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_renyi_entropy(prob, alpha, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_renyi_entropy(prob, alpha, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, -1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_renyi_entropy(prob, alpha, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.1, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_renyi_entropy(prob, alpha, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([0.5, 0.4999999])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_renyi_entropy(prob, alpha, backend=backend)

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
        test = classical_relative_renyi_entropy(
            prob, prob_q, alpha, base, backend=backend
        )
    with pytest.raises(TypeError):
        prob = np.random.rand(1, 2)[0]
        prob_q = np.array([])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_renyi_entropy(
            prob, prob_q, alpha, base, backend=backend
        )
    with pytest.raises(ValueError):
        prob = np.array([-1, 2.0])
        prob_q = np.random.rand(1, 5)[0]
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_renyi_entropy(
            prob, prob_q, alpha, base, backend=backend
        )
    with pytest.raises(ValueError):
        prob = np.random.rand(1, 2)[0]
        prob_q = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_renyi_entropy(
            prob, prob_q, alpha, base, backend=backend
        )
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob_q = np.random.rand(1, 2)[0]
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_renyi_entropy(
            prob, prob_q, alpha, base, backend=backend
        )
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob_q = np.array([0.0, 1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_renyi_entropy(
            prob, prob_q, alpha, base=-2, backend=backend
        )
    with pytest.raises(TypeError):
        prob = np.array([1.0, 0.0])
        prob_q = np.array([0.0, 1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_renyi_entropy(
            prob, prob_q, alpha="1", base=base, backend=backend
        )
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob_q = np.array([0.0, 1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_renyi_entropy(
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
        test = classical_tsallis_entropy(prob, alpha="2", backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_tsallis_entropy(prob, alpha=-2, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_tsallis_entropy(prob, alpha, base="2", backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_tsallis_entropy(prob, alpha, base=-2, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([[1.0], [0.0]])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_tsallis_entropy(prob, alpha, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_tsallis_entropy(prob, alpha, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, -1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_tsallis_entropy(prob, alpha, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.1, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_tsallis_entropy(prob, alpha, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([0.5, 0.4999999])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_tsallis_entropy(prob, alpha, backend=backend)

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
        target = backend.np.sum(prob_dist_p**alpha * target)

    if kind is not None:
        prob_dist_p = kind(prob_dist_p)
        prob_dist_q = kind(prob_dist_q)

    value = classical_relative_tsallis_entropy(
        prob_dist_p, prob_dist_q, alpha, base, backend
    )

    backend.assert_allclose(value, target)


@pytest.mark.parametrize("check_hermitian", [False, True])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_von_neumann_entropy(backend, base, check_hermitian):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        test = von_neumann_entropy(
            state, base=base, check_hermitian=check_hermitian, backend=backend
        )
    with pytest.raises(ValueError):
        state = np.array([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        test = von_neumann_entropy(
            state, base=0, check_hermitian=check_hermitian, backend=backend
        )
    with pytest.raises(TypeError):
        state = np.array([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        test = von_neumann_entropy(
            state, base=base, check_hermitian="False", backend=backend
        )

    state = random_unitary(4, backend=backend)
    test = von_neumann_entropy(state, base=base, check_hermitian=True, backend=backend)

    state = np.array([1.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(
        von_neumann_entropy(state, backend=backend), 0.0, atol=PRECISION_TOL
    )

    state = np.array([1.0, 0.0, 0.0, 0.0])
    state = np.outer(state, state)
    state = backend.cast(state, dtype=state.dtype)

    nqubits = 2
    state = backend.identity_density_matrix(nqubits)
    if base == 2:
        test = 2.0
    elif base == 10:
        test = 0.6020599913279624
    elif base == np.e:
        test = 1.3862943611198906
    else:
        test = 0.8613531161467861

    backend.assert_allclose(
        von_neumann_entropy(
            state, base, check_hermitian=check_hermitian, backend=backend
        ),
        test,
    )


@pytest.mark.parametrize("statevector", [False, True])
@pytest.mark.parametrize("check_hermitian", [False, True])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_relative_von_neumann_entropy(backend, base, check_hermitian, statevector):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        target = random_density_matrix(2, pure=True, backend=backend)
        test = relative_von_neumann_entropy(
            state, target, base=base, check_hermitian=check_hermitian, backend=backend
        )
    with pytest.raises(TypeError):
        target = np.random.rand(2, 3)
        target = backend.cast(target, dtype=target.dtype)
        state = random_density_matrix(2, pure=True, backend=backend)
        test = relative_von_neumann_entropy(
            state, target, base=base, check_hermitian=check_hermitian, backend=backend
        )
    with pytest.raises(ValueError):
        state = np.array([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        target = np.array([0.0, 1.0])
        target = backend.cast(target, dtype=target.dtype)
        test = relative_von_neumann_entropy(
            state, target, base=0, check_hermitian=check_hermitian, backend=backend
        )
    with pytest.raises(TypeError):
        state = np.array([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        target = np.array([0.0, 1.0])
        target = backend.cast(target, dtype=target.dtype)
        test = relative_von_neumann_entropy(
            state, target, base=base, check_hermitian="False", backend=backend
        )

    nqubits = 2
    dims = 2**nqubits

    state = (
        random_statevector(dims, backend=backend)
        if statevector
        else random_density_matrix(dims, backend=backend)
    )
    target = backend.identity_density_matrix(nqubits, normalize=True)

    entropy = relative_von_neumann_entropy(
        state, target, base=base, check_hermitian=check_hermitian, backend=backend
    )

    if statevector:
        state = backend.np.outer(state, backend.np.conj(state.T))

    entropy_target = von_neumann_entropy(
        state, base=base, check_hermitian=check_hermitian, backend=backend
    )

    backend.assert_allclose(
        entropy, np.log(dims) / np.log(base) - entropy_target, atol=1e-5
    )

    state = random_density_matrix(2, seed=8, pure=False, backend=backend)
    target = backend.cast([0.0, 1.0], dtype=np.float64)

    backend.assert_allclose(
        relative_von_neumann_entropy(state, target, backend=backend),
        -0.21227801,
        atol=1e-8,
    )


@pytest.mark.parametrize("check_hermitian", [False, True])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_mutual_information(backend, base, check_hermitian):
    with pytest.raises(ValueError):
        state = np.ones((3, 3))
        state = backend.cast(state, dtype=state.dtype)
        test = mutual_information(state, [0], backend)

    state_a = random_density_matrix(4, backend=backend)
    state_b = random_density_matrix(4, backend=backend)
    state = backend.np.kron(state_a, state_b)

    backend.assert_allclose(
        mutual_information(state, [0, 1], base, check_hermitian, backend),
        0.0,
        atol=1e-10,
    )


@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("alpha", [0, 1, 2, 3, np.inf])
def test_renyi_entropy(backend, alpha, base):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        test = renyi_entropy(state, alpha=alpha, base=base, backend=backend)
    with pytest.raises(TypeError):
        state = random_statevector(4, backend=backend)
        test = renyi_entropy(state, alpha="2", base=base, backend=backend)
    with pytest.raises(ValueError):
        state = random_statevector(4, backend=backend)
        test = renyi_entropy(state, alpha=-1, base=base, backend=backend)
    with pytest.raises(ValueError):
        state = random_statevector(4, backend=backend)
        test = renyi_entropy(state, alpha=alpha, base=0, backend=backend)

    state = random_density_matrix(4, backend=backend)

    if alpha == 0.0:
        target = np.log2(len(state)) / np.log2(base)
    elif alpha == 1.0:
        target = von_neumann_entropy(state, base=base, backend=backend)
    elif alpha == np.inf:
        target = backend.calculate_matrix_norm(state, order=2)
        target = -1 * backend.np.log2(target) / np.log2(base)
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
        test = relative_renyi_entropy(
            state, target, alpha=alpha, base=base, backend=backend
        )
    with pytest.raises(TypeError):
        target = np.random.rand(2, 3)
        target = backend.cast(target, dtype=target.dtype)
        state = random_density_matrix(4, backend=backend)
        test = relative_renyi_entropy(
            state, target, alpha=alpha, base=base, backend=backend
        )
    with pytest.raises(TypeError):
        state = random_statevector(4, backend=backend)
        target = random_statevector(4, backend=backend)
        test = relative_renyi_entropy(
            state, target, alpha="2", base=base, backend=backend
        )
    with pytest.raises(ValueError):
        state = random_statevector(4, backend=backend)
        target = random_statevector(4, backend=backend)
        test = relative_renyi_entropy(
            state, target, alpha=-1, base=base, backend=backend
        )
    with pytest.raises(ValueError):
        state = random_statevector(4, backend=backend)
        target = random_statevector(4, backend=backend)
        test = relative_renyi_entropy(
            state, target, alpha=alpha, base=0, backend=backend
        )

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
                    backend.np.outer(state, backend.np.conj(state.T))
                    if state_flag
                    else state
                )
                target_outer = (
                    backend.np.outer(target, backend.np.conj(target.T))
                    if target_flag
                    else target
                )
                new_state = matrix_power(state_outer, 0.5, backend=backend)
                new_target = matrix_power(target_outer, 0.5, backend=backend)

                log = backend.np.log2(
                    backend.calculate_matrix_norm(new_state @ new_target, order=1)
                )

                log = -2 * log / np.log2(base)
            else:
                if len(state.shape) == 1:
                    state = backend.np.outer(state, backend.np.conj(state))

                if len(target.shape) == 1:
                    target = backend.np.outer(target, backend.np.conj(target))

                log = matrix_power(state, alpha, backend=backend)
                log = log @ matrix_power(target, 1 - alpha, backend=backend)
                log = backend.np.log2(backend.np.trace(log))

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
        atol=1e-8,
    )


@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("alpha", [0, 1, 2, 3, 5.4])
def test_tsallis_entropy(backend, alpha, base):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        test = tsallis_entropy(state, alpha=alpha, base=base, backend=backend)
    with pytest.raises(TypeError):
        state = random_statevector(4, backend=backend)
        test = tsallis_entropy(state, alpha="2", base=base, backend=backend)
    with pytest.raises(ValueError):
        state = random_statevector(4, backend=backend)
        test = tsallis_entropy(state, alpha=-1, base=base, backend=backend)
    with pytest.raises(ValueError):
        state = random_statevector(4, backend=backend)
        test = tsallis_entropy(state, alpha=alpha, base=0, backend=backend)

    state = random_density_matrix(4, backend=backend)

    if alpha == 1.0:
        target = von_neumann_entropy(state, base=base, backend=backend)
    else:
        target = (1 / (1 - alpha)) * (
            backend.np.trace(matrix_power(state, alpha, backend=backend)) - 1
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
@pytest.mark.parametrize("check_hermitian", [False, True])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("alpha", [0, 0.5, 1, 1.9])
def test_relative_tsallis_entropy(
    backend, alpha, base, check_hermitian, state_flag, target_flag
):
    state = random_statevector(4, backend=backend)
    target = random_statevector(4, backend=backend)

    with pytest.raises(TypeError):
        test = relative_tsallis_entropy(state, target, alpha=1j, backend=backend)

    with pytest.raises(ValueError):
        test = relative_tsallis_entropy(state, target, alpha=3, backend=backend)

    with pytest.raises(ValueError):
        test = relative_tsallis_entropy(state, target, alpha=-1.0, backend=backend)

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

    value = relative_tsallis_entropy(
        state, target, alpha, base, check_hermitian, backend
    )

    if alpha == 1.0:
        target_value = relative_von_neumann_entropy(
            state, target, base=base, check_hermitian=check_hermitian, backend=backend
        )
    else:
        if alpha < 1.0:
            alpha = 2 - alpha

        if state_flag:
            state = backend.np.outer(state, backend.np.conj(state.T))

        if target_flag:
            target = backend.np.outer(target, backend.np.conj(target.T))

        target_value = matrix_power(state, alpha, backend=backend)
        target_value = target_value @ matrix_power(target, 1 - alpha, backend=backend)
        target_value = (1 - backend.np.trace(target_value)) / (1 - alpha)

    backend.assert_allclose(value, target_value, atol=1e-10)


@pytest.mark.parametrize("check_hermitian", [False, True])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("bipartition", [[0], [1]])
def test_entanglement_entropy(backend, bipartition, base, check_hermitian):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        test = entanglement_entropy(
            state,
            bipartition=bipartition,
            base=base,
            check_hermitian=check_hermitian,
            backend=backend,
        )
    with pytest.raises(ValueError):
        state = np.array([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        test = entanglement_entropy(
            state,
            bipartition=bipartition,
            base=0,
            check_hermitian=check_hermitian,
            backend=backend,
        )

    # Bell state
    state = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)
    state = backend.cast(state, dtype=state.dtype)

    entang_entrop = entanglement_entropy(
        state,
        bipartition=bipartition,
        base=base,
        check_hermitian=check_hermitian,
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
    state = backend.np.kron(
        random_statevector(2, backend=backend), random_statevector(2, backend=backend)
    )

    entang_entrop = entanglement_entropy(
        state,
        bipartition=bipartition,
        base=base,
        check_hermitian=check_hermitian,
        backend=backend,
    )
    backend.assert_allclose(entang_entrop, 0.0, atol=PRECISION_TOL)
