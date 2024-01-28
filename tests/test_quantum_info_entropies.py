import numpy as np
import pytest

from qibo.config import PRECISION_TOL
from qibo.quantum_info.entropies import entanglement_entropy, entropy
from qibo.quantum_info.random_ensembles import random_statevector, random_unitary


@pytest.mark.parametrize("check_hermitian", [False, True])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_entropy(backend, base, check_hermitian):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        test = entropy(
            state, base=base, check_hermitian=check_hermitian, backend=backend
        )
    with pytest.raises(ValueError):
        state = np.array([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        test = entropy(state, base=0, check_hermitian=check_hermitian, backend=backend)
    with pytest.raises(TypeError):
        state = np.array([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        test = entropy(state, base=base, check_hermitian="False", backend=backend)

    if backend.__class__.__name__ in ["CupyBackend", "CuQuantumBackend"]:
        with pytest.raises(NotImplementedError):
            state = random_unitary(4, backend=backend)
            test = entropy(state, base=base, check_hermitian=True, backend=backend)
    else:
        state = random_unitary(4, backend=backend)
        test = entropy(state, base=base, check_hermitian=True, backend=backend)

    state = np.array([1.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(entropy(state, backend=backend), 0.0, atol=PRECISION_TOL)

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
        entropy(state, base, check_hermitian=check_hermitian, backend=backend), test
    )


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
    if backend.__class__.__name__ == "CupyBackend":
        with pytest.raises(NotImplementedError):
            state = random_unitary(4, backend=backend)
            test = entanglement_entropy(
                state,
                bipartition=bipartition,
                base=base,
                check_hermitian=True,
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
    state = np.kron(
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
