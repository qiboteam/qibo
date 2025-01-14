import numpy as np
import pytest

from qibo import Circuit, gates, hamiltonians
from qibo.measurements import MeasurementResult
from qibo.result import QuantumState, load_result
from qibo.symbols import I, Z


def test_measurement_result_repr():
    result = MeasurementResult(gates.M(0).target_qubits)
    assert str(result) == "MeasurementResult(qubits=(0,), nshots=None)"


def test_measurement_result_error():
    result = MeasurementResult(gates.M(0).qubits)
    with pytest.raises(RuntimeError):
        samples = result.samples()


@pytest.mark.parametrize("target", range(5))
@pytest.mark.parametrize("density_matrix", [False, True])
def test_state_representation(backend, target, density_matrix):
    c = Circuit(5, density_matrix=density_matrix)
    c.add(gates.H(target))
    result = backend.execute_circuit(c)
    bstring = target * "0" + "1" + (4 - target) * "0"
    if density_matrix:
        target_str = 3 * [
            f"(0.5+0j)|00000><00000| + (0.5+0j)|00000><{bstring}| + (0.5+0j)|{bstring}><00000|"
            + f" + (0.5+0j)|{bstring}><{bstring}|"
        ]
    else:
        target_str = [
            f"(0.70711+0j)|00000> + (0.70711+0j)|{bstring}>",
            f"(0.7+0j)|00000> + (0.7+0j)|{bstring}>",
            f"(0.71+0j)|00000> + (0.71+0j)|{bstring}>",
        ]
    assert str(result) == target_str[0]
    assert result.symbolic(decimals=5) == target_str[0]
    assert result.symbolic(decimals=1) == target_str[1]
    assert result.symbolic(decimals=2) == target_str[2]


@pytest.mark.parametrize("density_matrix", [False, True])
def test_state_representation_max_terms(backend, density_matrix):
    c = Circuit(5, density_matrix=density_matrix)
    c.add(gates.H(i) for i in range(5))
    result = backend.execute_circuit(c)
    if density_matrix:
        assert (
            result.symbolic(max_terms=3)
            == "(0.03125+0j)|00000><00000| + (0.03125+0j)|00000><00001| + "
            + "(0.03125+0j)|00000><00010| + ..."
        )
        assert (
            result.symbolic(max_terms=5)
            == "(0.03125+0j)|00000><00000| + (0.03125+0j)|00000><00001| + "
            + "(0.03125+0j)|00000><00010| + (0.03125+0j)|00000><00011| + "
            + "(0.03125+0j)|00000><00100| + ..."
        )
    else:
        assert (
            result.symbolic(max_terms=3)
            == "(0.17678+0j)|00000> + (0.17678+0j)|00001> + (0.17678+0j)|00010> + ..."
        )
        assert (
            result.symbolic(max_terms=5)
            == "(0.17678+0j)|00000> + (0.17678+0j)|00001> + (0.17678+0j)|00010> + "
            + "(0.17678+0j)|00011> + (0.17678+0j)|00100> + ..."
        )


@pytest.mark.parametrize("density_matrix", [False, True])
def test_state_probabilities(backend, density_matrix):
    c = Circuit(4, density_matrix=density_matrix)
    c.add(gates.H(i) for i in range(4))
    result = backend.execute_circuit(c)
    # with pytest.raises(ValueError):
    #    final_probabilities = result.probabilities()

    c = Circuit(4, density_matrix=density_matrix)
    c.add(gates.H(i) for i in range(4))
    c.add(gates.M(*range(4)))
    result = backend.execute_circuit(c)
    final_probabilities = result.probabilities()
    target_probabilities = np.ones(16) / 16
    backend.assert_allclose(final_probabilities, target_probabilities)


def test_expectation_from_samples(backend):
    # fix seed to use the same samples in every execution
    np.random.seed(123)
    obs0 = 2 * Z(0, backend=backend) * Z(1, backend=backend) + Z(
        0, backend=backend
    ) * Z(2, backend=backend)
    obs1 = 2 * Z(0, backend=backend) * Z(1, backend=backend) + Z(
        0, backend=backend
    ) * Z(2, backend=backend) * I(3, backend=backend)
    h_sym = hamiltonians.SymbolicHamiltonian(obs0, backend=backend)
    h_dense = hamiltonians.Hamiltonian(3, h_sym.matrix, backend=backend)
    h1 = hamiltonians.SymbolicHamiltonian(obs1, backend=backend)
    c = Circuit(4)
    c.add(gates.RX(0, np.random.rand()))
    c.add(gates.RX(1, np.random.rand()))
    c.add(gates.RX(2, np.random.rand()))
    c.add(gates.RX(3, np.random.rand()))
    c.add(gates.M(0, 1, 2))
    nshots = 10**5
    result = backend.execute_circuit(c, nshots=nshots)
    expval_sym = result.expectation_from_samples(h_sym)
    expval_dense = result.expectation_from_samples(h_dense)
    expval = h1.expectation(result.state())
    backend.assert_allclose(expval_sym, expval_dense)
    backend.assert_allclose(expval_sym, expval, atol=10 / np.sqrt(nshots))


def test_state_numpy(backend):
    c = Circuit(1)
    result = backend.execute_circuit(c)
    assert isinstance(result.state(numpy=True), np.ndarray)


@pytest.mark.parametrize("agnostic_load", [False, True])
def test_state_dump_load(backend, agnostic_load):
    from os import remove

    c = Circuit(1)
    c.add(gates.H(0))
    state = backend.execute_circuit(c)
    state.dump("tmp.npy")
    if agnostic_load:
        loaded_state = load_result("tmp.npy")
    else:
        loaded_state = QuantumState.load("tmp.npy")
    assert str(state) == str(loaded_state)
    remove("tmp.npy")
