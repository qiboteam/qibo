"""
Testing tensorflow callbacks.
"""
import numpy as np
from qibo.models import Circuit
from qibo import gates, callbacks

# Absolute testing tolerance for the cases of zero entanglement entropy
_atol = 1e-12


def test_entropy_product_state():
    """Check that the |++> state has zero entropy."""
    entropy = callbacks.EntanglementEntropy([0])
    state = np.ones(4).reshape((2, 2)) / 2.0

    result = entropy(state).numpy()
    np.testing.assert_allclose(result, 0, atol=_atol)


def test_entropy_singlet_state():
    """Check that the singlet state has maximum entropy."""
    entropy = callbacks.EntanglementEntropy([0])
    state = np.zeros(4)
    state[0], state[-1] = 1, 1
    state = state.reshape((2, 2)) / np.sqrt(2)

    result = entropy(state).numpy()
    np.testing.assert_allclose(result, np.log(2))


def test_entropy_in_circuit():
    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.CNOT(0, 1))
    state = c(callback=entropy)

    result = entropy.results.numpy()
    print(entropy._results)
    np.testing.assert_allclose(result, [0, 0, np.log(2)], atol=_atol)


def test_entropy_in_compiled_circuit():
    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.CNOT(0, 1))
    c.compile(callback=entropy)
    state = c()

    result = entropy.results.numpy()
    print(entropy._results)
    np.testing.assert_allclose(result, [0, 0, np.log(2)], atol=_atol)
