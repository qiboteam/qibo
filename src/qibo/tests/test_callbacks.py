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
    state = np.ones(4) / 2.0

    result = entropy(state).numpy()
    np.testing.assert_allclose(result, 0, atol=_atol)


def test_entropy_singlet_state():
    """Check that the singlet state has maximum entropy."""
    entropy = callbacks.EntanglementEntropy([0])
    state = np.zeros(4)
    state[0], state[-1] = 1, 1
    state = state / np.sqrt(2)

    result = entropy(state).numpy()
    np.testing.assert_allclose(result, 1.0)


def test_entropy_random_state():
    """Check that entropy calculation agrees with numpy."""
    # Generate a random positive and hermitian density matrix
    rho = np.random.random((8, 8)) + 1j * np.random.random((8, 8))
    rho = rho + rho.conj().T
    _, u = np.linalg.eigh(rho)
    s = 5 * np.random.random(8)
    s = s / s.sum()
    rho = u.dot(np.diag(s)).dot(u.conj().T)

    result = callbacks.EntanglementEntropy._entropy(rho).numpy()
    target = - (s * np.log2(s)).sum()
    np.testing.assert_allclose(result, target)


def test_entropy_numerical():
    """Check that entropy calculation does not fail for tiny eigenvalues."""
    import tensorflow as tf
    from qibo.config import DTYPECPX
    eigvals = np.array([-1e-10, -1e-15, -2e-17, -1e-18, -5e-60, 1e-48, 4e-32,
                        5e-14, 1e-14, 9.9e-13, 9e-13, 5e-13, 1e-13, 1e-12,
                        1e-11, 1e-10, 1e-9, 1e-7, 1, 4, 10])
    rho = tf.convert_to_tensor(np.diag(eigvals), dtype=DTYPECPX)
    result = callbacks.EntanglementEntropy._entropy(rho).numpy()

    mask = eigvals > 0
    target = - (eigvals[mask] * np.log2(eigvals[mask])).sum()

    np.testing.assert_allclose(result, target)


def test_entropy_in_circuit():
    """Check that entropy calculation works in circuit."""
    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.CNOT(0, 1))
    state = c(callback=entropy)

    target = [0, 0, 1.0]
    np.testing.assert_allclose(entropy[0].numpy(), target, atol=_atol)


def test_entropy_in_compiled_circuit():
    """Check that entropy calculation works when circuit is compiled."""
    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.CNOT(0, 1))
    c.compile(callback=entropy)
    state = c()

    target = [0, 0, 1.0]
    np.testing.assert_allclose(entropy[0].numpy(), target, atol=_atol)


def test_entropy_steps():
    """Check that using steps skips the appropriate number of gates."""
    entropy = callbacks.EntanglementEntropy([0], steps=2)
    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.CNOT(0, 1))
    c.add(gates.H(1))
    c.add(gates.CNOT(1, 0))
    c.compile(callback=entropy)
    state = c()

    target = [0, 1.0, 1.0]
    np.testing.assert_allclose(entropy[0].numpy(), target, atol=_atol)


def test_entropy_multiple_executions():
    """Check entropy calculation when the callback is used in multiple executions."""
    entropy = callbacks.EntanglementEntropy([0], steps=2)

    c = Circuit(2)
    c.add(gates.RY(0, 0.1234))
    c.add(gates.CNOT(0, 1))
    state = c(callback=entropy)

    c = Circuit(2)
    c.add(gates.RY(0, 0.4321))
    c.add(gates.CNOT(0, 1))
    state = c(callback=entropy)

    def target_entropy(t):
        cos = np.cos(t / 2.0) ** 2
        sin = np.sin(t / 2.0) ** 2
        return - cos * np.log2(cos) - sin * np.log2(sin)

    target = [[0, target_entropy(0.1234)], [0, target_entropy(0.4321)]]
    np.testing.assert_allclose(entropy[:].numpy(), target)
