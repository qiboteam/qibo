import matplotlib
import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.ui.bloch import Bloch

# matplotlib.use("Agg")




def test_empty_sphere():
    bs = Bloch()
    bs.plot()


def test_state():
    bs = Bloch()
    state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2) * 1j], dtype="complex")
    bs.add_state(state)
    bs.plot()


@pytest.mark.parametrize("mode", ["vector", "point"])
def test_vector_point(mode):
    bs = Bloch()
    vector = np.array([0, 0, 1])
    bs.add_vector(vector, mode=mode, color="green")
    bs.plot()


def test_multiple_vectors_array():
    bs = Bloch()
    vectors = np.random.normal(size=(100, 3))
    vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    bs.add_vector(vectors, color=["royalblue"] * 100)
    bs.plot(save=True, filename="bloch_multiple_vectors.pdf")


def test_multiple_vectors_list():
    bs = Bloch()

    vectors = []
    for i in range(100):
        vector = np.random.normal(size=(3,))
        vector /= np.linalg.norm(vector)
        vectors.append(vector)

    bs.add_vector(vectors, color=["royalblue"] * 100)
    bs.plot(save=True, filename="bloch_1.pdf")


def test_multiple_states():
    bs = Bloch()
    states = np.random.normal(size=(100, 2))
    states /= np.linalg.norm(states, axis=1)[:, np.newaxis]
    bs.add_state(states)
    bs.plot(save=True, filename="bloch_2.pdf")


def test_clear():
    bs = Bloch()

    states = np.zeros(shape=(100, 2), dtype="complex")

    for i in range(100):
        real_part = np.random.normal(loc=0.0, scale=1.0, size=2)
        imag_part = np.random.normal(loc=0.0, scale=1.0, size=2)
        state = real_part + 1j * imag_part
        state /= np.linalg.norm(state)
        states[i] = state

    bs.add_state(states)
    bs.plot()
    bs.clear()
    bs.plot()


def test_classification():
    bs = Bloch()
    bs.add_state(np.array([1, 0]), color="black")
    bs.add_state(np.array([0, 1]), color="black")

    for i in range(20):
        circ = Circuit(1)
        circ.add(gates.RY(q=0, theta=np.random.randn() * 0.1))
        circ.add(gates.RX(q=0, theta=np.random.randn() * 0.1))
        circ.add(gates.RZ(q=0, theta=np.random.randn() * 0.1))
        state = circ().state()
        bs.add_state(state, mode="point", color="red")

    for i in range(20):
        circ = Circuit(1)
        circ.add(gates.RY(q=0, theta=np.random.randn() * 0.1))
        circ.add(gates.RX(q=0, theta=np.random.randn() * 0.1))
        circ.add(gates.RZ(q=0, theta=np.random.randn() * 0.1))
        state = circ(np.array([0, 1], dtype="complex")).state()
        bs.add_state(state, mode="point", color="blue")

    bs.plot(save=True, filename="binary_classification.pdf")


def test_multi_classification():
    bs = Bloch()
    bs.add_state(np.array([1, 0]), color="black")
    bs.add_state(np.array([0, 1]), color="black")
    bs.add_vector(np.array([0, 1, 0]), color="black")

    for i in range(20):
        circ = Circuit(1)
        circ.add(gates.RY(q=0, theta=np.random.randn() * 0.1))
        circ.add(gates.RX(q=0, theta=np.random.randn() * 0.1))
        circ.add(gates.RZ(q=0, theta=np.random.randn() * 0.1))
        state = circ().state()
        bs.add_state(state, mode="point", color="red")

    for i in range(20):
        circ = Circuit(1)
        circ.add(gates.RX(q=0, theta=-np.pi / 2))
        circ.add(gates.RY(q=0, theta=np.random.randn() * 0.1))
        circ.add(gates.RZ(q=0, theta=np.random.randn() * 0.1))
        circ.add(gates.RX(q=0, theta=np.random.randn() * 0.1))
        state = circ().state()
        bs.add_state(state, mode="point", color="orange")

    for i in range(20):
        circ = Circuit(1)
        circ.add(gates.RY(q=0, theta=np.random.randn() * 0.1))
        circ.add(gates.RX(q=0, theta=np.random.randn() * 0.1))
        circ.add(gates.RZ(q=0, theta=np.random.randn() * 0.1))
        state = circ(np.array([0, 1], dtype="complex")).state()
        bs.add_state(state, mode="point", color="blue")

    bs.plot(save=True, filename="multi_class_classification.pdf")


def test_qibo_output():
    bs = Bloch()

    # --Circuit--
    nqubits = 1
    layers = 2
    circ = Circuit(nqubits)
    for i in range(layers):
        circ.add(gates.RY(q=0, theta=np.random.randn()))
        circ.add(gates.RZ(q=0, theta=np.random.randn()))

    # state
    state = circ().state()
    bs.add_state(state, color="orange")
    bs.plot(save=True, filename="bloch_4.pdf")


def test_point_vector_state():
    bs = Bloch()

    # --Circuit--
    nqubits = 1
    layers = 2
    circ = Circuit(nqubits)
    for i in range(layers):
        circ.add(gates.RY(q=0, theta=np.random.randn()))
        circ.add(gates.RZ(q=0, theta=np.random.randn()))

    # state
    state = circ().state()
    bs.add_state(state, color="orange")

    # point
    point = np.array([0, 0, 1])
    bs.add_vector(point, mode="point", color="blue")

    # vector
    vector = np.array([1, 0, 0])
    bs.add_vector(vector, mode="vector", color="red")

    # --Sphere--
    bs.plot(save=True, filename="bloch_sphere_point_state_vector.pdf")


def test_save():
    bs = Bloch()

    # --Circuit--
    nqubits = 1
    layers = 2
    circ = Circuit(nqubits)
    for i in range(layers):
        circ.add(gates.RY(q=0, theta=np.random.randn()))
        circ.add(gates.RZ(q=0, theta=np.random.randn()))
    state = circ().state()

    # --Sphere--
    bs.add_state(state, color=["orange"])
    bs.plot(save=True, filename="bloch_sphere_1.pdf")


def test_mismatch_colors_state_1():
    bs = Bloch()

    # --Circuit--
    nqubits = 1
    layers = 2
    circ = Circuit(nqubits)
    for i in range(layers):
        circ.add(gates.RY(q=0, theta=np.random.randn()))
        circ.add(gates.RZ(q=0, theta=np.random.randn()))
    state = circ().state()

    # --Sphere--
    bs.add_state(state, color=["orange", "blue"])
    bs.plot()


def test_mismatch_colors_state_2():
    bs = Bloch()

    # --Circuit--
    nqubits = 1
    layers = 2
    states = []
    for j in range(10):
        circ = Circuit(nqubits)
        for i in range(layers):
            circ.add(gates.RY(q=0, theta=np.random.randn()))
            circ.add(gates.RZ(q=0, theta=np.random.randn()))
        state = circ().state()
        states.append(state)

    # --Sphere--
    bs.add_state(states, color=["red"])
    bs.plot()


def test_mismatch_colors_state_4():
    bs = Bloch()

    # --Circuit--
    nqubits = 1
    layers = 2
    states = np.zeros((10, 2), dtype=np.complex128)
    for j in range(10):
        circ = Circuit(nqubits)
        for i in range(layers):
            circ.add(gates.RY(q=0, theta=np.random.randn()))
            circ.add(gates.RZ(q=0, theta=np.random.randn()))
        state = circ().state()
        states[j] = state

    # --Sphere--
    bs.add_state(states, color=np.array(["red"]))
    bs.plot()


def test_mismatch_colors_state_5():
    bs = Bloch()

    # --Circuit--
    nqubits = 1
    layers = 2
    states = np.zeros((10, 2), dtype=np.complex128)
    for j in range(10):
        circ = Circuit(nqubits)
        for i in range(layers):
            circ.add(gates.RY(q=0, theta=np.random.randn()))
            circ.add(gates.RZ(q=0, theta=np.random.randn()))
        state = circ().state()
        states[j] = state

    # --Sphere--
    bs.add_state(states, color="red")
    bs.plot()


test_mismatch_colors_state_5()
