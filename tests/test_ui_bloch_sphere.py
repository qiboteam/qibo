import matplotlib
import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.ui.bloch import Bloch

BACKEND = "qtagg"


def test_empty_sphere():
    bs = Bloch(backend=BACKEND)
    bs.plot()


def test_state():
    bs = Bloch(backend=BACKEND)
    state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2) * 1j], dtype="complex")
    bs.add_state(state)
    bs.plot()


def test_vector_point():
    bs = Bloch(backend=BACKEND)
    vector = np.array([0, 0, 1])
    bs.add_vector(vector, mode="point", color="green")
    bs.plot()


def test_multiple_vectors_array():
    bs = Bloch(backend=BACKEND)
    vectors = np.random.normal(size=(100, 3))
    vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    bs.add_vector(vectors, color="royalblue")
    bs.plot()


def test_multiple_vectors_list():
    bs = Bloch(backend=BACKEND)

    vectors = []
    for i in range(100):
        vector = np.random.normal(size=(3,))
        vector /= np.linalg.norm(vector)
        vectors.append(vector)

    bs.add_vector(vectors, color=["royalblue"] * 100)
    bs.plot()


def test_multiple_states():
    bs = Bloch(backend=BACKEND)
    states = np.random.normal(size=(100, 2))
    states /= np.linalg.norm(states, axis=1)[:, np.newaxis]
    bs.add_state(states)
    bs.plot()


def test_state_clear():
    bs = Bloch(backend=BACKEND)

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


def _circuit():
    circ = Circuit(1)
    circ.add(gates.RY(q=0, theta=np.random.randn() * 0.1))
    circ.add(gates.RX(q=0, theta=np.random.randn() * 0.1))
    circ.add(gates.RZ(q=0, theta=np.random.randn() * 0.1))
    return circ


def test_classification():
    bs = Bloch(backend=BACKEND)
    bs.add_state(np.array([1, 0]), color="black")
    bs.add_state(np.array([0, 1]), color="black")

    for i in range(20):
        state = _circuit()(np.array([1, 0], dtype="complex")).state()
        bs.add_state(state, mode="point", color="red")

    for i in range(20):
        state = _circuit()(np.array([0, 1], dtype="complex")).state()
        bs.add_state(state, mode="point", color="blue")

    bs.plot()


def test_multi_classification():
    bs = Bloch(backend=BACKEND)
    bs.add_state(np.array([1, 0]), color="black")
    bs.add_state(np.array([0, 1]), color="black")
    bs.add_vector(np.array([0, 1, 0]), color="black")

    for i in range(20):
        state = _circuit()(np.array([1, 0], dtype="complex")).state()
        bs.add_state(state, mode="point", color="red")

    for i in range(20):
        circ = _circuit()
        circ.add(gates.RX(q=0, theta=-np.pi / 2))
        bs.add_state(circ().state(), mode="point", color="orange")

    for i in range(20):
        circ = _circuit()
        state = circ(np.array([0, 1], dtype="complex")).state()
        bs.add_state(state, mode="point", color="blue")

    bs.plot()


def test_qibo_output():
    bs = Bloch(backend=BACKEND)

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
    bs.plot()


def test_point_vector_state():
    bs = Bloch(backend=BACKEND)

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

    bs.plot()


def test_save():
    bs = Bloch(backend=BACKEND)

    nqubits = 1
    layers = 2
    circ = Circuit(nqubits)
    for i in range(layers):
        circ.add(gates.RY(q=0, theta=np.random.randn()))
        circ.add(gates.RZ(q=0, theta=np.random.randn()))
    state = circ().state()

    bs.add_state(state, color=["orange"])
    bs.save("bloch.pdf")


def class_state():

    circ = _circuit()
    state = circ().state()
    vector = [[0, 1, 0], [0, 1, 1]]
    bs = Bloch(backend=BACKEND)
    bs.plot()


