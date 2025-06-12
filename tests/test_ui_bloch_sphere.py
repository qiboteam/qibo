import numpy as np
import matplotlib

matplotlib.use("Agg")

import pytest

from qibo import Circuit, gates
from qibo.ui.bloch import Bloch


def test_empty_sphere():
    bs = Bloch()
    bs.plot()


def test_state():
    bs = Bloch()
    state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2) * 1j], dtype="complex")
    bs.add_state(state, color="blue")
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
    bs.add_vector(vectors)
    bs.plot()


def test_multiple_vectors_list():
    bs = Bloch()

    vectors = []
    for i in range(100):
        vector = np.random.normal(size=(3,))
        vector /= np.linalg.norm(vector)
        vectors.append(vector)

    bs.add_vector(vectors)
    bs.plot()


def test_multiple_states():
    bs = Bloch()
    states = np.random.normal(size=(100, 2))
    states /= np.linalg.norm(states, axis=1)[:, np.newaxis]
    bs.add_state(states)
    bs.plot()


def test_clear():
    bs = Bloch()
    states = np.random.normal(size=(100, 2))
    states /= np.linalg.norm(states, axis=1)[:, np.newaxis]
    bs.add_state(states)
    bs.plot()
    bs.clear()
    bs.plot()


def test_qibo_output():
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
    bs.plot()
