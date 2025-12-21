import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use("agg")

from qibo import Circuit, gates
from qibo.ui.bloch import BlochSphere


def _circuit(weight, boolean=False):
    circ = Circuit(1, density_matrix=boolean)
    circ.add(gates.RY(q=0, theta=np.random.randn() * weight))
    circ.add(gates.RX(q=0, theta=np.random.randn() * weight))
    circ.add(gates.RZ(q=0, theta=np.random.randn() * weight))
    return circ


def test_empty_sphere():
    bs = BlochSphere()
    bs.render()
    plt.show()


def test_state():
    bs = BlochSphere()
    state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2) * 1j], dtype="complex")
    bs.add_state(state)
    bs.render()
    plt.show()


def test_vector_point():
    bs = BlochSphere()
    vector = np.array([0, 0, 1])
    bs.add_vector(vector, mode="point", color="green")
    bs.render()
    plt.show()


def test_multiple_vectors_array():
    bs = BlochSphere()
    vectors = np.random.normal(size=(100, 3))
    vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    bs.add_vector(vectors, color="royalblue")
    bs.render()
    plt.show()


def test_multiple_vectors_list():
    bs = BlochSphere()

    vectors = []
    for _ in range(100):
        vector = np.random.normal(size=(3,))
        vector /= np.linalg.norm(vector)
        vectors.append(vector)

    bs.add_vector(vectors, color="royalblue")
    bs.render()
    plt.show()


def test_multiple_states():
    bs = BlochSphere()
    states = np.random.normal(size=(100, 2))
    states /= np.linalg.norm(states, axis=1)[:, np.newaxis]
    bs.add_state(states)
    bs.render()
    plt.show()


def test_state_clear():
    bs = BlochSphere()
    states = np.zeros(shape=(100, 2), dtype="complex")

    for i in range(100):
        real_part = np.random.normal(loc=0.0, scale=1.0, size=2)
        imag_part = np.random.normal(loc=0.0, scale=1.0, size=2)
        state = real_part + 1j * imag_part
        state /= np.linalg.norm(state)
        states[i] = state

    bs.add_state(states)
    bs.render()
    plt.show()

    bs.clear()
    bs.render()
    plt.show()


def test_classification():
    bs = BlochSphere()
    bs.add_state(np.array([1, 0]), color="black")
    bs.add_state(np.array([0, 1]), color="black")

    weight = 0.1

    for _ in range(20):
        state = _circuit(weight)(np.array([1, 0], dtype="complex")).state()
        bs.add_state(state, mode="point", color="red")

    for _ in range(20):
        state = _circuit(weight)(np.array([0, 1], dtype="complex")).state()
        bs.add_state(state, mode="point", color="blue")

    bs.render()
    plt.show()


def test_multi_classification():
    bs = BlochSphere()
    bs.add_state(np.array([1, 0]), color="black")
    bs.add_state(np.array([0, 1]), color="black")
    bs.add_vector(np.array([0, 1, 0]), color="black")

    weight = 0.1

    for _ in range(20):
        state = _circuit(weight)(np.array([1, 0], dtype="complex")).state()
        bs.add_state(state, mode="point", color="red")

    for _ in range(20):
        circ = _circuit(weight)
        circ.add(gates.RX(q=0, theta=-np.pi / 2))
        bs.add_state(circ().state(), mode="point", color="orange")

    for _ in range(20):
        circ = _circuit(weight)
        state = circ(np.array([0, 1], dtype="complex")).state()
        bs.add_state(state, mode="point", color="magenta")

    bs.render()
    plt.show()


def test_qibo_output():
    bs = BlochSphere()

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
    bs.render()
    plt.show()


def test_point_vector_state():
    bs = BlochSphere()

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

    bs.render()
    plt.show()


def test_save():
    bs = BlochSphere()

    nqubits = 1
    layers = 2

    circ = Circuit(nqubits)
    for _ in range(layers):
        circ.add(gates.RY(q=0, theta=np.random.randn()))
        circ.add(gates.RZ(q=0, theta=np.random.randn()))
    state = circ().state()

    bs.add_state(state, color=["orange"])
    bs.render()
    plt.savefig("bloch.pdf")
    plt.close()


def test_many_spheres():
    bs = BlochSphere()
    bs.render()

    weight = 0.1

    for _ in range(20):
        circ = _circuit(weight)
        state = circ(np.array([0, 1], dtype="complex")).state()
        bs.add_state(state, mode="point", color="blue")

    bs.render()
    plt.show()
    bs.clear()

    for _ in range(20):
        circ = _circuit(weight)
        state = circ(np.array([0, 1], dtype="complex")).state()
        bs.add_state(state, mode="point", color="red")

    bs.render()
    plt.show()
    bs.clear()

    bs.render()
    plt.show()


def test_density_matrix():
    bs = BlochSphere()
    weight = 1.0
    boolean = True

    states = _circuit(weight, boolean)(
        np.array([[1, 0], [0, 0]], dtype="complex")
    ).state()

    bs.add_state(states, mode="vector", color="red")

    bs.render()
    plt.show()


def test_density_matrix_vs_state():
    """There are six possible scenario:
        1. Single state vector --> Shape = (2,)
        2. Two state vectors --> Shape = (2,2)
        3. n state vector --> Shape = (n,2)
        4. Single rho --> Shape = (2,2)
        5. Two rhos --> Shape = (2,2,2)
        6. n rhos --> Shape = (n,2,2)
    So the code could have problems distinguishing case 2 and case 4.
    We can solve this problem just by adding an extra dimension to case 4, thus (2,2) -> (1,2,2).
    In this way every time that we have a density matrix the input will have three dimensions.
    """

    bs = BlochSphere()
    weight = 1.0

    # Scenario 1
    boolean = False
    state = _circuit(weight, boolean)(np.array([1, 0], dtype="complex")).state()
    bs.add_state(state, mode="vector", color="red")
    bs.render()
    plt.show()
    bs.clear()

    # Scenario 2
    boolean = False
    states = [
        _circuit(weight, boolean)(np.array([1, 0], dtype="complex")).state()
        for _ in range(2)
    ]
    bs.add_state(states, mode="vector", color="red")
    bs.render()
    plt.show()
    bs.clear()

    # Scenario 3
    boolean = False
    states = [
        _circuit(weight, boolean)(np.array([1, 0], dtype="complex")).state()
        for _ in range(10)
    ]
    bs.add_state(states, mode="vector", color="red")
    bs.render()
    plt.show()
    bs.clear()

    # Scenario 4
    boolean = True
    rho = _circuit(weight, boolean)(np.array([[1, 0], [0, 0]], dtype="complex")).state()
    bs.add_state(rho, mode="vector", color="blue")
    bs.render()
    plt.show()
    bs.clear()

    # Scenario 5
    boolean = True
    rhos = [
        _circuit(weight, boolean)(np.array([[1, 0], [0, 0]], dtype="complex")).state()
        for _ in range(2)
    ]
    bs.add_state(rhos, mode="vector", color="blue")
    bs.render()
    plt.show()
    bs.clear()

    # Scenario 6
    boolean = True
    rhos = [
        _circuit(weight, boolean)(np.array([[1, 0], [0, 0]], dtype="complex")).state()
        for _ in range(10)
    ]
    bs.add_state(rhos, mode="vector", color="blue")
    bs.render()
    plt.show()
    bs.clear()
    plt.close()


def test_mixed_state():
    bs = BlochSphere()

    state = [0.5, 0, 0]
    bs.add_vector(state)

    bs.render()
    plt.show()


def test_bloch_error_mode():
    bs = BlochSphere()

    vector = np.array([1, 0, 0])

    try:
        bs.add_vector(vector, mode="invalid_mode")
    except ValueError as e:
        assert str(e) == "Mode not supported. Try: `point` or `vector`."

    try:
        bs.add_state(vector, mode="invalid_mode")
    except ValueError as e:
        assert str(e) == "Mode not supported. Try: `point` or `vector`."
