from dataclasses import dataclass, field
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from numpy.typing import ArrayLike

from qibo.config import raise_error
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z


class Arrow3D(FancyArrowPatch):
    """This class creates the arrows for the Bloch sphere."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self):
        """This function performs the 3D projection of the arrow rendering.

        This method is automatically called by Matplotlib's 3D axis
        to transform the arrow's 3D coordinates into 2D screen coordinates,
        handling perspective and position for correct display.
        """
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


@dataclass
class BlochSphere:
    """This class creates a Bloch sphere."""

    @staticmethod
    def _make_style():
        _STYLE = {
            "figure.figsize": (6, 6),
            "lines.linewidth": 0.9,
        }
        return _STYLE

    @staticmethod
    def _make_style_text():
        return {"text.color": "black", "font.size": 19}

    # Plot style sheets
    STYLE_TEXT: dict = field(default_factory=_make_style_text)
    STYLE: dict = field(default_factory=_make_style)

    # Data
    _points: list = field(default_factory=list)
    _vectors: list = field(default_factory=list)

    # Color data
    _color_points: list = field(default_factory=list)
    _color_vectors: list = field(default_factory=list)

    _shown: bool = False

    def __post_init__(self):
        # No toolbar
        mpl.rcParams["toolbar"] = "None"

        # Figure
        self.fig = plt.figure(figsize=self.STYLE["figure.figsize"])
        self.ax = self.fig.add_subplot(projection="3d", elev=30, azim=30)

        # Title of the window
        self.fig.canvas.manager.set_window_title("Bloch sphere")

    def _new_window(self):
        """It creates a new Figure object and it adds to it a new Axis."""
        self.fig = plt.figure(figsize=self.STYLE["figure.figsize"])
        self.ax = self.fig.add_subplot(projection="3d", elev=30, azim=30)
        self.fig.canvas.manager.set_window_title("Bloch sphere")

    # -----Sphere-----
    def _sphere_surface(self):
        """Helper method to `_create_sphere` to construct the sphere's surface"""
        phi, theta = np.mgrid[0.0 : np.pi : 100j, 0.0 : 2.0 * np.pi : 100j]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return x, y, z

    def _axis(self):
        """Helper method to `_create_sphere` to construct the sphere's axis."""
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.zeros(100)
        x = np.sin(theta)
        y = np.cos(theta)
        return x, y, z

    def _parallel(self, z):
        """Helper method to `_create_sphere` to construct the sphere's parallels."""
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.full(100, z)
        r = np.sqrt(1 - z[0] ** 2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y, z

    def _meridian(self, phi):
        """Helper method to `_create_sphere` to construct the sphere's meridians."""
        theta = np.linspace(0, 2 * np.pi, 100)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return x, y, z

    def _create_sphere(self):
        "This function builds an empty Bloch sphere."

        # Empty sphere
        self.ax.plot_surface(*self._sphere_surface(), color="lavenderblush", alpha=0.2)

        # Axis
        x, y, z = self._axis()
        combinations_axis = [(x, y, z), (z, x, y), (y, z, x)]

        # Axis lines
        line, zeros = np.linspace(-1, 1, 100), np.zeros(shape=(100))
        combinations_axis_line = [
            (line, zeros, zeros),
            (zeros, line, zeros),
            (zeros, zeros, line),
        ]

        # Meridian and Parallel
        phi = [(n + 1) * np.pi / 3 for n in range(6)]
        lat = (0.4, -0.4, 0.9, -0.9)
        meridian = lambda x: self.ax.plot(*self._meridian(x), color="darkgrey")
        parallel = lambda x: self.ax.plot(*self._parallel(x), color="darkgrey")

        # Axis, Axis lines, Meridians, Parallels
        with mpl.rc_context(self.STYLE):
            [self.ax.plot(*combinations_axis[i], color="darkgrey") for i in range(3)]
            [
                self.ax.plot(*combinations_axis_line[i], color="darkgrey")
                for i in range(3)
            ]
            [meridian(p) for p in phi]
            [parallel(l) for l in lat]

        # Text
        with mpl.rc_context(self.STYLE_TEXT):
            self.ax.text(1.2, 0, 0, "x", ha="center")
            self.ax.text(0, 1.2, 0, "y", ha="center")
            self.ax.text(0, 0, 1.2, r"$|0\rangle$", ha="center")
            self.ax.text(0, 0, -1.3, r"$|1\rangle$", ha="center")

    # -----States and Vectors-----
    def _paulis_expectation(self, state):
        """This function computes the expectation value of Pauli matrices
        on the considered state and yields its cartesian coordinates on the Bloch sphere.
        """

        sigma_X = SymbolicHamiltonian(X(0))
        sigma_Y = SymbolicHamiltonian(Y(0))
        sigma_Z = SymbolicHamiltonian(Z(0))

        x = sigma_X.expectation_from_state(state)
        y = sigma_Y.expectation_from_state(state)
        z = sigma_Z.expectation_from_state(state)
        return x, y, z

    def _coordinates(self, state):
        """This function determines the coordinates of a qubit in the sphere."""

        x, y, z = 0, 0, 0
        if state.ndim == 1:
            if state[0] == 1 and state[0] == 0:
                z = 1
                return x, y, z

            if state[0] == 0 and state[0] == 1:
                z = -1
                return x, y, z

            return self._paulis_expectation(state)

        return self._paulis_expectation(state)

    def _is_density_matrix(self, rho: np.ndarray) -> bool:
        """This function is used only to check whether an input of shape (2,2) is two state vectors or one density matrix."""
        return np.allclose(rho, rho.conj().T) and np.isclose(np.trace(rho), 1)

    def _broadcasting_semantics(self, vector, mode, color):
        """This function makes sure that `vector`, `mode`, `color` have the same sizes."""
        if isinstance(vector, list):
            vector = np.array(vector)
        if isinstance(mode, (list, str)):
            mode = np.array(mode)
        if isinstance(color, (list, str)):
            color = np.array(color)

        # Check to distinguish if (2,2) is one density matrix or two state vectors.
        if vector.ndim == 2 and vector.shape == (2, 2):
            if self._is_density_matrix(vector):
                vector = np.expand_dims(vector, axis=0)

        vector = np.atleast_2d(vector)
        vector, mode, color = np.broadcast_arrays(vector, mode, color)
        return vector, mode.flatten(), color.flatten()

    def add_vector(
        self,
        vector: ArrayLike,
        mode: Union[str, list[str]] = "vector",
        color: Union[str, list[str]] = "black",
    ):
        """This function adds a vector to the sphere."""

        vectors, modes, colors = self._broadcasting_semantics(vector, mode, color)
        for vector, color, mode in zip(vectors, colors, modes):
            if mode not in ("vector", "point"):
                raise_error(ValueError, "Mode not supported. Try: `point` or `vector`.")
            if mode == "vector":
                self._vectors.append(vector)
                self._color_vectors.append(color)
            else:
                self._points.append(vector)
                self._color_points.append(color)

    def add_state(
        self,
        state: ArrayLike,
        mode: Union[str, list[str]] = "vector",
        color: Union[str, list[str]] = "black",
    ):
        """This function adds a state to the sphere."""

        vectors, modes, colors = self._broadcasting_semantics(state, mode, color)
        for vector, color, mode in zip(vectors, colors, modes):
            if mode not in ("vector", "point"):
                raise_error(ValueError, "Mode not supported. Try: `point` or `vector`.")

            x, y, z = self._coordinates(vector)

            if mode == "vector":
                self._vectors.append(np.array([x, y, z]))
                self._color_vectors.append(color)
            else:
                self._points.append(np.array([x, y, z]))
                self._color_points.append(color)

    # -----Clear and produce the sphere-----
    def clear(self):
        """This function clears the sphere."""
        plt.close()
        self._new_window()

        # Clear data
        self._points = []
        self._vectors = []

        # Clear Color
        self._color_points = []
        self._color_vectors = []

        self.render()

    def render(self):
        """This function creates the empty sphere and plots the
        vectors and points on it."""
        if self._shown == True:
            plt.close()
            self._new_window()
        self._shown = True

        self._create_sphere()

        for color, vector in zip(self._color_vectors, self._vectors):
            xs3d = vector[0] * np.array([0, 1])
            ys3d = vector[1] * np.array([0, 1])
            zs3d = vector[2] * np.array([0, 1])
            a = Arrow3D(
                xs3d,
                ys3d,
                zs3d,
                lw=2.0,
                arrowstyle="-|>",
                mutation_scale=20,
                color=color,
            )
            self.ax.add_artist(a)

        for color, point in zip(self._color_points, self._points):
            self.ax.scatter(point[0], point[1], point[2], color=color, s=10)

        self.ax.set_xlim([-0.7, 0.7])
        self.ax.set_ylim([-0.7, 0.7])
        self.ax.set_zlim([-0.7, 0.7])
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.fig.tight_layout()
