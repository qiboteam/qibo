import numpy as np
import matplotlib as mpl
import tkinter
import importlib

from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.backends import backend_agg, backend_tkagg

from qibo import hamiltonians
from qibo.config import raise_error
from qibo.symbols import X, Y, Z

from dataclasses import dataclass, field
from typing import Union, Optional
from numpy.typing import ArrayLike
from cycler import cycler


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


@dataclass
class Bloch:

    STYLE = {
        "figure.figsize": (6, 6),
        "lines.linewidth": 0.9,
    }

    STYLE_TEXT = {"text.color": "black", "font.size": 19}

    main_alpha: float = 0.6
    secondary_alpha: float = 0.2

    _shown: bool = False

    # Data
    points: list = field(default_factory=list)
    vectors: list = field(default_factory=list)

    # Color data
    color_points: list = field(default_factory=list)
    color_vectors: list = field(default_factory=list)

    # Figure and axis
    def __post_init__(self):
        self.fig = Figure(figsize=self.STYLE["figure.figsize"])
        self.ax = self.fig.add_subplot(projection="3d", elev=30, azim=30)

    def clear(self):
        # Figure
        self.fig.clear()

        # Data
        self.points = []
        self.vectors = []

        # Color data
        self.color_points = []
        self.color_vectors = []

    def _sphere_surface(self):
        phi, theta = np.mgrid[0.0 : np.pi : 100j, 0.0 : 2.0 * np.pi : 100j]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return x, y, z

    def _axis(self):
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.zeros(100)
        x = np.sin(theta)
        y = np.cos(theta)
        return x, y, z

    def _latitude(self, z):
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.full(100, z)
        r = np.sqrt(1 - z[0] ** 2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y, z

    def _meridian(self, phi):
        theta = np.linspace(0, 2 * np.pi, 100)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return x, y, z

    def create_sphere(self):
        "Function to create an empty sphere."

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

        # Meridian and Latitude
        phi = [(n + 1) * np.pi / 3 for n in range(6)]
        lat = (0.4, -0.4, 0.9, -0.9)
        meridian = lambda x: self.ax.plot(*self._meridian(x), color="darkgrey")
        latitude = lambda x: self.ax.plot(*self._latitude(x), color="darkgrey")

        # Axis, Axis lines, Meridians, Latitudes
        with mpl.rc_context(self.STYLE):
            [self.ax.plot(*combinations_axis[i], color="darkgrey") for i in range(3)]
            [
                self.ax.plot(*combinations_axis_line[i], color="darkgrey")
                for i in range(3)
            ]
            [meridian(p) for p in phi]
            [latitude(l) for l in lat]

        # Text
        with mpl.rc_context(self.STYLE_TEXT):
            self.ax.text(1.2, 0, 0, "x", ha="center")
            self.ax.text(0, 1.2, 0, "y", ha="center")
            self.ax.text(0, 0, 1.2, r"$|0\rangle$", ha="center")
            self.ax.text(0, 0, -1.3, r"$|1\rangle$", ha="center")

    def _coordinates(self, state):
        "Function to determine the coordinates of a qubit in the sphere."

        x, y, z = 0, 0, 0
        if state[0] == 1 and state[0] == 0:
            z = 1
        elif state[0] == 0 and state[0] == 1:
            z = -1
        else:
            sigma_X = hamiltonians.SymbolicHamiltonian(X(0))
            sigma_Y = hamiltonians.SymbolicHamiltonian(Y(0))
            sigma_Z = hamiltonians.SymbolicHamiltonian(Z(0))

            x = sigma_X.expectation(state)
            y = sigma_Y.expectation(state)
            z = sigma_Z.expectation(state)
        return x, y, z

    def _broadcasting_semantics(self, vector, mode, color):
        if isinstance(vector, list):
            vectors = np.array(vector)

        num_vector = len(vectors)
        if isinstance(mode, str):
            modes = [mode] * num_vector
        elif len(mode) == num_vector:
            modes = [mode]
        else:
            raise_error(ValueError, "`mode` and `vector` should have same length.")

        if isinstance(color, str):
            colors = [color] * num_vector
        elif len(color) == num_vector:
            colors = [color]
        else:
            raise_error(ValueError, "`mode` and `vector` should have same length.")

        return vectors, colors, modes

    def add_vector(
        self,
        vector: ArrayLike,
        mode: Union[str, list[str]] = "vector",
        color: Union[str, list[str]] = "black",
    ):
        vectors, colors, modes = self._broadcasting_semantics(vector, mode, color)
        for vector, color, mode in zip(vectors, colors, modes):
            if mode == "vector":
                self.vectors.append(vector)
                self.color_vectors.append(color)
            elif mode == "point":
                self.points.append(vector)
                self.color_points.append(color)
            else:
                raise_error(ValueError, "Mode not supported. Try: `point` or `vector`.")

    def add_state(
        self,
        state: ArrayLike,
        mode: Union[str, list[str]] = "vector",
        color: Union[str, list[str]] = "black",
    ):
        "Function to add a state to the sphere."

        vectors, colors, modes = self._broadcasting_semantics(vector, mode, color)
        for vector, color, mode in zip(vectors, colors, modes):
            x, y, z = self._coordinates(vector)
            if mode == "vector":
                self.vectors.append(np.array([x, y, z]))
                self.color_vectors.append(color)
            elif mode == "point":
                self.points.append(np.array([x, y, z]))
                self.color_points.append(color)
            else:
                raise_error(ValueError, "Mode not supported. Try: `point` or `vector`.")

    def _rendering(self):
        if self._shown == True:
            self.fig.clear()

        self._shown = True

        self.create_sphere()

        for color, vector in zip(self.color_vectors, self.vectors):
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

        for color, point in zip(self.color_points, self.points):
            self.ax.scatter(point[0], point[1], point[2], color=color, s=10)

        self.ax.set_xlim([-0.7, 0.7])
        self.ax.set_ylim([-0.7, 0.7])
        self.ax.set_zlim([-0.7, 0.7])
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.fig.tight_layout()

    def save(self, filename="bloch_sphere.pdf"):
        self._rendering()

        mpl.use("Agg")
        canvas = backend_agg.FigureCanvas(self.fig)

        self.fig.savefig(filename)

    def plot(self):
        self._rendering()

        mpl.use("tkagg")
        manager = backend_tkagg.new_figure_manager_given_figure(1, self.fig)
        manager.show()
        tkinter.mainloop()

        self.fig.show()
