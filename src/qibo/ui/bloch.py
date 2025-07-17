import importlib
import tkinter as tk
from dataclasses import dataclass, field
from typing import Union
from IPython.display import display, clear_output

import matplotlib as mpl
from matplotlib.backends import backend_agg, backend_qtagg, backend_tkagg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d

import numpy as np
from numpy.typing import ArrayLike

from qibo import hamiltonians
from qibo.config import raise_error
from qibo.symbols import X, Y, Z


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
    """This class creates a Bloch sphere.

    It takes advantage of three Matplotlibs's backends: qtagg, tkagg, agg.
    The first two are interactive backends required to display the sphere in a window, the last one is
    non-interactive and it is used to save the figure."""

    backend: str = "tkagg"

    # Plot style sheets
    STYLE = {
        "figure.figsize": (6, 6),
        "lines.linewidth": 0.9,
    }
    STYLE_TEXT = {"text.color": "black", "font.size": 19}

    # Data
    _points: list = field(default_factory=list)
    _vectors: list = field(default_factory=list)

    # Color data
    _color_points: list = field(default_factory=list)
    _color_vectors: list = field(default_factory=list)

    # Figure and axis
    def __post_init__(self):
        # Figure
        self.fig = Figure(figsize=self.STYLE["figure.figsize"])
        self.ax = self.fig.add_subplot(projection="3d", elev=30, azim=30)

        # Backends
        if self.backend == "qtagg":
            mpl.use(self.backend)
            self._backend = importlib.import_module(
                "matplotlib.backends.backend_" + self.backend
            )
        elif self.backend == "tkagg":
            mpl.use(self.backend)
        elif self.backend == "agg":
            mpl.use(self.backend)
        elif self.backend == "jupyter":
            pass
        else:
            raise_error(
                ValueError,
                "Backend not supported. Try: `qtagg`, `tkagg` (interactive), `agg` (non-interactive) or `jupyter`.",
            )

    def _new_window(self):
        """It creates a new Figure object and it adds to it a new Axis."""
        self.fig = Figure(figsize=self.STYLE["figure.figsize"])
        self.ax = self.fig.add_subplot(projection="3d", elev=30, azim=30)

    # -----Sphere-----
    def _sphere_surface(self):
        """Helper method to `_create_sphere` to construct the sphere's surface"""
        phi, theta = np.mgrid[0.0 : np.pi : 100j, 0.0 : 2.0 * np.pi : 100j]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return x, y, z

    def _axis(self):
        """Helper method to `_create_sphere` to construct the sphere's axis"""
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.zeros(100)
        x = np.sin(theta)
        y = np.cos(theta)
        return x, y, z

    def _latitude(self, z):
        """Helper method to `_create_sphere` to construct the sphere's latitudes"""
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.full(100, z)
        r = np.sqrt(1 - z[0] ** 2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y, z

    def _meridian(self, phi):
        """Helper method to `_create_sphere` to construct the sphere's meridians"""
        theta = np.linspace(0, 2 * np.pi, 100)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return x, y, z

    def _create_sphere(self):
        "This function constructs an empty Bloch sphere."

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

    # -----States and Vectors-----
    def _coordinates(self, state):
        "This function determines the coordinates of a qubit in the sphere."
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

    def _homogeneous(self, vector):
        "Helper method to `_broadcasting_semantics()`."
        if len(vector.shape) == 1:
            return [vector]
        elif len(vector.shape) == 2:
            return vector
        else:
            raise_error(ValueError, "Only `2D` or `1D` np.ndarray / list is accepted.")

    def _broadcasting_semantics(self, vector, mode, color):
        "This function makes sure that `vector`, `mode`, `color` have the same sizes."
        if isinstance(vector, list):
            vector = np.array(vector)

        vector = self._homogeneous(vector)

        num_vector = len(vector)
        if isinstance(mode, str):
            mode = [mode] * num_vector

        if isinstance(color, str):
            color = [color] * num_vector

        return vector, mode, color

    def add_vector(
        self,
        vector: ArrayLike,
        mode: Union[str, list[str]] = "vector",
        color: Union[str, list[str]] = "black",
    ):
        """This function adds a vector to the sphere."""

        vectors, modes, colors = self._broadcasting_semantics(vector, mode, color)
        for vector, color, mode in zip(vectors, colors, modes):
            if mode == "vector":
                self._vectors.append(vector)
                self._color_vectors.append(color)
            elif mode == "point":
                self._points.append(vector)
                self._color_points.append(color)
            else:
                raise_error(ValueError, "Mode not supported. Try: `point` or `vector`.")

    def add_state(
        self,
        state: ArrayLike,
        mode: Union[str, list[str]] = "vector",
        color: Union[str, list[str]] = "black",
    ):
        "This function adds a state to the sphere."

        vectors, modes, colors = self._broadcasting_semantics(state, mode, color)
        for vector, color, mode in zip(vectors, colors, modes):
            x, y, z = self._coordinates(vector)
            if mode == "vector":
                self._vectors.append(np.array([x, y, z]))
                self._color_vectors.append(color)
            elif mode == "point":
                self._points.append(np.array([x, y, z]))
                self._color_points.append(color)
            else:
                raise_error(ValueError, "Mode not supported. Try: `point` or `vector`.")

    def clear(self):
        """This function clears the sphere."""

        self._new_window()

        # Clear data
        self._points = []
        self._vectors = []

        # Clear Color
        self._color_points = []
        self._color_vectors = []

    # ----Plot and Save-----
    def _rendering(self):
        """This function creates the empty sphere and plots the vectors and points on it."""

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

    def _qt_window(self):
        "Helper method to `plot()` for the `qtagg` backend."
        manager = self._backend.new_figure_manager_given_figure(1, self.fig)
        manager.show()
        manager.set_window_title("Bloch Sphere")
        self._backend.Show().mainloop()

    def _tk_window(self):
        "Helper method to `plot()` for the `tkagg` backend."

        root = tk.Tk()
        root.title("Bloch Sphere")
        # Frame
        frame = tk.Frame(root)

        # Canvas widget
        canvas = FigureCanvasTkAgg(self.fig, master=root)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()
        root.mainloop()

    def _jupyter_window(self):
        "Helper method to `plot()` for the `jupyter` backend."
        self._new_window()
        self._rendering()
        display(self.fig)

    def plot(self):
        "This function plots the sphere."
        self._rendering()

        if self.backend == "tkagg":
            self._tk_window()
        elif self.backend == "qtagg":
            self._qt_window()
        elif self.backend == "jupyter":
            self._jupyter_window()

    def save(self, filename="bloch_sphere.pdf"):
        "This function saves the sphere."
        self._rendering()

        mpl.use("Agg")
        canvas = backend_agg.FigureCanvas(self.fig)
        self.fig.savefig(filename)
