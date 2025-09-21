from dataclasses import dataclass, field
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from numpy.typing import ArrayLike

from qibo import hamiltonians
from qibo.config import raise_error
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

        Args:
            None.

        Returns:
            float: The minimum z-depth of the projected arrow's points.
            This is used by Matplotlib to determine the drawing order of the objects.
        """
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


@dataclass
class Bloch:
    """This class creates a Bloch sphere."""

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

    # Backend
    backend: str = "tkagg"

    def __post_init__(self):
        # Backend
        mpl.use(self.backend)

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

    # -----Sphere-----
    def _sphere_surface(self):
        """Helper method to `_create_sphere` to construct the sphere's surface

        Args:
            None

        Returns:
            Three arrays containing the cartesian coordinates
            of 100 points used to draw the surface of the sphere.
        """
        phi, theta = np.mgrid[0.0 : np.pi : 100j, 0.0 : 2.0 * np.pi : 100j]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return x, y, z

    def _axis(self):
        """Helper method to `_create_sphere` to construct the sphere's axis

        Args:
            None

        Returns:
            Three arrays containing the cartesian coordinates
            of 100 points used to draw the axis of the sphere.
        """
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.zeros(100)
        x = np.sin(theta)
        y = np.cos(theta)
        return x, y, z

    def _latitude(self, z):
        """Helper method to `_create_sphere` to construct the sphere's latitudes
        Args:
            z (float): latitude at which the circle will be drawn

        Returns:
            Three arrays containing the cartesian coordinates
            of 100 points used to draw the latitudes of the sphere.
        """
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.full(100, z)
        r = np.sqrt(1 - z[0] ** 2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y, z

    def _meridian(self, phi):
        """Helper method to `_create_sphere` to construct the sphere's meridians
        Args:
            phi (float): angle at which the meridian will be drawn.

        Returns:
            Three arrays containing the cartesian coordinates
            of 100 points used to draw the meridians of the sphere.
        """
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
        """This function determines the coordinates of a qubit in the sphere.

        Args:
            state (np.ndarray): quantum state.

        Returns:
            The coordinates of the state on the sphere.
        """
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
        """Helper method to `_broadcasting_semantics()`."

        Args:
            vector (np.ndarray / list): quantum state.

        Returns:
            None
        """
        if len(vector.shape) == 1:
            return [vector]
        elif len(vector.shape) == 2:
            return vector
        else:
            raise_error(ValueError, "Only `2D` or `1D` np.ndarray / list is accepted.")

    def _broadcasting_semantics(self, vector, mode, color):
        """This function makes sure that `vector`, `mode`, `color` have the same sizes."

        Args:
            vector (np.ndarray): quantum state.
            mode (str): the two possible representations on the
                sphere: "point" or "vector".
            color (str): the color that the "point"/"vector"
                will have on the sphere.

        Returns:
            Three array of the same dimensions.
        """
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
        """This function adds a vector to the sphere.

        Args:
            vector (ArrayLike): quantum state with three cordinates.
            mode (str): the two possible representations on the
                sphere: "point" or "vector".
            color (str): the color that the "point"/"vector"
                will have on the sphere.

        Returns:
            None
        """

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
        """This function adds a state to the sphere.

        Args:
            state (ArrayLike): 2D vector.
            mode (str): the two possible representations on the
                sphere: "point" or "vector".
            color (str): the color that the "point"/"vector"
                will have on the sphere.

        Returns:
            None
        """

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
        """This function clears the sphere.

        Args:
            None

        Returns:
            None
        """
        plt.close()
        self._new_window()

        # Clear data
        self._points = []
        self._vectors = []

        # Clear Color
        self._color_points = []
        self._color_vectors = []

    def render(self):
        """This function creates the empty sphere and plots the
        vectors and points on it.

        Args:
            None

        Returns:
            None
        """

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
