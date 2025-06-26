import numpy as np
import matplotlib as mpl

mpl.use("Qt5Agg")

from dataclasses import dataclass, field

from matplotlib.figure import Figure
from matplotlib.backend_bases import FigureManagerBase
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d

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

    STYLE = {
        "figure.figsize": (5, 5),
        "legend.fontsize": 18,
        "axes.titlesize": 18,
        "lines.linewidth": 0.9,
        "lines.color": "#383838",
    }

    STYLE_TEXT = {"text.color": "black", "font.size": 18}

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
        self.ax = self.fig.add_subplot(111, projection="3d", elev=30, azim=30)

    def clear(self):
        # Figure
        self.fig.clear()

        # Data
        self.points = []
        self.vectors = []

        # Color data
        self.color_points = []
        self.color_vectors = []

    def create_sphere(self):
        "Function to create an empty sphere."

        # Empty sphere
        phi, theta = np.mgrid[0.0 : np.pi : 100j, 0.0 : 2.0 * np.pi : 100j]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        with mpl.rc_context(self.STYLE):
            self.ax.plot_surface(x, y, z)

        # ----Circular curves over the surface----
        # Axis
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.zeros(100)
        x = np.sin(theta)
        y = np.cos(theta)
        with mpl.rc_context(self.STYLE):
            self.ax.plot(x, y, z)
            self.ax.plot(z, x, y)
            self.ax.plot(y, z, x)

        # Latitude
        z1 = np.full(100, 0.4)
        r1 = np.sqrt(1 - z1[0] ** 2)
        x1 = r1 * np.cos(theta)
        y1 = r1 * np.sin(theta)
        with mpl.rc_context(self.STYLE):
            self.ax.plot(x1, y1, z1)

        z1 = np.full(100, 0.9)
        r1 = np.sqrt(1 - z1[0] ** 2)
        x1 = r1 * np.cos(theta)
        y1 = r1 * np.sin(theta)
        with mpl.rc_context(self.STYLE):
            self.ax.plot(
                x1,
                y1,
                z1,
                alpha=self.secondary_alpha,
            )

        z2 = np.full(100, -0.9)
        r2 = np.sqrt(1 - z2[0] ** 2)
        x2 = r2 * np.cos(theta)
        y2 = r2 * np.sin(theta)
        with mpl.rc_context(self.STYLE):
            self.ax.plot(
                x2,
                y2,
                z2,
                alpha=self.secondary_alpha,
            )

        # Longitude
        phi_list = np.linspace(0, 2 * np.pi, 6)
        theta = np.linspace(0, 2 * np.pi, 100)

        for phi in phi_list:
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            with mpl.rc_context(self.STYLE):
                self.ax.plot(
                    x,
                    y,
                    z,
                    alpha=self.secondary_alpha,
                )

        # ----Axis lines----
        line = np.linspace(-1, 1, 100)
        zeros = np.zeros_like(line)

        with mpl.rc_context(self.STYLE):
            self.ax.plot(
                line,
                zeros,
                zeros,
                alpha=self.main_alpha,
            )
            self.ax.plot(
                zeros,
                line,
                zeros,
                alpha=self.main_alpha,
            )
            self.ax.plot(
                zeros,
                zeros,
                line,
            )

        with mpl.rc_context(self.STYLE_TEXT):
            self.ax.text(1.2, 0, 0, "x", ha="center")
            self.ax.text(0, 1.2, 0, "y", ha="center")
            self.ax.text(
                0,
                0,
                1.2,
                r"$|0\rangle$",
                ha="center",
            )
            self.ax.text(
                0,
                0,
                -1.3,
                r"$|1\rangle$",
                ha="center",
            )

        self.ax.set_xlim([-0.7, 0.7])
        self.ax.set_ylim([-0.7, 0.7])
        self.ax.set_zlim([-0.7, 0.7])

    def coordinates(self, state):
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

    def add_vector(self, vector, mode="vector", color="black"):

        vectors, modes, colors, lenght_state = self._normalize_input(
            vector, mode, color
        )

        for vector, color, mode in zip(vectors, colors, modes):
            self._check_normalisation(vector)
            if mode == "vector":
                if len(vector) == 2:
                    x, y, z = self.coordinates(vector)
                    self.vectors.append(np.array([x, y, z]))
                    self.color_vectors.append(color)
                else:
                    self.vectors.append(vector)
                    self.color_vectors.append(color)
            else:
                if len(vector) == 2:
                    x, y, z = self.coordinates(vector)
                    self.points.append(np.array([x, y, z]))
                    self.color_points.append(color)
                else:
                    self.points.append(vector)
                    self.color_points.append(color)

    def add_state(self, state, mode="vector", color="black"):
        "Function to add a state to the sphere."

        vectors, modes, colors, lenght_vectors = self._normalize_input(
            state, mode, color
        )

        for vector, color, mode in zip(vectors, colors, modes):
            self._check_normalisation(vector)
            x, y, z = self.coordinates(vector)
            if mode == "vector":
                self.vectors.append(np.array([x, y, z]))
                self.color_vectors.append(color)
            else:
                self.points.append(np.array([x, y, z]))
                self.color_points.append(color)

    def rendering(self):
        if self._shown == True:
            plt.close(self.fig)
            self.fig = plt.figure(figsize=self.figsize)
            self.ax = self.fig.add_subplot(111, projection="3d", elev=30, azim=30)

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

    def _view(self):
        self.rendering()
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.fig.tight_layout()

    def save(self, filename="bloch_sphere.pdf"):
        self._view()
        self.fig.savefig(filename)

    def plot(self):
        self._view()
        # breakpoint()
        # FigureManagerBase.pyplot_show(block=False)
        self.fig.show(warn=True)
