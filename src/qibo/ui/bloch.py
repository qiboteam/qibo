import matplotlib.pyplot as plt
import numpy as np

from typing import Union
from logging import warning

from dataclasses import dataclass
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch

from qibo import hamiltonians
from qibo.symbols import X, Y, Z
from qibo.config import raise_error


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


class Bloch:
    def __init__(self, state=None, vector=None, point=None):
        # Size and style
        self.figsize = [5, 5]
        self.fontsize = 18
        self.arrow_style = "-|>"
        self.arrow_width = 2.0
        self.mutation_scale = 20
        self.linewidth = 0.9
        self.linecolor = "#383838"
        self.main_alpha = 0.6
        self.secondary_alpha = 0.2

        # Figure and axis
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection="3d", elev=30, azim=30)

        # Bool variable
        self._shown = False

        # Data
        self.points = []
        self.vectors = []

        # Color data
        self.color_points = []
        self.color_vectors = []

    def normalize_input(self, vectors, modes, colors):
        vectors, length_vectors = self._normalize_vectors(vectors)
        modes = self._normalize_modes_colors(modes, length_vectors, "modes")
        colors = self._normalize_modes_colors(colors, length_vectors, "colors")

        return vectors, modes, colors, length_vectors

    def _normalize_modes_colors(self, element, length_states, option):
        length_option = 0
        if isinstance(element, np.ndarray):
            if element.ndim == 1:
                element = [element[0]]
                length_option = len(element)
            elif element.ndim == 2:
                length_option = len(element)
            else:
                raise_error(
                    ValueError,
                    "`" + option + "` must be 1D or 2D `np.ndarray`, `list`, `str`.",
                )
        elif isinstance(element, list):
            length_option = len(element)
        elif isinstance(element, str):
            element = [element]
            length_option = len(element)
        else:
            raise_error(
                ValueError,
                f"Unsupported type for `"
                + option
                + "`. Types supported: `np.ndarray`, `list`, `str`.",
            )

        if length_states > length_option:
            element = self._mismatch(element, length_states, length_option, option)

        return element

    def _normalize_vectors(self, element):
        length = 0
        if isinstance(element, np.ndarray):
            if element.ndim == 1:
                element = [element]
                length = len(element)
            elif element.ndim == 2:
                length = len(element)
            else:
                raise_error(
                    ValueError,
                    "`state` must be 1D or 2D `np.ndarray` or `list`.",
                )
        elif isinstance(element, list):
            length = len(element)
        else:
            raise_error(
                ValueError,
                f"Unsupported type for `state`. Types supported: `np.ndarray` or `list`.",
            )

        return element, length

    def _mismatch(self, element, length_states, length_option, option):

        option_list = 0
        if option == "colors":
            option_list = ["black"]
        elif option == "modes":
            option_list = ["vector"]

        element = [element[0]] + option_list * (length_states - length_option)
        warning(
            f"Mismatch between number of states ({length_states}) and colors ({length_option}). Defaulting missing "
            + option
            + " to '"
            + option_list[0]
            + "' to match the number of states."
        )
        return element

    def _check_normalisation(self, element):
        if len(element) == 2:
            norm = np.linalg.norm(element)
            if not np.isclose(norm, 1):
                raise_error(ValueError, "Unnormalized state detected.")
        elif len(element) == 3:
            norm = np.linalg.norm(element)
            if not np.isclose(norm, 1.0):
                raise_error(ValueError, "The vector does not lie on the Bloch sphere.")
        else:
            raise_error(
                ValueError,
                "States must have two components. Vectors must have three components.",
            )

    def clear(self):
        # Data
        self.states = []
        self.points = []
        self.vectors = []

        # Color data
        self.color_states = []
        self.color_points = []
        self.color_vectors = []

    def create_sphere(self):
        "Function to create an empty sphere."

        # Empty sphere
        phi, theta = np.mgrid[0.0 : np.pi : 100j, 0.0 : 2.0 * np.pi : 100j]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        self.ax.plot_surface(x, y, z, color="lavenderblush", alpha=0.2)

        # ----Circular curves over the surface----
        # Axis
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.zeros(100)
        x = np.sin(theta)
        y = np.cos(theta)
        self.ax.plot(
            x,
            y,
            z,
            color=self.linecolor,
            alpha=self.main_alpha,
            linewidth=self.linewidth,
        )
        self.ax.plot(
            z,
            x,
            y,
            color=self.linecolor,
            alpha=self.main_alpha,
            linewidth=self.linewidth,
        )
        self.ax.plot(
            y,
            z,
            x,
            color=self.linecolor,
            alpha=self.main_alpha,
            linewidth=self.linewidth,
        )

        # Latitude
        z1 = np.full(100, 0.4)
        r1 = np.sqrt(1 - z1[0] ** 2)
        x1 = r1 * np.cos(theta)
        y1 = r1 * np.sin(theta)
        self.ax.plot(
            x1,
            y1,
            z1,
            color=self.linecolor,
            alpha=self.secondary_alpha,
            linewidth=self.linewidth,
        )

        z1 = np.full(100, 0.9)
        r1 = np.sqrt(1 - z1[0] ** 2)
        x1 = r1 * np.cos(theta)
        y1 = r1 * np.sin(theta)
        self.ax.plot(
            x1,
            y1,
            z1,
            color=self.linecolor,
            alpha=self.secondary_alpha,
            linewidth=self.linewidth,
        )

        z2 = np.full(100, -0.9)
        r2 = np.sqrt(1 - z2[0] ** 2)
        x2 = r2 * np.cos(theta)
        y2 = r2 * np.sin(theta)
        self.ax.plot(
            x2,
            y2,
            z2,
            color=self.linecolor,
            alpha=self.secondary_alpha,
            linewidth=self.linewidth,
        )

        # Longitude
        phi_list = np.linspace(0, 2 * np.pi, 6)
        theta = np.linspace(0, 2 * np.pi, 100)

        for phi in phi_list:
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            self.ax.plot(
                x,
                y,
                z,
                color=self.linecolor,
                alpha=self.secondary_alpha,
                linewidth=self.linewidth,
            )

        # ----Axis lines----
        line = np.linspace(-1, 1, 100)
        zeros = np.zeros_like(line)

        self.ax.plot(
            line,
            zeros,
            zeros,
            color=self.linecolor,
            alpha=self.main_alpha,
            linewidth=self.linewidth,
        )
        self.ax.plot(
            zeros,
            line,
            zeros,
            color=self.linecolor,
            alpha=self.main_alpha,
            linewidth=self.linewidth,
        )
        self.ax.plot(
            zeros,
            zeros,
            line,
            color=self.linecolor,
            alpha=self.main_alpha,
            linewidth=self.linewidth,
        )

        self.ax.text(1.2, 0, 0, "x", color="black", fontsize=self.fontsize, ha="center")
        self.ax.text(0, 1.2, 0, "y", color="black", fontsize=self.fontsize, ha="center")
        self.ax.text(
            0,
            0,
            1.2,
            r"$|0\rangle$",
            color="black",
            fontsize=self.fontsize,
            ha="center",
        )
        self.ax.text(
            0,
            0,
            -1.3,
            r"$|1\rangle$",
            color="black",
            fontsize=self.fontsize,
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

        vectors, modes, colors, lenght_state = self.normalize_input(vector, mode, color)

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

        vectors, modes, colors, lenght_vectors = self.normalize_input(
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
                lw=self.arrow_width,
                arrowstyle=self.arrow_style,
                mutation_scale=self.mutation_scale,
                color=color,
            )
            self.ax.add_artist(a)

        for color, point in zip(self.color_points, self.points):
            self.ax.scatter(point[0], point[1], point[2], color=color, s=10)

    def plot(self, save=False, filename="bloch_sphere.pdf"):
        self.rendering()
        self.ax.set_aspect("equal")
        plt.axis("off")
        plt.tight_layout()
        if save:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
