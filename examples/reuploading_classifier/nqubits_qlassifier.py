import os

import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

from qibo import Circuit, gates
from qibo.examples.reuploading_classifier.datasets import (
    create_dataset,
    create_target_n,
    fig_template_3D,
)
from qibo.examples.reuploading_classifier.qlassifier import (
    fidelity,
    single_qubit_classifier,
)


class n_qubit_classifier(single_qubit_classifier):
    def __init__(
        self,
        name,
        layers,
        nqubits,
        entanglement=False,
        data_dimension=2,
        target_naive=True,
        grid=11,
        test_samples=1000,
        seed=0,
    ):
        """Class with all computations needed for classification.

        Args:
            name (str): Name of the problem to create the dataset, to choose between
                ['circle', '3 circles', 'square', '4 squares', 'crown', 'tricrown', 'wavy lines'].
            layers (int): Number of layers to use in the classifier.
            grid (int): Number of points in one direction defining the grid of points.
                If not specified, the dataset does not follow a regular grid.
            samples (int): Number of points in the set, randomly located.
                This argument is ignored if grid is specified.
            seed (int): Random seed.

        Returns:
            Dataset for the given problem (x, y).
        """
        np.random.seed(seed)

        self.name = name
        self.nqubits = nqubits
        self.data_dimension = data_dimension
        self.entanglement = entanglement
        self.layers = layers
        self.target_naive = target_naive

        self.training_set = create_dataset(
            name, dimensions=self.data_dimension, grid=grid
        )
        self.test_set = create_dataset(
            name, dimensions=self.data_dimension, samples=test_samples
        )

        self.params = np.random.randn(layers * 2 * self.data_dimension * self.nqubits)
        self.target = create_target_n(name, nqubits, target_naive)
        self._circuit = self._initialize_circuit()
        try:
            os.makedirs("results/" + self.name + "/%s_layers" % self.layers)
        except:
            pass

    def _initialize_circuit(self):
        """Creates variational circuit."""
        C = Circuit(self.nqubits)

        nqubits = self.nqubits
        layers = {}

        # TODO: Merge both
        # For even
        if np.mod(nqubits, 2) == 0:
            for i in range(0, nqubits, 2):
                layers[i] = []
                for j in range(nqubits // 2):
                    control = j * 2
                    target = np.mod(j * 2 + 1 + i, nqubits)
                    if control < target:
                        layers[i].append((control, target))
                    elif control > target:
                        layers[i].append((target, control))

        # For odd we do the same removing the extra qubit
        if np.mod(nqubits, 2) == 1:
            nqubits += 1
            for i in range(0, nqubits, 2):
                layers[i] = []
                for j in range(nqubits // 2):
                    control = j * 2
                    target = np.mod(j * 2 + 1 + i, nqubits)
                    if control == nqubits - 1:
                        next
                    elif target == nqubits - 1:
                        next
                    elif control < target:
                        layers[i].append((control, target))
                    elif control > target:
                        layers[i].append((target, control))

            nqubits -= 1

        for i in range(self.layers):
            # Single qubit gates:
            for q in range(self.nqubits):
                for j in range(self.data_dimension):
                    if np.mod(j, 2) == 0:
                        C.add(gates.RY(q, theta=0))
                    else:
                        C.add(gates.RZ(q, theta=0))

            # TODO: Do we care abot who is control and target ? (Swap them by layers?)
            if self.entanglement and i != self.layers - 1:
                layer = layers[np.mod(i, len(layers))]
                for qubits in layer:
                    C.add(gates.CZ(qubits[0], qubits[1]))

        return C

    def circuit(self, x):
        """Method creating the circuit for a point (in the datasets).

        Args:
            x (array): Point to create the circuit.

        Returns:
            Qibo circuit.
        """
        params = {}
        layer_parameters = self.data_dimension * 2 * self.layers
        # TODO: Check how the circuit assings params
        for k, q in enumerate(range(self.nqubits)):
            params[q] = []
            for i in range(
                0, self.data_dimension * 2 * self.layers, self.data_dimension * 2
            ):
                for j in range(len(x)):
                    params[q].append(
                        self.params[layer_parameters * k + i + 2 * j] * x[j]
                        + self.params[layer_parameters * k + i + 1 + 2 * j]
                    )

        params_aux = list(params.values())
        params_list = []
        for l in params_aux:
            params_list += l
        self._circuit.set_parameters(params_list)

        return self._circuit

    def paint_results_3D(self):
        """Method for plotting the guessed labels and the right guesses.

        Returns:
            plot with results.
        """
        fig, axs = fig_template_3D(self.name)
        guess_labels = self.eval_test_set_fidelity()
        colors_classes = get_cmap("tab10")
        norm_class = Normalize(vmin=0, vmax=10)
        x = self.test_set[0]
        x_0, x_1, x_2 = x[:, 0], x[:, 1], x[:, 2]
        axs[0].scatter(
            x_0, x_1, x_2, c=guess_labels, s=2, cmap=colors_classes, norm=norm_class
        )

        colors_rightwrong = get_cmap("RdYlGn")
        norm_rightwrong = Normalize(vmin=-0.1, vmax=1.1)

        checks = [int(g == l) for g, l in zip(guess_labels, self.test_set[1])]
        axs[1].scatter(
            x_0, x_1, x_2, c=checks, s=2, cmap=colors_rightwrong, norm=norm_rightwrong
        )
        print(
            "The accuracy for this classification is %.2f"
            % (100 * np.sum(checks) / len(checks)),
            "%",
        )

        fig.savefig("results/" + self.name + "/%s_layers/test_set.pdf" % self.layers)
