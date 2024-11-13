import os

import numpy as np
from datasets import create_dataset, create_target, fig_template, world_map_template
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

import qibo
from qibo import Circuit, gates


class single_qubit_classifier:
    def __init__(self, name, layers, grid=11, test_samples=1000, seed=0):
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
        self.layers = layers
        self.training_set = create_dataset(name, grid=grid)
        self.test_set = create_dataset(name, samples=test_samples)
        self.target = create_target(name)
        self.params = np.random.randn(layers * 4)
        self._circuit = self._initialize_circuit()
        try:
            os.makedirs("results/" + self.name + "/%s_layers" % self.layers)
        except:
            pass

    def set_parameters(self, new_params):
        """Method for updating parameters of the class.

        Args:
            new_params (array): New parameters to update
        """
        self.params = new_params

    def _initialize_circuit(self):
        """Creates variational circuit."""
        C = Circuit(1)
        for l in range(self.layers):
            C.add(gates.RY(0, theta=0))
            C.add(gates.RZ(0, theta=0))
        return C

    def circuit(self, x):
        """Method creating the circuit for a point (in the datasets).

        Args:
            x (array): Point to create the circuit.

        Returns:
            Qibo circuit.
        """
        params = []
        for i in range(0, 4 * self.layers, 4):
            params.append(self.params[i] * x[0] + self.params[i + 1])
            params.append(self.params[i + 2] * x[1] + self.params[i + 3])
        self._circuit.set_parameters(params)
        return self._circuit

    def cost_function_one_point_fidelity(self, x, y):
        """Method for computing the cost function for
        a given sample (in the datasets), using fidelity.

        Args:
            x (array): Point to create the circuit.
            y (int): label of x.

        Returns:
            float with the cost function.
        """
        C = self.circuit(x)
        state = C.execute().state()
        cf = 0.5 * (1 - fidelity(state, self.target[y])) ** 2
        return cf

    def cost_function_fidelity(self, params=None):
        """Method for computing the cost function for the training set, using fidelity.

        Args:
            params(array): new parameters to update before computing

        Returns:
            float with the cost function.
        """
        if params is None:
            params = self.params

        self.set_parameters(params)
        cf = 0
        for x, y in zip(self.training_set[0], self.training_set[1]):
            cf += self.cost_function_one_point_fidelity(x, y)
        cf /= len(self.training_set[0])
        return cf

    def minimize(self, method="BFGS", options=None, compile=True):
        loss = self.cost_function_fidelity

        if method == "cma":
            # Genetic optimizer
            import cma

            r = cma.fmin2(lambda p: loss(p), self.params, 2)
            result = r[1].result.fbest
            parameters = r[1].result.xbest

        elif method == "sgd":
            qibo.set_backend(backend="qiboml", platform="tensorflow")
            tf = qibo.get_backend().tf

            circuit = self.circuit(self.training_set[0])

            sgd_options = {
                "nepochs": 5001,
                "nmessage": 1000,
                "optimizer": "Adamax",
                "learning_rate": 0.5,
            }
            if options is not None:
                sgd_options.update(options)

            # proceed with the training
            vparams = tf.Variable(self.params)
            optimizer = getattr(tf.optimizers, sgd_options["optimizer"])(
                learning_rate=sgd_options["learning_rate"]
            )

            def opt_step():
                with tf.GradientTape() as tape:
                    l = loss(vparams)
                grads = tape.gradient(l, [vparams])
                optimizer.apply_gradients(zip(grads, [vparams]))
                return l, vparams

            if compile:
                opt_step = tf.function(opt_step)

            l_optimal, params_optimal = 10, self.params
            for e in range(sgd_options["nepochs"]):
                l, vparams = opt_step()
                if l < l_optimal:
                    l_optimal, params_optimal = l, vparams
                if e % sgd_options["nmessage"] == 0:
                    print("ite %d : loss %f" % (e, l))

            result = np.array(self.cost_function(params_optimal))
            parameters = np.array(params_optimal)

        else:
            import numpy as np
            from scipy.optimize import minimize

            m = minimize(lambda p: loss(p), self.params, method=method, options=options)
            result = m.fun
            parameters = m.x

        return result, parameters

    def eval_test_set_fidelity(self):
        """Method for evaluating points in the training set, using fidelity.

        Returns:
            list of guesses.
        """
        labels = [[0]] * len(self.test_set[0])
        for j, x in enumerate(self.test_set[0]):
            C = self.circuit(x)
            state = C.execute().state()
            fids = np.empty(len(self.target))
            for i, t in enumerate(self.target):
                fids[i] = fidelity(state, t)
            labels[j] = np.argmax(fids)

        return labels

    def paint_results(self):
        """Method for plotting the guessed labels and the right guesses.

        Returns:
            plot with results.
        """
        fig, axs = fig_template(self.name)
        guess_labels = self.eval_test_set_fidelity()
        colors_classes = get_cmap("tab10")
        norm_class = Normalize(vmin=0, vmax=10)
        x = self.test_set[0]
        x_0, x_1 = x[:, 0], x[:, 1]
        axs[0].scatter(
            x_0, x_1, c=guess_labels, s=2, cmap=colors_classes, norm=norm_class
        )
        colors_rightwrong = get_cmap("RdYlGn")
        norm_rightwrong = Normalize(vmin=-0.1, vmax=1.1)

        checks = [int(g == l) for g, l in zip(guess_labels, self.test_set[1])]
        axs[1].scatter(
            x_0, x_1, c=checks, s=2, cmap=colors_rightwrong, norm=norm_rightwrong
        )
        print(
            "The accuracy for this classification is %.2f"
            % (100 * np.sum(checks) / len(checks)),
            "%",
        )

        fig.savefig("results/" + self.name + "/%s_layers/test_set.pdf" % self.layers)

    def paint_world_map(self):
        """Method for plotting the proper labels on the Bloch sphere.

        Returns:
            plot with 2D representation of Bloch sphere.
        """
        angles = np.zeros((len(self.test_set[0]), 2))
        from datasets import laea_x, laea_y

        fig, ax = world_map_template()
        colors_classes = get_cmap("tab10")
        norm_class = Normalize(vmin=0, vmax=10)
        for i, x in enumerate(self.test_set[0]):
            C = self.circuit(x)
            state = C.execute().state()
            angles[i, 0] = np.pi / 2 - np.arccos(
                np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2
            )
            angles[i, 1] = np.angle(state[1] / state[0])

        ax.scatter(
            laea_x(angles[:, 1], angles[:, 0]),
            laea_y(angles[:, 1], angles[:, 0]),
            c=self.test_set[1],
            cmap=colors_classes,
            s=15,
            norm=norm_class,
        )

        if len(self.target) == 2:
            angles_0 = np.zeros(len(self.target))
            angles_1 = np.zeros(len(self.target))
            angles_0[0] = np.pi / 2
            angles_0[1] = -np.pi / 2
            col = list(range(2))

        elif len(self.target) == 3:
            angles_0 = np.zeros(len(self.target) + 1)
            angles_1 = np.zeros(len(self.target) + 1)
            angles_0[0] = np.pi / 2
            angles_0[1] = -np.pi / 6
            angles_0[2] = -np.pi / 6
            angles_0[3] = -np.pi / 6
            angles_1[2] = np.pi
            angles_1[3] = -np.pi
            col = list(range(3)) + [2]

        else:
            angles_0 = np.zeros(len(self.target))
            angles_1 = np.zeros(len(self.target))
            for i, state in enumerate(self.target):
                angles_0[i] = np.pi / 2 - np.arccos(
                    np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2
                )
                angles_1[i] = np.angle(state[1] / state[0])
            col = list(range(len(self.target)))

        ax.scatter(
            laea_x(angles_1, angles_0),
            laea_y(angles_1, angles_0),
            c=col,
            cmap=colors_classes,
            s=500,
            norm=norm_class,
            marker="P",
            zorder=11,
        )

        ax.axis("off")

        fig.savefig("results/" + self.name + "/%s_layers/world_map.pdf" % self.layers)


def fidelity(state1, state2):
    return np.abs(np.sum(np.conj(state2) * state1)) ** 2
