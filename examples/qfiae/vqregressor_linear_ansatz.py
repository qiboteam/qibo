import matplotlib.pyplot as plt
import numpy as np

from qibo import Circuit, gates


class VQRegressor_linear_ansatz:
    def __init__(self, layers, ndata, function, xmax=np.pi, xmin=-np.pi):
        """
        This class implements an Adam Descent optimization performed on a
        1-qubit Variational Quantum Circuit defined by the linear_ansatz function.
        Args:
            layers: integer value representing the number of layers
            ndata: integer value representing the training set's cardinality
            function: target function f(x)
            xmax: upper limit in the x variable
            xmin: lower limit in the x variable
        """
        self.xmin = xmin
        self.xmax = xmax
        self.nqubits = 1
        self.layers = layers
        self.f = function
        self.params = np.random.randn(3 * layers + 3).astype("float64")
        self.nparams = (layers + 1) * 3
        self.features, self.labels, self.norm = self.prepare_training_set(ndata)
        self.nsample = len(self.labels)

    def S(self, circuit, x):
        """Data encoding circuit block."""

        circuit.add(gates.RZ(q=0, theta=x))

    def A(self, circuit, theta_vec):
        """Trainable circuit block."""

        circuit.add(gates.RZ(q=0, theta=theta_vec[0]))
        circuit.add(gates.RY(q=0, theta=theta_vec[1]))
        circuit.add(gates.RZ(q=0, theta=theta_vec[2]))

    def linear_ansatz(self, weights, x=None):
        """Variational Quantum Circuit of the Quantum Neural Network that performs the fit of f(x)."""

        c = Circuit(self.nqubits)
        # First unitary, Layer 0
        self.A(c, weights[0:3])
        # Adding different layers
        for i in range(3, len(weights), 3):
            self.S(c, x)
            self.A(c, weights[i : i + 3])
        c.add(gates.M(0))
        return c

    def label_points(self, x):
        """
        Function which implement the target function
        Args:
            x: np.float64 array of input variables
        Returns: np.float64 array of output variables
        """

        y = self.f(x)
        # We normalize the labels
        ymax = np.max(np.abs(y))
        y = y / ymax
        return y, ymax

    def prepare_training_set(self, n_sample):
        """
        This function picks a random sample from the uniform U[xmin,xmax]
        Args:
            n_sample: integer desired dataset cardinality
        Returns: the features, the labels and the normalization constant
        """

        x = np.random.uniform(self.xmin, self.xmax, n_sample)
        labels, norm = self.label_points(x)
        return x, labels, norm

    def show_predictions(self, title, save, features=None):
        """
        This function shows the VQR predictions on the training dataset
        Args:
            title: string title of the plot
            save:  boolean variable; pick True if you want to save the image as title.png
                   pick False if you don't want to save it
            features: np.matrix in the form (2**nqubits, n_sample) on which you desire to perform
                   the predictions. Default = None and it takes the training dataset
        """

        if features is None:
            features = self.features

        labels, norm = self.label_points(features)

        predictions = self.predict_sample(features)

        plt.figure(figsize=(15, 7))
        plt.title(title)
        plt.scatter(
            features,
            predictions * norm,
            label="Predicted",
            s=100,
            color="blue",
            alpha=0.65,
        )
        plt.scatter(
            features,
            labels * norm,
            label="Original",
            s=100,
            color="red",
            alpha=0.65,
        )
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.tight_layout()
        if save is True:
            plt.savefig(title + ".pdf")
            plt.close()
        plt.show()

    def set_parameters(self, new_params):
        """
        Function which set the new parameters into the circuit
        Args:
            new_params: np.array of the new parameters; it has to be (3 * nlayers) long
        """

        self.params = new_params

    def one_prediction(self, this_feature):
        """
        This function executes one prediction
        Args:
            this_feature: np.float64 array 2**nqubits long, containing a specific feature
        Returns: circuit's prediction of the output variable, evaluated as difference of probabilities
        """

        circuit = self.linear_ansatz(self.params, this_feature)
        results = circuit().probabilities(qubits=[0])
        res = results[0] - results[1]
        return res

    def predict_sample(self, features=None):
        """
        This function calculates the predictions related to a specific sample
        Args:
            features: np.matrix containing the N states; each state must be prepared (with
                      the opportune self.prepare_states function as an array with dim 2**nqubits)
        Returns: np.array of the predictions
        """

        if features is None:
            features = self.features

        predictions = np.zeros(len(features))
        for i in range(len(features)):
            predictions[i] = self.one_prediction(features[i])

        return predictions

    def one_target_loss(self, this_feature, this_label):
        """
        Evaluation of the loss function for a single feature knowing its label
        Args:
            this_feature: the feature in form of an 2**nqubits-dim np.array
            this_label: the associated label
        Returns: one target loss function's value
        """

        this_prediction = self.one_prediction(this_feature)
        cf = (this_prediction - this_label) ** 2
        return cf

    def loss(self, params, features=None, labels=None):
        """
        Evaluation of the total loss function
        Args:
            params: np.array of the params which define the circuit
            features: np.matrix containig the n_sample-long vector of states
            labels: np.array of the labels related to features
        Returns: loss function evaluated by summing contributes of each data
        """

        if params is None:
            params = self.params

        if features is None:
            features = self.features

        if labels is None:
            labels = self.labels

        self.set_parameters(params)
        cf = 0
        for feat, lab in zip(features, labels):
            cf = cf + self.one_target_loss(feat, lab)
        cf = cf / len(labels)
        return cf

    def dloss(self, features, labels):
        """
        This function calculates the loss function's gradients with respect to self.params
        Args:
            features: np.matrix containig the n_sample-long vector of states
            labels: np.array of the labels related to features
        Returns: np.array of length self.nparams containing the loss function's gradients
        """

        loss_gradients = np.zeros(self.nparams)
        loss = 0

        for feat, label in zip(features, labels):
            prediction_evaluation = self.one_prediction(feat)
            loss += (label - prediction_evaluation) ** 2
            obs_gradients = self.parameter_shift(feat)
            for i in range(self.nparams):
                loss_gradients[i] += (2 * prediction_evaluation * obs_gradients[i]) - (
                    2 * label * obs_gradients[i]
                )

        return loss_gradients, (loss / len(features))

    def shift_a_parameter(self, i, this_feature):
        """
        Parameter shift's execution on a single parameter
        Args:
            i: integer index which identify the parameter into self.params
            this_feature: np.array 2**nqubits-long containing the state vector assciated to a data
        Returns: derivative of the observable (here the prediction) with respect to self.params[i]
        """

        original = self.params.copy()
        shifted = self.params.copy()

        shifted[i] += np.pi / 2
        self.set_parameters(shifted)
        forward = self.one_prediction(this_feature)

        shifted[i] -= np.pi
        self.set_parameters(shifted)
        backward = self.one_prediction(this_feature)

        self.set_parameters(original)
        result = 0.5 * (forward - backward)

        return result

    def parameter_shift(self, this_feature):
        """
        Full parameter-shift rule's implementation
        Args:
            this_feature: np.array 2**nqubits-long containing the state vector assciated to a data
        Returns: np.array of the observable's gradients with respect to the variational parameters
        """

        obs_gradients = np.zeros(self.nparams, dtype=np.float64)
        for ipar in range(self.nparams - 1):
            obs_gradients[ipar] = self.shift_a_parameter(ipar, this_feature)
        return obs_gradients

    def AdamDescent(
        self,
        learning_rate,
        m,
        v,
        features,
        labels,
        iteration,
        beta_1=0.85,
        beta_2=0.99,
        epsilon=1e-8,
    ):
        """
        Implementation of the Adam optimizer: during a run of this function parameters are updated.
        Furthermore, new values of m and v are calculated.
        Args:
            learning_rate: np.float value of the learning rate
            m: momentum's value before the execution of the Adam descent
            v: velocity's value before the execution of the Adam descent
            features: np.matrix containig the n_sample-long vector of states
            labels: np.array of the labels related to features
            iteration: np.integer value corresponding to the current training iteration
            beta_1: np.float value of the Adam's beta_1 parameter; default 0.85
            beta_2: np.float value of the Adam's beta_2 parameter; default 0.99
            epsilon: np.float value of the Adam's epsilon parameter; default 1e-8
        Returns: np.float new values of momentum and velocity
        """

        grads, loss = self.dloss(features, labels)

        for i in range(self.nparams):
            m[i] = beta_1 * m[i] + (1 - beta_1) * grads[i]
            v[i] = beta_2 * v[i] + (1 - beta_2) * grads[i] * grads[i]
            mhat = m[i] / (1.0 - beta_1 ** (iteration + 1))
            vhat = v[i] / (1.0 - beta_2 ** (iteration + 1))
            self.params[i] -= learning_rate * mhat / (np.sqrt(vhat) + epsilon)

        return m, v, loss

    def train_with_psr(self, epochs, learning_rate, batches, J_treshold):
        """
        This function performs the full Adam descent's procedure
        Args:
            epochs: np.integer value corresponding to the epochs of training
            learning_rate: np.float value of the learning rate
            batches: np.integer value of the number of batches which divide the dataset
            J_treshold: np.float value of the desired loss function's treshold
        Returns: list of loss values, one for each epoch
        """

        losses = []
        indices = []

        # create index list
        idx = np.arange(0, self.nsample)

        m = np.zeros(self.nparams)
        v = np.zeros(self.nparams)

        # create index blocks on which we run
        for ib in range(batches):
            indices.append(np.arange(ib, self.nsample, batches))

        iteration = 0

        for epoch in range(epochs):
            if epoch != 0 and losses[-1] < J_treshold:
                print(
                    "Desired sensibility is reached, here we stop: ",
                    iteration,
                    " iteration",
                )
                break
            # shuffle index list
            np.random.shuffle(idx)
            # run over the batches
            for ib in range(batches):
                iteration += 1

                features = self.features[idx[indices[ib]]]
                labels = self.labels[idx[indices[ib]]]
                # update parameters
                m, v, this_loss = self.AdamDescent(
                    learning_rate, m, v, features, labels, iteration
                )
                # track the training
                print(
                    "Iteration ",
                    iteration,
                    " epoch ",
                    epoch + 1,
                    " | loss: ",
                    this_loss,
                )
                # in case one wants to plot J in function of the iterations
                losses.append(this_loss)

        return
