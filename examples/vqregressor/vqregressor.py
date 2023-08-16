import matplotlib.pyplot as plt
import numpy as np

from qibo import Circuit, gates

# Here we use the default numpy backend


class VQRegressor:
    def __init__(self, layers, ndata, states=None):
        """
        This class implement an Adam Descent optimization performed on a
        1-qubit Variational Quantum Circuit like this:
        H --> [RY, H, RZ] --> ... --> [RY, H, RZ]
        where each couple of squared parenthesis represents a layer.
        Args:
            layers: integer value representing the number of layers
            ndata: integer value representing the training set's cardinality
            states: (2, N)-dim np.matrix containing the states on which we perform the training.
                    You can prepare a state sample generating some point in [-1,1] and submitting them to
                    VQRegressor.prepare_states()
        """

        self.nqubits = 1
        self.layers = layers
        self.params = np.random.randn(3 * layers).astype("float64")
        self.nparams = len(self.params)
        self.features, self.labels = self.prepare_training_set(ndata, states)
        self.nsample = len(self.labels)
        self._circuit = self.ansatz(layers)

    def ansatz(self, layers):
        """
        The circuit's ansatz: a sequence of RZ and RY with a beginning H gate
        Args:
            layers: integer, number of layers which compose the circuit
        Returns: abstract qibo circuit
        """
        c = Circuit(self.nqubits)

        c.add(gates.H(q=0))
        for l in range(layers):
            c.add(gates.RY(q=0, theta=0))
            c.add(gates.H(0))
            c.add(gates.RZ(q=0, theta=0))
        c.add(gates.M(0))
        return c

    def label_points(self, x):
        """
        Function which implement the target function
        Args:
            x: np.float64 array of input variables
        Returns: np.float64 array of output variables
        """
        # here you can define the function you want to fit
        y = np.sin(2 * x)

        # that is normalized here
        ymax = np.max(np.abs(y))
        y = y / ymax

        return y

    def prepare_states(self, x):
        """
        State preparation: each data have to be embedded into a 2**nqubits vector state
        Args:
            x: np.array of input data with length N
        Returns: np.matrix (2**nqubits, N) of the states
        """
        N = len(x)
        states = np.zeros((N, 2))
        for i in range(N):
            states[i][0] = x[i]
            states[i][1] = 0.33
        return states

    def prepare_training_set(self, n_sample, states=None):
        """
        This function picks a random sample from the uniform U[0,1]
        Args:
            n_sample: integer desired dataset cardinality
        Returns: np.matrix containing the states and np.array containing the labels
        """
        if states is None:
            x = np.random.uniform(-1, 1, n_sample)
            labels = self.label_points(x)
            states = self.prepare_states(x)
        else:
            states = states
            labels = self.label_points(states.T[0])
        return states, labels

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

        labels = self.label_points(features.T[0])

        predictions = self.predict_sample(features)

        plt.figure(figsize=(15, 7))
        plt.title(title)
        plt.scatter(
            features.T[0],
            predictions,
            label="Predicted",
            s=100,
            color="blue",
            alpha=0.65,
        )
        plt.scatter(
            features.T[0], labels, label="Original", s=100, color="red", alpha=0.65
        )
        plt.xlabel("x")
        plt.ylabel("y")
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

    def get_parameters(self):
        return self.params

    def circuit(self, feature):
        """
        This function prepare the circuit implementing the data re-uploading strategy. In this case
        we choose to implement RY-RZ layers, depending on 3 parameters. The data affect the first
        parameter in each layer.
        Args:
            feature: 2**nqubits dimensional np.array containing a specific feature
        Returns: modified qibo circuit according to re-uploading strategy
        """
        params = []
        for i in range(0, 3 * self.layers, 3):
            params.append(self.params[i] * feature[0] + self.params[i + 1])
            params.append(self.params[i + 2])
        self._circuit.set_parameters(params)
        return self._circuit

    def one_prediction(self, this_feature):
        """
        This function execute one prediction
        Args:
            this_feature: np.float64 array 2**nqubits long, containing a specific feature
        Returns: circuit's prediction of the output variable, evaluated as difference of probabilities
        """
        nshots = 1024
        c = self.circuit(this_feature)
        results = c(nshots=nshots).probabilities(qubits=[0])
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
            prediction_evaluation = self.one_prediction(feat)  #  <B>
            loss += (label - prediction_evaluation) ** 2
            obs_gradients = self.parameter_shift(feat)  # d<B>
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

        # customized parameter shift rule when x contributes to param's definition
        if i % 3 == 0:
            shifted[i] += np.pi / 2 / this_feature[0]

            self.set_parameters(shifted)
            forward = self.one_prediction(this_feature)

            shifted[i] -= np.pi / this_feature[0]

            self.set_parameters(shifted)
            backward = self.one_prediction(this_feature)

            self.set_parameters(original)

            result = 0.5 * (forward - backward) * this_feature[0]

        else:
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
        for ipar in range(self.nparams):
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

        return losses
