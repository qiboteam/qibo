import numpy as np
from numpy.random import randn

from qibo import gates, hamiltonians, models
from qibo.backends import matrices
from qibo.config import raise_error


class StyleQGAN:
    """Model that implements and trains a style-based quantum generative adversarial network.

    For original manuscript: `arXiv:2110.06933 <https://arxiv.org/abs/2110.06933>`_

    Args:
        latent_dim (int): number of latent dimensions.
        layers (int): number of layers for the quantum generator. Provide this value only if not using
            a custom quantum generator.
        circuit (:class:`qibo.models.circuit.Circuit`): custom quantum generator circuit. If not provided,
            the default quantum circuit will be used.
        set_parameters (function): function that creates the array of parameters for the quantum generator.
            If not provided, the default function will be used.

    Example:
        .. testcode::

            import numpy as np
            import qibo
            from qibo.models.qgan import StyleQGAN
            # set qibo backend to tensorflow which supports gradient descent training
            qibo.set_backend("tensorflow")
            # Create reference distribution.
            # Example: 3D correlated Gaussian distribution normalized between [-1,1]
            reference_distribution = []
            samples = 10
            mean = [0, 0, 0]
            cov = [[0.5, 0.1, 0.25], [0.1, 0.5, 0.1], [0.25, 0.1, 0.5]]
            x, y, z = np.random.multivariate_normal(mean, cov, samples).T/4
            s1 = np.reshape(x, (samples,1))
            s2 = np.reshape(y, (samples,1))
            s3 = np.reshape(z, (samples,1))
            reference_distribution = np.hstack((s1,s2,s3))
            # Train qGAN with your particular setup
            train_qGAN = StyleQGAN(latent_dim=1, layers=2)
            train_qGAN.fit(reference_distribution, n_epochs=1)
    """

    def __init__(
        self,
        latent_dim,
        layers=None,
        circuit=None,
        set_parameters=None,
        discriminator=None,
    ):
        # qgan works only with tensorflow
        from qibo.backends import TensorflowBackend

        self.backend = TensorflowBackend()

        if layers is not None and circuit is not None:
            raise_error(
                ValueError,
                "Set the number of layers for the default quantum generator "
                "or use a custom quantum generator, do not define both.",
            )
        elif layers is None and circuit is None:
            raise_error(
                ValueError,
                "Set the number of layers for the default quantum generator "
                "or use a custom quantum generator.",
            )

        if set_parameters is None and circuit is not None:
            raise_error(
                ValueError,
                "Set parameters function has to be given for your custom quantum generator.",
            )
        elif set_parameters is not None and circuit is None:
            raise_error(
                ValueError,
                "Define the custom quantum generator to use custom set parameters function.",
            )

        self.discriminator = discriminator
        self.circuit = circuit
        self.layers = layers
        self.latent_dim = latent_dim
        if set_parameters is not None:
            self.set_parameters = set_parameters
        else:
            self.set_parameters = self.set_params

    def define_discriminator(self, alpha=0.2, dropout=0.2):
        """Define the standalone discriminator model."""
        from tensorflow.keras.layers import (  # pylint: disable=E0611,import-error
            Conv2D,
            Dense,
            Dropout,
            Flatten,
            LeakyReLU,
            Reshape,
        )
        from tensorflow.keras.models import (  # pylint: disable=E0611,import-error
            Sequential,
        )
        from tensorflow.keras.optimizers import (  # pylint: disable=E0611,import-error
            Adadelta,
        )

        model = Sequential()
        model.add(Dense(200, use_bias=False, input_dim=self.nqubits))
        model.add(Reshape((10, 10, 2)))
        model.add(
            Conv2D(
                64,
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_initializer="glorot_normal",
            )
        )
        model.add(LeakyReLU(alpha=alpha))
        model.add(
            Conv2D(
                32,
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_initializer="glorot_normal",
            )
        )
        model.add(LeakyReLU(alpha=alpha))
        model.add(
            Conv2D(
                16,
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_initializer="glorot_normal",
            )
        )
        model.add(LeakyReLU(alpha=alpha))
        model.add(
            Conv2D(
                8,
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_initializer="glorot_normal",
            )
        )
        model.add(Flatten())
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation="sigmoid"))

        # compile model
        opt = Adadelta(learning_rate=0.1)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model

    def set_params(self, circuit, params, x_input, i):
        """Set the parameters for the quantum generator circuit."""
        p = []
        index = 0
        noise = 0
        for l in range(self.layers):
            for q in range(self.nqubits):
                p.append(params[index] * x_input[noise][i] + params[index + 1])
                index += 2
                noise = (noise + 1) % self.latent_dim
                p.append(params[index] * x_input[noise][i] + params[index + 1])
                index += 2
                p.append(params[index] * x_input[noise][i] + params[index + 1])
                index += 2
                noise = (noise + 1) % self.latent_dim
                p.append(params[index] * x_input[noise][i] + params[index + 1])
                index += 2
                noise = (noise + 1) % self.latent_dim
            for i in range(0, self.nqubits - 1):
                p.append(params[index] * x_input[noise][i] + params[index + 1])
                index += 2
                noise = (noise + 1) % self.latent_dim
            p.append(params[index] * x_input[noise][i] + params[index + 1])
            index += 2
            noise = (noise + 1) % self.latent_dim
        for q in range(self.nqubits):
            p.append(params[index] * x_input[noise][i] + params[index + 1])
            index += 2
            noise = (noise + 1) % self.latent_dim
        circuit.set_parameters(p)

    def generate_latent_points(self, samples):
        """Generate points in latent space as input for the quantum generator."""
        # generate points in the latent space
        x_input = randn(self.latent_dim * samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(samples, self.latent_dim)
        return x_input

    def generate_fake_samples(self, params, samples, circuit, hamiltonians_list):
        import tensorflow as tf  # pylint: disable=import-error

        """Use the generator to generate fake examples, with class labels."""
        # generate points in latent space
        x_input = self.generate_latent_points(samples)
        x_input = np.transpose(x_input)
        # generator outputs
        X = []
        for i in range(self.nqubits):
            X.append([])
        # quantum generator circuit
        for i in range(samples):
            self.set_parameters(circuit, params, x_input, i)
            final_state = self.backend.execute_circuit(circuit, return_array=True)
            for ii in range(self.nqubits):
                X[ii].append(hamiltonians_list[ii].expectation(final_state))
        # shape array
        X = tf.stack([X[i] for i in range(len(X))], axis=1)
        # create class labels
        y = np.zeros((samples, 1))
        return X, y

    def define_cost_gan(
        self, params, discriminator, samples, circuit, hamiltonians_list
    ):
        import tensorflow as tf  # pylint: disable=import-error

        """Define the combined generator and discriminator model, for updating the generator."""
        # generate fake samples
        x_fake, y_fake = self.generate_fake_samples(
            params, samples, circuit, hamiltonians_list
        )
        # create inverted labels for the fake samples
        y_fake = np.ones((samples, 1))
        # evaluate discriminator on fake examples
        disc_output = discriminator(x_fake)
        loss = tf.keras.losses.binary_crossentropy(y_fake, disc_output)
        loss = tf.reduce_mean(loss)
        return loss

    def train(self, d_model, circuit, hamiltonians_list, save=True):
        """Train the quantum generator and classical discriminator."""
        import tensorflow as tf  # pylint: disable=import-error

        def generate_real_samples(samples, distribution, real_samples):
            """Generate real samples with class labels."""
            # generate samples from the distribution
            idx = np.random.randint(real_samples, size=samples)
            X = distribution[idx, :]
            # generate class labels
            y = np.ones((samples, 1))
            return X, y

        d_loss = []
        g_loss = []
        # determine half the size of one batch, for updating the discriminator
        half_samples = int(self.batch_samples / 2)
        if self.initial_params is not None:
            initial_params = tf.Variable(self.initial_params, dtype=tf.complex128)
        else:
            n = 10 * self.layers * self.nqubits + 2 * self.nqubits
            initial_params = tf.Variable(
                np.random.uniform(-0.15, 0.15, n), dtype=tf.complex128
            )

        optimizer = tf.optimizers.Adadelta(learning_rate=self.lr)
        # prepare real samples
        s = self.reference
        # manually enumerate epochs
        for i in range(self.n_epochs):
            # prepare real samples
            x_real, y_real = generate_real_samples(
                half_samples, s, self.training_samples
            )
            # prepare fake examples
            x_fake, y_fake = self.generate_fake_samples(
                initial_params, half_samples, circuit, hamiltonians_list
            )
            # update discriminator
            d_loss_real, _ = d_model.train_on_batch(x_real, y_real)
            d_loss_fake, _ = d_model.train_on_batch(x_fake, y_fake)
            d_loss.append((d_loss_real + d_loss_fake) / 2)
            # update generator
            with tf.GradientTape() as tape:
                loss = self.define_cost_gan(
                    initial_params,
                    d_model,
                    self.batch_samples,
                    circuit,
                    hamiltonians_list,
                )
            grads = tape.gradient(loss, initial_params)
            optimizer.apply_gradients([(grads, initial_params)])
            g_loss.append(loss)
            if save:  # pragma: no cover
                # saving is skipped in tests to avoid creating files
                params = (
                    self.nqubits,
                    self.latent_dim,
                    self.layers,
                    self.training_samples,
                    self.batch_samples,
                    self.lr,
                )
                filename = "_".join(str(p) for p in params)
                np.savetxt(f"PARAMS_{filename}", [initial_params.numpy()], newline="")
                np.savetxt(f"dloss_{filename}", [d_loss], newline="")
                np.savetxt(f"gloss_{filename}", [g_loss], newline="")
                # serialize weights to HDF5
                d_model.save_weights(f"discriminator_{filename}.h5")

    def fit(
        self,
        reference,
        initial_params=None,
        batch_samples=128,
        n_epochs=20000,
        lr=0.5,
        save=True,
    ):
        """Execute qGAN training.

        Args:
            reference (array): samples from the reference input distribution.
            initial_parameters (array): initial parameters for the quantum generator. If not provided,
                the default initial parameters will be used.
            discriminator (:class:`tensorflow.keras.models`): custom classical discriminator. If not provided,
                the default classical discriminator will be used.
            batch_samples (int): number of training examples utilized in one iteration.
            n_epochs (int): number of training iterations.
            lr (float): initial learning rate for the quantum generator.
                It controls how much to change the model each time the weights are updated.
            save (bool): If ``True`` the results of training (trained parameters and losses)
                will be saved on disk. Default is ``True``.
        """
        if initial_params is None and self.circuit is not None:
            raise_error(
                ValueError,
                "Set the initial parameters for your custom quantum generator.",
            )
        elif initial_params is not None and self.circuit is None:
            raise_error(
                ValueError,
                "Define the custom quantum generator to use custom initial parameters.",
            )

        self.reference = reference
        self.nqubits = reference.shape[1]
        self.training_samples = reference.shape[0]
        self.initial_params = initial_params
        self.batch_samples = batch_samples
        self.n_epochs = n_epochs
        self.lr = lr

        # create classical discriminator
        if self.discriminator is None:
            discriminator = self.define_discriminator()
        else:
            discriminator = self.discriminator

        if discriminator.input_shape[1] is not self.nqubits:
            raise_error(
                ValueError,
                "The number of input neurons in the discriminator has to be equal to "
                "the number of qubits in the circuit (dimension of the input reference distribution).",
            )

        # create quantum generator
        if self.circuit is None:
            circuit = models.Circuit(self.nqubits)
            for l in range(self.layers):
                for q in range(self.nqubits):
                    circuit.add(gates.RY(q, 0))
                    circuit.add(gates.RZ(q, 0))
                    circuit.add(gates.RY(q, 0))
                    circuit.add(gates.RZ(q, 0))
                for i in range(0, self.nqubits - 1):
                    circuit.add(gates.CRY(i, i + 1, 0))
                circuit.add(gates.CRY(self.nqubits - 1, 0, 0))
            for q in range(self.nqubits):
                circuit.add(gates.RY(q, 0))
        else:
            circuit = self.circuit

        if circuit.nqubits != self.nqubits:
            raise_error(
                ValueError,
                "The number of qubits in the circuit has to be equal to "
                "the number of dimensions in the reference distribution.",
            )

        # define hamiltonian to generate fake samples
        def hamiltonian(nqubits, position):
            kron = []
            for i in range(nqubits):
                if i == position:
                    kron.append(matrices.Z)
                else:
                    kron.append(matrices.I)
            for i in range(nqubits - 1):
                if i == 0:
                    ham = np.kron(kron[i + 1], kron[i])
                else:
                    ham = np.kron(kron[i + 1], ham)
            return hamiltonians.Hamiltonian(nqubits, ham, backend=self.backend)

        hamiltonians_list = []
        for i in range(self.nqubits):
            hamiltonians_list.append(hamiltonian(self.nqubits, i))

        # train model
        self.train(discriminator, circuit, hamiltonians_list, save)
