import argparse
import math
from pathlib import Path

import numpy as np

import qibo
from qibo import Circuit, gates

LOCAL_FOLDER = Path(__file__).parent


def main(n_layers, batch_size, nepochs, train_size, filename, lr_boundaries):
    """Implements training of variational quantum circuit, as described in https://doi.org/10.3390/particles6010016.

    Args:
        n_layers (int): number of ansatz circuit layers (default 6).
        batch_size (int): number of samples in one training batch (default 20).
        nepochs (int): number of training epochs (default 20).
        train_size (int): number of samples used for training, the remainings are used for performance evaluation, total samples are 7000 (default 5000).
        filename (str): location and file name where trained parameters are saved (default "parameters/trained_params.npy").
        lr_boundaries (list): epochs when learning rate is reduced, 6 monotone growing values from 0 to nepochs (default [3,6,9,12,15,18]).
    """

    qibo.set_backend(backend="qiboml", platform="tensorflow")
    tf = qibo.get_backend().tf

    # Circuit ansatz
    def make_encoder(n_qubits, n_layers, params, q_compression):
        """Create encoder quantum circuit.

        Args:
            n_qubits (int): number of qubits in the circuit.
            n_layers (int): number of ansatz circuit layers.
            params (tf.Variable): parameters of the circuit.
            q_compression (int): number of compressed qubits.

        Returns:
            :class:`qibo.models.Circuit`: Variational quantum circuit.
        """

        index = 0
        encoder = Circuit(n_qubits)
        for i in range(n_layers):
            for j in range(n_qubits):
                encoder.add(gates.RX(j, params[index]))
                encoder.add(gates.RY(j, params[index + 1]))
                encoder.add(gates.RZ(j, params[index + 2]))
                index += 3

            for j in range(n_qubits):
                encoder.add(gates.CNOT(j, (j + 1) % n_qubits))

        for j in range(q_compression):
            encoder.add(gates.RX(j, params[index]))
            encoder.add(gates.RY(j, params[index + 1]))
            encoder.add(gates.RZ(j, params[index + 2]))
            index += 3
        return encoder

    # Evaluate loss function (3 qubit compression) for one sample
    @tf.function
    def compute_loss(encoder, params, vector):
        """Evaluate loss function for one train sample.

        Args:
            encoder (:class:`qibo.models.Circuit`): variational quantum circuit.
            params (:class:`tf.Variable`): parameters of the circuit.
            vector (:class:`tf.Tensor`): train sample, in the form of 1d vector.

        Returns:
            :class:`tf.Variable`: Loss of the training sample.
        """

        encoder.set_parameters(params)
        reconstructed = encoder(vector)
        # 3 qubits compression
        loss = (
            reconstructed.probabilities(qubits=[0])[0]
            + reconstructed.probabilities(qubits=[1])[0]
            + reconstructed.probabilities(qubits=[2])[0]
        )
        return loss

    # One optimization step
    @tf.function
    def train_step(batch_size, encoder, params, dataset):
        """Evaluate loss function on one train batch.

        Args:
            batch_size (int): number of samples in one training batch.
            encoder (:class:`qibo.models.Circuit`): variational quantum circuit.
            params (:class:`tf.Variable`): parameters of the circuit.
            vector (:class:`tf.Tensor`): train sample, in the form of 1d vector.

        Returns:
            :class:`tf.Variable`: Average loss of the training batch.
        """

        loss = 0.0
        with tf.GradientTape() as tape:
            for sample in range(batch_size):
                loss = loss + compute_loss(encoder, params, dataset[sample])
            loss = loss / batch_size
            grads = tape.gradient(loss, params)
            optimizer.apply_gradients(zip([grads], [params]))
        return loss

    # Other hyperparameters
    n_qubits = 6
    q_compression = 3

    # Load and pre-process data
    file_dataset = LOCAL_FOLDER / "data" / "standard_data.npy"
    dataset_np = np.load(file_dataset)
    dataset = tf.convert_to_tensor(dataset_np)
    train = dataset[0:train_size]

    # Initialize random parameters
    n_params = (n_layers * n_qubits + q_compression) * 3
    params = tf.Variable(tf.random.normal((n_params,)))

    # Create and print encoder circuit
    encoder = make_encoder(n_qubits, n_layers, params, q_compression)
    print("Circuit model summary")
    encoder.draw()

    # Define optimizer parameters
    steps_for_epoch = math.ceil(train_size / batch_size)
    boundaries = [
        steps_for_epoch * lr_boundaries[0],
        steps_for_epoch * lr_boundaries[1],
        steps_for_epoch * lr_boundaries[2],
        steps_for_epoch * lr_boundaries[3],
        steps_for_epoch * lr_boundaries[4],
        steps_for_epoch * lr_boundaries[5],
    ]
    values = [0.4, 0.2, 0.08, 0.04, 0.01, 0.005, 0.001]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

    # This array contains the parameters at each epoch
    trained_params = np.zeros((nepochs, n_params), dtype=float)

    # Training
    print("Trained parameters will be saved in: ", filename)
    print("Start training")
    for epoch in range(nepochs):
        tf.random.shuffle(train)
        for i in range(steps_for_epoch):
            loss = train_step(
                batch_size,
                encoder,
                params,
                train[i * batch_size : (i + 1) * batch_size],
            )
        trained_params[epoch] = params.numpy()
        print("Epoch: %d  Loss: %f" % (epoch + 1, loss))

    # Save parameters of last epoch
    np.save(filename, trained_params[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_layers",
        default=6,
        type=int,
        help="(int): number of ansatz circuit layers",
    )
    parser.add_argument(
        "--batch_size",
        default=20,
        type=int,
        help="(int): number of samples in one training batch",
    )
    parser.add_argument(
        "--nepochs",
        default=20,
        type=int,
        help="(int): number of training epochs",
    )
    parser.add_argument(
        "--train_size",
        default=5000,
        type=int,
        help="(int): number of samples used for training, the remainings are used for performance evaluation (total samples 7000)",
    )
    parser.add_argument(
        "--filename",
        default=LOCAL_FOLDER / "parameters" / "trained_params.npy",
        type=str,
        help="(str): location and file name where trained parameters are saved",
    )
    parser.add_argument(
        "--lr_boundaries",
        default=[3, 6, 9, 12, 15, 18],
        type=list,
        help="(list): epochs when learning rate is reduced (6 monotone growing values from 0 to nepochs)",
    )
    args = parser.parse_args()
    main(**vars(args))
