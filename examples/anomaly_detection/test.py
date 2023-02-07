import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import qibo
from qibo import gates
from qibo.models import Circuit


def main(n_layers, train_size, filename, plot, save_loss):
    """Implements performance evaluation of a trained circuit.

    Args:
        n_layers (int): number of ansatz circuit layers (default 6).
        train_size (int): number of samples used for training, the remainings are used for performance evaluation, total samples are 7000 (default 5000).
        filename (str): location and file name of trained parameters to be tested (default "parameters/trained_params.npy").
        plot (bool): make plots of ROC and loss function distribution (default True).
        save_loss (bool): save losses for standard and anomalous data (default False).
    """

    qibo.set_backend("tensorflow")

    # Circuit ansatz
    def make_encoder(n_qubits, n_layers, params, q_compression):
        """Create encoder quantum circuit.

        Args:
            n_qubits (int): number of qubits in the circuit.
            n_layers (int): number of ansatz circuit layers.
            params (tf.Tensor): parameters of the circuit.
            q_compression (int): number of compressed qubits.

        Returns:
            encoder (:class:`qibo.models.circuit.Circuit`): variational quantum circuit.
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
    def compute_loss_test(encoder, vector):
        """Evaluate loss function for one test sample.

        Args:
            encoder (:class:`qibo.models.circuit.Circuit`): variational quantum circuit (trained).
            vector (tf.Tensor): test sample, in the form of 1d vector.

        Returns:
            loss (tf.Variable): loss of the test sample.
        """
        reconstructed = encoder(vector)
        # 3 qubits compression
        loss = (
            reconstructed.probabilities(qubits=[0])[0]
            + reconstructed.probabilities(qubits=[1])[0]
            + reconstructed.probabilities(qubits=[2])[0]
        )
        return loss

    # Other hyperparameters
    n_qubits = 6
    q_compression = 3

    # Load and pre-process data
    dataset_np_s = np.load("data/standard_data.npy")
    dataset_np_s = dataset_np_s[train_size:]
    dataset_s = tf.convert_to_tensor(dataset_np_s)
    dataset_np_a = np.load("data/anomalous_data.npy")
    dataset_np_a = dataset_np_a[train_size:]
    dataset_a = tf.convert_to_tensor(dataset_np_a)

    # Load trained parameters
    trained_params_np = np.load(filename)
    trained_params = tf.convert_to_tensor(trained_params_np)

    # Create and print encoder circuit
    encoder_test = make_encoder(n_qubits, n_layers, trained_params, q_compression)
    encoder_test.compile()
    print("Circuit model summary")
    print(encoder_test.draw())

    print("Computing losses...")
    # Compute loss for standard data
    loss_s = []
    for i in range(len(dataset_np_s)):
        loss_s.append(compute_loss_test(encoder_test, dataset_s[i]).numpy())

    # Compute loss for anomalous data
    loss_a = []
    for i in range(len(dataset_np_a)):
        loss_a.append(compute_loss_test(encoder_test, dataset_a[i]).numpy())

    if save_loss:
        np.save("results/losses_standard_data", loss_s)
        np.save("results/losses_anomalous_data", loss_a)

    # Make graphs for performance analysis
    if plot:
        """Loss distribution graph"""
        plt.hist(loss_a, bins=60, histtype="step", color="red", label="Anomalous data")
        plt.hist(loss_s, bins=60, histtype="step", color="blue", label="Standard data")
        plt.ylabel("Number of images")
        plt.xlabel("Loss value")
        plt.title("Loss function distribution (MNIST dataset)")
        plt.legend()
        plt.savefig("results/loss_distribution.png")
        plt.close()

        """Compute ROC curve"""
        max1 = np.amax(loss_s)
        max2 = np.amax(loss_a)
        ma = max(max1, max2)
        min1 = np.amin(loss_s)
        min2 = np.amin(loss_a)
        mi = min(min1, min2)

        tot_neg = len(loss_s)
        tot_pos = len(loss_a)

        n_step = 100.0
        n_step_int = 100
        step = (ma - mi) / n_step
        fpr = []
        tpr = []
        for i in range(n_step_int):
            treshold = i * step + mi
            c = 0
            for j in range(tot_neg):
                if loss_s[j] > treshold:
                    c += 1
            false_positive = c / float(tot_neg)
            fpr.append(false_positive)
            c = 0
            for j in range(tot_pos):
                if loss_a[j] > treshold:
                    c += 1
            true_positive = c / float(tot_pos)
            tpr.append(true_positive)

        """Roc curve graph """
        plt.title("Receiver Operating Characteristic")
        plt.plot(fpr, tpr)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.savefig("results/ROC.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_layers",
        default=6,
        type=int,
        help="(int): number of ansatz circuit layers",
    )
    parser.add_argument(
        "--train_size",
        default=5000,
        type=int,
        help="(int): number of samples used for training, the remainings are used for performance evaluation (total samples 7000)",
    )
    parser.add_argument(
        "--filename",
        default="parameters/trained_params.npy",
        type=str,
        help="(str): location and file name of trained parameters to be tested",
    )
    parser.add_argument(
        "--plot",
        default=True,
        type=bool,
        help="(bool): make plots of ROC and loss function distribution",
    )
    parser.add_argument(
        "--save_loss",
        default=False,
        type=bool,
        help="(bool): save losses for standard and anomalous data",
    )
    args = parser.parse_args()
    main(**vars(args))
