import argparse
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, schedules
import qibo
from qibo import gates
from qibo.models import Circuit

def main(n_layers, batch_size, nepochs, train_size, filename, lr_boundaries):
    qibo.set_backend("tensorflow")

    def make_encoder(n_qubits, n_layers, params, q_compression):
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


    @tf.function
    def compute_loss(encoder, params, vector):
        encoder.set_parameters(params)
        reconstructed = encoder(vector)
        # 3 qubits compression
        loss = (
            reconstructed.probabilities(qubits=[0])[0]
            + reconstructed.probabilities(qubits=[1])[0]
            + reconstructed.probabilities(qubits=[2])[0]
        )
        return loss


    @tf.function
    def train_step(batch_size, encoder, params, dataset):
        loss = 0.0
        with tf.GradientTape() as tape:
            for sample in range(batch_size):
                loss = loss + compute_loss(encoder, params, dataset[sample])
            loss = loss / batch_size
            grads = tape.gradient(loss, params)
            optimizer.apply_gradients(zip([grads], [params]))
        return loss


    dataset_np = np.load("data/standard_data.npy")
    dataset = tf.convert_to_tensor(dataset_np)
    n_qubits = 6
    q_compression = 3
    n_params = (n_layers * n_qubits + q_compression) * 3
    params = tf.Variable(tf.random.normal((n_params,)))
    encoder = make_encoder(n_qubits, n_layers, params, q_compression)
    print("Circuit model summary")
    print(encoder.draw())

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

    train = dataset[0:train_size]
    trained_params = np.zeros((nepochs, n_params), dtype=float)

    print("Trained parameters will be saved in: ", filename)
    print("Start training")
    for epoch in range(nepochs):
        tf.random.shuffle(train)
        for i in range(steps_for_epoch):
            loss = train_step(
                batch_size, encoder, params, train[i * batch_size : (i + 1) * batch_size]
            )
        trained_params[epoch] = params.numpy()
        print("Epoch: %d  Loss: %f" % (epoch + 1, loss))

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
        default="parameters/trained_params",
        type=str,
        help="(str): location and file name where trained parameters are saved",
    )
    parser.add_argument(
        "--lr_boundaries",
        default=[3,6,9,12,15,18],
        type=list,
        help="(list): epochs when learning rate is reduced (6 monotone growing values from 0 to nepochs)",
    )
    args = parser.parse_args()
    main(**vars(args))
