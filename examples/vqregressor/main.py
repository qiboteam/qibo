import argparse

import numpy as np
from vqregressor import VQRegressor

parser = argparse.ArgumentParser()
parser.add_argument(
    "--layers", default=3, help="Number of layers you want to involve", type=int
)
parser.add_argument(
    "--learning_rate",
    default=0.045,
    help="Learning rate for the Adam Descent",
    type=float,
)
parser.add_argument("--epochs", default=150, help="Number of training epochs", type=int)
parser.add_argument(
    "--batches",
    default=1,
    help="Number of batches which divide the training sample",
    type=int,
)
parser.add_argument(
    "--ndata", default=100, help="Number of data in the training set", type=int
)
parser.add_argument(
    "--J_treshold", default=1e-4, help="Number of data in the training set", type=float
)


def main(layers, learning_rate, epochs, batches, ndata, J_treshold):
    # We initialize the quantum regressor
    vqr = VQRegressor(layers=layers, ndata=ndata)
    # and the initial parameters
    initial_params = np.random.randn(3 * layers)
    # Let's go with the training
    vqr.train_with_psr(
        epochs=epochs,
        learning_rate=learning_rate,
        batches=batches,
        J_treshold=J_treshold,
    )
    vqr.show_predictions("Predictions of the VQR after training", False)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
