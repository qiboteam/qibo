import argparse
import os

import h5py
import numpy as np

import qibo

qibo.set_backend("qibojit")

import qkmedians as qkmed


def train_qkmedians(
    train_size,
    read_file,
    seed=None,
    k=2,
    tolerance=1.0e-3,
    min_type="classic",
    nshots=10000,
    save_dir=None,
    verbose=False,
    nprint=10,
):
    """Perform training of quantum k-medians.

    Parameters
    ----------
    train_size : int
        Number of training samples.
    read_file : str
        Name of the file where training data is saved.
    seed : int
        Seed for data shuffling.
    k : int
        Number of classes in quantum k-medians.
    tolerance : float
        Tolerance for algorithm convergence.
    min_type : str
        Type of minimization for distance calculation, classic or quantum.
    nshots : int
        Number of shots for executing quantum circuit.
    save_dir : str
        Name of the file for saving results.
    nprint : int
        Print loss function value each `nprint` epochs if `verbose` is `True`.
    verbose : str
        Print log messages during the training if `True`.
    """

    # read train data
    with h5py.File(read_file, "r") as file:
        data = file["latent_space"]
        l1 = data[:, 0, :]
        l2 = data[:, 1, :]

        data_train = np.vstack([l1[:train_size], l2[:train_size]])
        if seed:
            np.random.seed(seed)  # matters for small data sizes
        np.random.shuffle(data_train)

    # Intialize centroids
    centroids = qkmed.initialize_centroids(data_train, k)

    if verbose:
        print("Training of quantum k-medians")
        print(
            f"The algorithm will automatically break after the target {tolerance} tolerance is reached"
        )

    i = 0
    new_tol = 1
    loss = []
    while True:
        # find nearest centroids
        cluster_label, _ = qkmed.find_nearest_neighbour(
            data_train, centroids, min_type, nshots
        )
        # find new centroids
        new_centroids = qkmed.find_centroids(data_train, cluster_label, clusters=k)

        # calculate loss -> distance old_centroids to new_centroids
        loss_epoch = np.linalg.norm(centroids - new_centroids)
        loss.append(loss_epoch)

        # if verbose
        if (i % nprint == 0) and (verbose is True):
            print(f"Loss at epoch {i+1}: {loss[-1]:.8}")

        if loss_epoch < tolerance:
            centroids = new_centroids
            print(f"Converged after {i+1} iterations.")
            break
        elif (
            loss_epoch > tolerance and i > new_tol * 200
        ):  # if after 200*new_tol epochs, difference != 0, lower the tolerance
            tolerance *= 10
            new_tol += 1
        i += 1
        centroids = new_centroids

    if save_dir:
        # if save_dir doesn't exist it is created
        if os.path.exists(save_dir) is False:
            os.system(f"mkdir {save_dir}")

        np.save(
            f"{save_dir}/cluster_label.npy",
            cluster_label,
        )
        np.save(f"{save_dir}/centroids.npy", centroids)
        np.save(f"{save_dir}/loss.npy", loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read arguments for qkmedians training"
    )
    parser.add_argument(
        "--train_size", dest="train_size", type=int, help="Training data size"
    )
    parser.add_argument(
        "--read_file", dest="read_file", type=str, help="Path to training data"
    )
    parser.add_argument(
        "--seed", dest="seed", type=int, help="Seed for consistent results"
    )
    parser.add_argument("--k", dest="k", type=int, default=2, help="Number of classes")
    parser.add_argument(
        "--tolerance",
        dest="tolerance",
        type=float,
        default=1.0e-3,
        help="Convergence tolerance",
    )
    parser.add_argument(
        "--save_dir", dest="save_dir", type=str, help="Directory to save results"
    )
    parser.add_argument(
        "--min_type",
        dest="min_type",
        type=str,
        default="classic",
        help="Type of minimization for distance calculation, classic or quantum",
    )
    parser.add_argument(
        "--nshots",
        dest="nshots",
        type=int,
        default=10000,
        help="Number of shots for executing quantum circuit",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        type=bool,
        default=False,
        help="Verbose level during training",
    )
    parser.add_argument(
        "--nprint",
        dest="nprint",
        type=int,
        default=10,
        help="Log messages are printed every nprint epochs",
    )

    args = parser.parse_args()

    if args.min_type not in ["classic", "quantum"]:
        raise ValueError("Minimization is either classic or quantum procedure.")

    train_qkmedians(
        train_size=args.train_size,
        read_file=args.read_file,
        seed=args.seed,
        k=args.k,
        tolerance=args.tolerance,
        min_type=args.min_type,
        nshots=args.nshots,
        save_dir=args.save_dir,
        verbose=args.verbose,
        nprint=args.nprint,
    )
