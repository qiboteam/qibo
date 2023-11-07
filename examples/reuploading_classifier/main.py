# /usr/bin/env python
import argparse
import pickle

from qibo.examples.reuploading_classifier.nqubits_qlassifier import n_qubit_classifier
from qibo.examples.reuploading_classifier.qlassifier import single_qubit_classifier

# TODO: fix issue with .pkl

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", default="tricrown", help="Name of the example", type=str
)
parser.add_argument("--layers", default=10, help="Number of layers.", type=int)
parser.add_argument("--nqubits", default=1, help="Number of qubits.", type=int)
parser.add_argument("--dimension", default=2, help="Dimension of the data.", type=int)
parser.add_argument(
    "--entanglement", default=False, help="Enable entangling gates.", type=bool
)
parser.add_argument(
    "--target_naive", default=True, help="Target state definition.", type=bool
)


# TODO: Choose
# TODO: The dataset name will encode the data dimension
def main(
    dataset, layers, nqubits=1, data_dimension=2, entanglement=False, target_naive=True
):
    """Perform classification for a given problem and number of layers.

    Args:
        dataset (str): Problem to create the dataset, to choose between
            ['circle', '3_circles', 'square', '4_squares', 'crown', 'tricrown', 'wavy_lines']
        layers (int): Number of layers to use in the classifier
    """
    if nqubits == 1:
        ql = single_qubit_classifier(dataset, layers)
    else:
        ql = n_qubit_classifier(
            dataset, layers, nqubits, entanglement, data_dimension, target_naive
        )

    # try:
    #     with open("saved_parameters.pkl", "rb") as f:
    #         # Load previous results. Have we ever run these problem?
    #         data = pickle.load(f)
    # except:
    data = {dataset: {}}

    # TODO: Get this back
    # try:
    #     parameters = data[dataset][layers]
    #     print("Problem solved before, obtaining parameters from file...")
    #     print("-" * 60)
    # except:
    print("Problem never solved, finding optimal parameters...")
    result, parameters = ql.minimize(method="l-bfgs-b", options={"disp": True})

    # FIXME: This is crashing for new data
    data[dataset][layers] = parameters
    with open("saved_parameters.pkl", "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    ql.set_parameters(parameters)
    value_loss = ql.cost_function_fidelity()
    print("The value of the cost function achieved is %.6f" % value_loss)

    if data_dimension == 2:
        ql.paint_results()
    elif data_dimension == 3:
        ql.paint_results_3D()
        # pass
    if nqubits == 1:
        ql.paint_world_map()


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
