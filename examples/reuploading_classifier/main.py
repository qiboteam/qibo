# /usr/bin/env python
import argparse
import pickle

from qlassifier import single_qubit_classifier

# TODO: fix issue with .pkl

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", default="tricrown", help="Name of the example", type=str
)
parser.add_argument("--layers", default=10, help="Number of layers.", type=int)


def main(dataset, layers):
    """Perform classification for a given problem and number of layers.

    Args:
        dataset (str): Problem to create the dataset, to choose between
            ['circle', '3_circles', 'square', '4_squares', 'crown', 'tricrown', 'wavy_lines']
        layers (int): Number of layers to use in the classifier
    """
    ql = single_qubit_classifier(dataset, layers)  # Define classifier
    try:
        with open("saved_parameters.pkl", "rb") as f:
            # Load previous results. Have we ever run these problem?
            data = pickle.load(f)
    except:
        data = {dataset: {}}

    try:
        parameters = data[dataset][layers]
        print("Problem solved before, obtaining parameters from file...")
        print("-" * 60)
    except:
        print("Problem never solved, finding optimal parameters...")
        result, parameters = ql.minimize(method="l-bfgs-b", options={"disp": True})

        data[dataset][layers] = parameters
        with open("saved_parameters.pkl", "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    ql.set_parameters(parameters)
    value_loss = ql.cost_function_fidelity()
    print("The value of the cost function achieved is %.6f" % value_loss)
    ql.paint_results()
    ql.paint_world_map()


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
