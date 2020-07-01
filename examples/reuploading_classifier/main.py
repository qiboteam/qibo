#/usr/bin/env python
import datasets as ds
import numpy as np
from qlassifier import single_qubit_classifier
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", default='tricrown', help="Name of the example", type=str)
parser.add_argument("--layers", default=10, help="Number of layers.", type=int)

def main(name, layers):
    """Perform classification for a given problem and number of layers
    Args:
        name (str): Name of the problem to create the dataset, to choose between ['circle', '3_circles', 'square',
                                                            '4_squares', 'crown', 'tricrown', 'wavy_lines']
        layers (int): Number of layers to use in the classifier

    """

    ql = single_qubit_classifier(name, layers) # Define classifier
    with open('saved_parameters.pkl', 'rb') as f:
        data = pickle.load(f) # Load previous results. Have we ever run these problem?
    try:
        parameters = data[name][layers]
        print('Problem solved before, obtaining parameters from file...')
        print('-'*60)
    except:
        print('Problem never solved, finding optimal parameters...')
        result, parameters = ql.minimize(method='l-bfgs-b', options={'disp': True})
        data[name][layers] = parameters
        with open('saved_parameters.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    ql.set_parameters(parameters)
    value_loss = ql.cost_function_fidelity()
    print('The value of the cost function achieved is %.6f'%value_loss.numpy())
    ql.paint_results()
    ql.paint_world_map()

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
