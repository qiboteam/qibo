from canonizator import *
import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--N", default=100, help="Number of random states.", type=int)
parser.add_argument("--p", default=0.001, help="Probability of occurring an error.", type=float)
parser.add_argument("--shots", default=1000, help="Shots used for measuring every circuit.", type=float)
parser.add_argument("--post_selection", default=True, help="Post selection technique", type=bool)

def main(N, p, shots, post_selection):
    #Initialize exact and measured tangles
    tangles = np.empty(N)
    opt_tangles = np.empty(N)
    if p != 0:
        # backend is adapted to the error implementation
        from qibo import set_backend
        set_backend("matmuleinsum")

    circuit = ansatz(p)
    for i in range(N):
        """
        For every seed from 0 to N, the steps to follow are
        1) A random state with three qubits is created from the seed
        2) The tangle of the recently created random state is computed exactly
        3) Transformation to the up-to-phases canonical form is performed by applying local operations
                that drive some coefficients to zero
        4) The tangle of the up-to-phases canonical form of the created state is measured from the outcomes of the state.
                Post-selection can be applied at this stage

        Results are stored into variables to paint the results
        """
        if i % 10 == 0:
            print('Initialized state with seed %s'%i + '/ %s'%N)
        state = create_random_state(i)
        tangles[i] = compute_random_tangle(i)
        fun, params = canonize(state, circuit, shots=shots)
        opt_tangles[i] = canonical_tangle(state, params, circuit, post_selection=post_selection)

    print('Painting results')
    fig, ax = plt.subplots() # Plotting
    if post_selection:
        color = 'red'
    else:
        color='green'
    ax.scatter(tangles, opt_tangles, s=20, c=color)
    ax.plot([0., 1.], [0., 1.], color='black')
    ax.set(xlabel='Exact tangle', ylabel='Measured tangle', xlim=[0,1], ylim=[0,1])
    plt.grid('on')
    plt.show()


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
