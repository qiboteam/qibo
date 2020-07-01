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
        from qibo import set_backend
        set_backend("matmuleinsum")
    for i in range(N):
        print('Initialized state with seed %s'%i)
        state = create_random_state(i) # Create a random state from the seed
        tangles[i] = compute_random_tangle(i) # Compute its tangle
        fun, params = canonize(state, p=p, shots=shots) # Perform transformation
        opt_tangles[i] = canonical_tangle(state, params, post_selection=post_selection) # Measuring the tangle with or without post selection

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