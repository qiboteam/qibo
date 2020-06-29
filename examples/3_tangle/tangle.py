from canonizator import *
import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--N", default=100, help="Number of random states.", type=int)
parser.add_argument("--p", default=0, help="Probability of occurring an error.", type=float)
parser.add_argument("--shots", default=1000, help="Shots used for measuring every circuit.", type=float)

def main(N, p, shots):
    tangles = np.empty(N)
    opt_tangles = np.empty(N)

    for i in range(N):
        state = create_random_state(i)
        tangles[i] = compute_random_tangle(i)
        fun, params = canonize(state, p=p, shots=shots)
        opt_tangles[i] = canonical_tangle(state, params, post_selection=False)
        print(fun, opt_tangles[i])

    fig, ax = plt.subplots()
    ax.scatter(tangles, opt_tangles, s=20)
    ax.plot([0., 1.], [0., 1.], color='black')
    ax.set(xlabel='Exact tangle', ylabel='Measured tangle')
    plt.show()



if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)