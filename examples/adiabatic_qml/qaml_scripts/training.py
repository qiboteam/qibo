import matplotlib.pyplot as plt
import numpy as np
from qaml_scripts.evolution import perform_adiabatic

import qibo


def train_adiabatic_evolution(
    nsteps,
    xarr,
    cdf,
    training_n,
    init_params,
    e0,
    e1,
    target_loss,
    finalT,
    h0,
    h1,
    obs_target,
):
    """Train the adiabatic evolution to fit a target empirical CDF"""

    # --------------------------- PLOTTING FUNCTION -----------------------------------------------

    def plot_for_energies(parameters, label="", true_law=None, title=""):
        """Plot energies, training points and CDF for a set of energies given a set of parameters"""
        energies = perform_adiabatic(
            params=parameters,
            finalT=finalT,
            h0=h0,
            h1=h1,
            obs_target=obs_target,
        )

        plt.title(title)
        plt.plot(xarr, -np.array(cdf), label="eCDF", color="black", lw=1, ls="--")
        plt.plot(
            xarr, -np.array(energies), label=label, color="purple", lw=2, alpha=0.8
        )
        plt.plot(
            xarr[idx_training],
            -np.array(cdf_training),
            "o",
            label="Training points",
            color="orange",
            alpha=0.85,
            markersize=8,
        )
        if true_law != None:
            plt.plot(xarr, true_law, c="orange", lw=1, ls="--")
        plt.xlabel("x")
        plt.ylabel("cdf")
        plt.legend()
        plt.show()

    # ----------------------------- LOSS FUNCTION ---------------------------------------------

    def loss_evaluation(params, penalty=True):
        """Evaluating loss function related to the cdf fit"""

        # Retrieve the energy per time step for this set of parameters
        energies = perform_adiabatic(
            params=params,
            finalT=finalT,
            h0=h0,
            h1=h1,
            obs_target=obs_target,
        )

        # Select the points we are training on
        e_train = energies[idx_training]

        loss = np.mean((e_train - cdf_training) ** 2 / norm_cdf)

        if penalty:
            # Penalty term for negative derivative
            delta_energy = good_direction * np.diff(energies)
            # Remove non-monotonous values
            delta_energy *= delta_energy < 0

            pos_penalty = np.abs(np.sum(delta_energy))
            val_loss = loss

            loss = val_loss + pos_penalty

        return loss

    # ------------------------------ GENETIC ALGORITHM CALL --------------------------------------------------

    def optimize(
        force_positive=False,
        target=5e-2,
        max_iterations=50000,
        max_evals=500000,
        initial_p=None,
    ):
        """Use Qibo to optimize the parameters of the schedule function"""

        options = {
            "verbose": -1,
            "tolfun": 1e-12,
            "ftarget": target,  # Target error
            "maxiter": max_iterations,  # Maximum number of iterations
            "maxfeval": max_evals,  # Maximum number of function evaluations
            "maxstd": 20,
        }

        if force_positive:
            options["bounds"] = [0, 1e5]

        if initial_p is None:
            initial_p = initial_p
        else:
            print("Reusing previous best parameters")

        result = qibo.optimizers.optimize(
            loss_evaluation, initial_p, method="cma", options=options
        )

        return result, result[1]

    # ------------------------------ BUILD TRAINING SET AND OPTIMIZE! --------------------------------

    # Definition of the loss function and optimization routine
    good_direction = 1 if (e1 - e0) > 0 else -1

    # But select those for which the difference between them is greater than some threshold
    min_step = 1e-3
    # but never go more than max_skip points without selecting one
    max_skip = 0

    if training_n > nsteps:
        raise Exception("The notebook cannot run with nsteps < training_n")

    # Select a subset of points for training, but skip first and include last
    idx_training_raw = np.linspace(0, nsteps, num=training_n, endpoint=True, dtype=int)[
        1:
    ]

    # And, from this subset, remove those that do not add that much info
    idx_training = []
    cval = cdf[0]
    nskip = 0
    for p in idx_training_raw[:-2]:
        diff = cval - cdf[p]
        if diff > min_step or nskip > max_skip:
            nskip = 0
            idx_training.append(p)
            cval = cdf[p]
        else:
            nskip += 1

    idx_training.append(idx_training_raw[-1])

    cdf_training = cdf[idx_training]
    norm_cdf = np.abs(
        cdf_training
    )  # To normalize the points according to their absolute value

    # Definition of the loss function and optimization routine
    good_direction = 1 if (e1 - e0) > 0 else -1

    # Fit before training
    plot_for_energies(init_params, label="Initial state", title="Not trained evolution")
    print(f"Training on {len(idx_training)} points of the total of {nsteps}")

    _, best_params = optimize(
        target=target_loss, force_positive=False, initial_p=init_params
    )

    # Fit after training
    plot_for_energies(best_params, label="Initial state", title="Trained evolution")

    return best_params
