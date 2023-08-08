import numpy as np

from qibo import callbacks, models


def generate_schedule(params, schedule="poly"):
    """Generate the scheduling used into the adiabatic evolution."""
    nparams = len(params)

    def poly(t):
        """General polynomial scheduling satisfying s(0)=0 and s(1)=1"""
        f = np.sum(t ** np.arange(1, 1 + nparams) * params)
        f /= np.sum(params)
        return f

    def derpoly(t):
        "Derivative of the polynomial above"
        f = np.sum(np.arange(1, 1 + nparams) * (t ** np.arange(0, nparams) * params))
        return f / np.sum(params)

    return poly, derpoly


def generate_adiabatic(params, h0, h1, obs_target, dt=1e-1, solver="exp"):
    """Generate the adiabatic evolution object."""
    energy = callbacks.Energy(obs_target)
    # same scheduling function
    schedule, _ = generate_schedule(params, schedule="poly")
    return (
        models.AdiabaticEvolution(
            h0,
            h1,
            schedule,
            dt=1e-1,
            solver="exp",
            callbacks=[energy],
        ),
        energy,
    )


# Perform adiabatic evolution (returns energy callback)
def perform_adiabatic(params, finalT, h0, h1, obs_target, dt=1e-1, solver="exp"):
    """Return energy callback as an array"""
    evolution, energy = generate_adiabatic(
        params, h0, h1, obs_target, dt=1e-1, solver="exp"
    )
    _ = evolution(final_time=finalT)
    return np.array(energy.results)
