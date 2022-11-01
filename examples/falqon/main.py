import argparse

from qibo import hamiltonians, models


def main(nqubits, delta_t=0.1, max_layers=100):
    # create XXZ Hamiltonian for nqubits qubits
    hamiltonian = hamiltonians.XXZ(nqubits)
    # create FALQON model for this Hamiltonian
    falqon = models.FALQON(hamiltonian)

    best_energy, final_parameters = falqon.minimize(delta_t, max_layers)[:2]

    print("The optimal energy found is", best_energy)

    return best_energy, final_parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", default=5, type=int, help="Number of qubits.")
    parser.add_argument(
        "--delta_t",
        default=0.1,
        type=float,
        help="Optimization parameter, time step for the first layer",
    )
    parser.add_argument(
        "--max_layers", default=100, type=int, help="Maximum number of layers"
    )
    args = vars(parser.parse_args())
    main(**args)
