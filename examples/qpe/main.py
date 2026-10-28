import argparse
import math

from qibo import Circuit, gates, hamiltonians
from qibo.models import QPE
from qibo.quantum_info import bitstring_to_phase


def main(nqubits_counting, nshots, tau):
    # Hamiltonian of two qubits, whose energies are the target of the estimation
    hamiltonian = hamiltonians.TFIM(2, h=0.5)
    energies = hamiltonian.eigenvalues()
    eigenvectors = hamiltonian.eigenvectors()
    print(f"Exact energies: {energies}\n")

    # The eigenvalue of U = exp(-i tau H) for the energy E is exp(2 pi i phase), with
    # phase = -tau E / (2 pi) modulo 1
    unitary = Circuit(2)
    unitary.add(gates.Unitary(hamiltonian.exp(tau), 0, 1))
    qpe = QPE(unitary, nqubits_counting)
    target_qubits = (nqubits_counting, nqubits_counting + 1)

    # Phase estimation starting from the ground state. The eigenvectors are the columns of
    # a unitary, so it maps the state |00> into the ground state
    circuit = Circuit(qpe.nqubits)
    circuit.add(gates.Unitary(eigenvectors, *target_qubits))
    circuit.add(qpe)
    frequencies = circuit(nshots=nshots).frequencies()
    phase = bitstring_to_phase(frequencies.most_common(1)[0][0])
    energy = -2 * math.pi * phase / tau
    resolution = 2 * math.pi / (tau * 2**nqubits_counting)
    print("Ground state")
    print(f"Estimated phase: {phase}")
    print(f"Estimated energy: {energy:.4f}, exact energy: {energies[0]:.4f}")
    print(f"Energy resolution: {resolution:.4f}\n")

    # Phase estimation starting from a superposition of the ground state and the state
    # with highest energy, whose phases are 0.5 or larger, as the energy is positive
    circuit = Circuit(qpe.nqubits)
    circuit.add(gates.H(target_qubits[0]))
    circuit.add(gates.CNOT(*target_qubits))
    circuit.add(gates.Unitary(eigenvectors, *target_qubits))
    circuit.add(qpe)
    frequencies = circuit(nshots=nshots).frequencies()
    print("Superposition of the ground state and the state with highest energy")
    for bitstring, counts in frequencies.most_common(2):
        phase = bitstring_to_phase(bitstring)
        # phases above 1 / 2 correspond to negative values of tau E
        energy = -2 * math.pi * (phase if phase < 0.5 else phase - 1) / tau
        print(
            f"Phase {phase} observed in {counts} of {nshots} shots, energy {energy:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nqubits-counting", default=8, type=int, help="Number of counting qubits."
    )
    parser.add_argument("--nshots", default=1000, type=int, help="Number of shots.")
    parser.add_argument(
        "--tau", default=1.0, type=float, help="Evolution time of the unitary."
    )
    args = vars(parser.parse_args())
    main(**args)
