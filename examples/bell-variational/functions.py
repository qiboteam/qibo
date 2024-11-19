import numpy as np

from qibo import Circuit, gates


def bell_circuit(basis):
    """Create a Bell circuit with a distinct measurement basis and parametrizable gates.

    Args:
        basis (str): '00', '01, '10', '11' where a '1' marks a measurement in the X basis
            and a '0' a measurement in the Z basis.

    Returns:
        :class:`qibo.core.circuit.Circuit`

    """
    circuit = Circuit(2)
    circuit.add(gates.RY(0, theta=0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RY(0, theta=0))
    for a in range(2):
        if basis[a] == "1":
            circuit.add(gates.H(a))
    circuit.add(gates.M(*range(2)))
    return circuit


def set_parametrized_circuits():
    """Create all Bell circuits needed to compute the CHSH inequelities.
    Returns:
        list(:class:`qibo.core.circuit.Circuit`)

    """
    chsh_circuits = []
    basis = ["00", "01", "10", "11"]
    for base in basis:
        chsh_circuits.append(bell_circuit(base))
    return chsh_circuits


def compute_chsh(frequencies, nshots):
    """Computes the CHSH inequelity value given the restults of an experiment.
    Args:
        frequencies (list): list of dictionaries with the result for the measurement
                            and the number of times it repeated, for each measurement
                            basis
        nshots (int): number of samples taken in each run of the circuit.

    Returns:
        chsh (float): value of the CHSH inequality.

    """
    chsh = 0
    aux = 0
    for freq in frequencies:
        for outcome in freq:
            if aux == 1:
                chsh -= (-1) ** (int(outcome[0]) + int(outcome[1])) * freq[outcome]
            else:
                chsh += (-1) ** (int(outcome[0]) + int(outcome[1])) * freq[outcome]
        aux += 1
    chsh /= nshots
    return chsh


def cost_function(parameters, circuits, nshots):
    """Compute the cost function that attempts to maximize the violation of CHSH inequalities.
    Args:
        parameters (np.array): parameters for the variational algorithm. First determines the
                               initial state and second the measurement direction.
        circuits (list(:class:`qibo.core.circuit.Circuit`)): Bell circuits in the 4 needed basis.
        nshots (int): number of sampling for the experiment.

    Returns:
        cost (float): closer to 0 as the inequalities are maximally violated. Can go negative due
                     to sampling. (can be squared to avoid that)

    """
    frequencies = []
    for circuit in circuits:
        circuit.set_parameters(parameters)
        frequencies.append(circuit(nshots=nshots).frequencies())
    chsh = compute_chsh(frequencies, nshots)
    cost = np.sqrt(2) * 2 - np.abs(chsh)
    return cost
