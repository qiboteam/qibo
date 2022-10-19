import numpy as np
from qibo import gates


def get_gammas(c, solve=True):
    """Standalone function to compute the ZNE coefficients given the noise levels.
        
    Args:
        c (numpy.ndarray): Array containing the different noise levels, note that in the CNOT insertion paradigm this corresponds to the number of CNOT pairs to be inserted. The canonical ZNE noise levels are obtained as 2*c + 1.
        solve (bool): If ``True`` computes the coeffients by solving the linear system. Otherwise, use the analytical solution valid for the CNOT insertion method.

    Returns:
        gammas (numpy.ndarray): The computed coefficients.
    """
    if solve:
        c = 2*c + 1
        a = np.array([ c**i for i in range(len(c)) ])
        b = np.zeros(len(c))
        b[0] = 1
        gammas = np.linalg.solve(a, b)
    else:
        cmax = c[-1]
        gammas = np.array([
            1/(2**(2*cmax)*np.math.factorial(i)) * (-1)**i/(1+2*i) * np.math.factorial(1+2*cmax)/(np.math.factorial(cmax)*np.math.factorial(cmax-i))
            for i in c
        ])
    return gammas

def get_noisy_circuit(circuit, cj):
    """Standalone function to generate the noisy circuit with the CNOT pairs insertion.
        
    Args:
        circuit (qibo.models.circuit.Circuit): Input circuit to modify.
        cj (int): Number of CNOT pairs to add.
    
    Returns:
        noisy_circuit (qibo.models.circuit.Circuit): The circuit with the inserted CNOT pairs.
    """
    noisy_circuit = circuit.__class__(**circuit.init_kwargs)
    for gate in circuit.queue:
        noisy_circuit.add(gate)
        if gate.__class__ == gates.CNOT:
            control = gate.control_qubits
            target = gate.target_qubits
            for i in range(cj):
                noisy_circuit.add(gates.CNOT(control, target))
                noisy_circuit.add(gates.CNOT(control, target))
    return noisy_circuit

def ZNE(circuit, observable, c, init_state=None, CNOT_noise_model=None):
    """Runs the Zero Noise Extrapolation method for error mitigation.
    The different noise levels are realized by the insertion of pairs of CNOT gates that resolve to the identiy in the noise-free case.

    Args:
        circuit (qibo.models.circuit.Circuit): Input circuit.
        observable (numpy.ndarray): Observable to measure.
        c (numpy.ndarray): Sequence of noise levels.
        init_state (numpy.ndarray): Initial state. 
        CNOT_noise_model (qibo.noise.NoiseModel): Noise model applied to each CNOT gate, used for simulating noisy CNOTs.

    Returns:
        Estimate of the expected value of ``observable`` in the noise free condition.
    """
    assert circuit.density_matrix, "Circuit.density_matrix == True is needed."
    expected_val = []
    for cj in c:
        noisy_circuit = get_noisy_circuit(circuit, cj)
        if CNOT_noise_model != None:
            noisy_circuit = CNOT_noise_model.apply(noisy_circuit)
        rho = noisy_circuit(initial_state = init_state)
        expected_val.append(observable.dot(rho.state()).trace())
    gamma = get_gammas(c, solve=False)
    return (gamma*expected_val).sum()


