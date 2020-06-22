import numpy as np
import matplotlib.pyplot as plt
from qibo.models import Circuit
from qibo import gates
from aux_functions import *

"""
This file provides all required functions for performing calculations in unary basis
"""

def rw_circuit(qubits, parameters, X=True):
    """
    Circuit that emulates the probability sharing between neighbours seen usually
    in Brownian motion and stochastic systems
    :param qubits: number of qubits of the circuit
    :param parameters: parameters for performing the circuit
    :return: Quantum circuit
    """
    if qubits%2==0:
        mid1 = int(qubits/2)
        mid0 = int(mid1-1)
        if X:
            yield gates.X(mid1)
        #SWAP-Ry gates
        #-------------------------------------------------
        yield gates.fSim(mid1, mid0, parameters[mid0]/2, 0)
        #-------------------------------------------------
        for i in range(mid0):
            #SWAP-Ry gates
            #-------------------------------------------------
            yield gates.fSim(mid0-i, mid0-i-1, parameters[mid0-i-1]/2, 0)
            #-------------------------------------------------
            yield gates.fSim(mid1+i, mid1+i+1, parameters[mid1+i]/2, 0)
            #-------------------------------------------------
    else:
        mid = int((qubits-1)/2) #The random walk starts from the middle qubit
        if X:
            yield gates.X(mid)
        for i in range(mid):
            #SWAP-Ry gates
            #-------------------------------------------------
            yield gates.fSim(mid-i, mid-i-1, parameters[mid-i-1]/2, 0)
            #-------------------------------------------------
            yield gates.fSim(mid+i, mid+i+1, parameters[mid+i]/2, 0)
            #-------------------------------------------------

def rw_circuit_inv(qubits, parameters, X=True):
    """
    Circuit that emulates the probability sharing between neighbours seen usually
    in Brownian motion and stochastic systems, INVERTED!
    :param qubits: number of qubits of the circuit
    :param parameters: parameters for performing the circuit
    :return: Quantum circuit
    """
    if qubits%2==0:
        mid1 = int(qubits/2)
        mid0 = int(mid1-1)
        for i in range(mid0 - 1, -1, -1):
            # SWAP-Ry gates
            # -------------------------------------------------
            yield gates.fSim(mid0 - i, mid0 - i - 1, -parameters[mid0 - i - 1]/2, 0)
            # -------------------------------------------------
            yield gates.fSim(mid1 + i, mid1 + i + 1, -parameters[mid1 + i]/2, 0)
            # -------------------------------------------------

        # SWAP-Ry gates
        # -------------------------------------------------
        yield gates.fSim(mid1, mid0, -parameters[mid0]/2, 0)

        if X:
            yield gates.X(mid1)

    else:
        mid = int((qubits-1)/2) #The random walk starts from the middle qubit
        for i in range(mid, -1, -1):
            #SWAP-Ry gates
            #-------------------------------------------------
            yield gates.fSim(mid-i, mid-i-1, -parameters[mid-i-1], 0)
            #-------------------------------------------------
            yield gates.fSim(mid+i, mid+i+1, -parameters[mid+i]/2, 0)
            #-------------------------------------------------
        if X:
            yield gates.X(mid)


def create_qc(qubits):
    q = [i for i in range(qubits)]
    ancilla = qubits
    circuit = Circuit(qubits+1)
    return q, ancilla, circuit

def rw_parameters(qubits, pdf):
    """
    Solving for the exact angles for the random walk circuit that enables the loading of
    a desired probability distribution function
    :param qubits: number of qubits of the circuit
    :param pdf: probability distribution function to emulate
    :return: set of parameters
    """
    if qubits%2==0:
        mid = qubits // 2
    else:
        mid = (qubits-1)//2 #Important to keep track of the centre
    last = 1
    parameters = []
    for i in range(mid-1):
        angle = 2 * np.arctan(np.sqrt(pdf[i]/(pdf[i+1] * last)))
        parameters.append(angle)
        last = (np.cos(angle/2))**2 #The last solution is needed to solve the next one
    angle = 2 * np.arcsin(np.sqrt(pdf[mid-1]/last))
    parameters.append(angle)
    last = (np.cos(angle/2))**2
    for i in range(mid, qubits-1):
        angle = 2 * np.arccos(np.sqrt(pdf[i]/last))
        parameters.append(angle)
        last *= (np.sin(angle/2))**2
    return parameters

def measure_probability(q):
    """
    Circuit to sample the created probability distribution function
    :param qubits: number of qubits of the circuit
    :return: circuit + measurements
    """
    yield gates.M(*q, register_name='prob') #No measure on the ancilla qubit is necessary

def extract_probability(qubits, counts, samples):
    """
    From the retuned sampling, extract only probabilities of unary states
    :param qubits: number of qubits of the circuit
    :param counts: Number of measurements extracted from the circuit
    :param samples: Number of measurements applied the circuit
    :return: probabilities of unary states
    """
    form = '{0:0%sb}' % str(qubits) # qubits?
    prob = []
    for i in reversed(range(qubits)):
        prob.append(counts.get(form.format(2**i), 0)/samples)
    return prob

def get_pdf(qu, S0, sig, r, T):
    mu = (r - 0.5 * sig ** 2) * T + np.log(S0)  # Define all the parameters to be used in the computation
    mean = np.exp(
        mu + 0.5 * T * sig ** 2)  # Set the relevant zone of study and create the mapping between qubit and option price, and
    # generate the target lognormal distribution within the interval
    variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)
    values = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), qu)
    pdf = log_normal(values, mu, sig * np.sqrt(T))

    return (values, pdf), (mu, mean, variance)

def load_quantum_sim(qu, S0, sig, r, T):
    """
    Function to create quantum circuit for the unary representation to return an approximate probability distribution.
        This function is thought to be the prelude of run_quantum_sim
    :param qu: Number of qubits
    :param S0: Initial price
    :param sig: Volatility
    :param r: Interest rate
    :param T: Maturity date
    :return: quantum circuit, backend, (values of prices, prob. distri. function), (mu, mean, variance)
    """
    mu = (r - 0.5 * sig ** 2) * T + np.log(S0) # Define all the parameters to be used in the computation
    mean = np.exp(mu + 0.5 * T * sig ** 2) # Set the relevant zone of study and create the mapping between qubit and option price, and generate the target lognormal distribution within the interval
    variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)
    values = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), qu)
    pdf = log_normal(values, mu, sig * np.sqrt(T))
    (values, pdf), (mu, mean, variance) = get_pdf(qu, S0, sig, r, T)
    q, ancilla, circuit = create_qc(qu)
    lognormal_parameters = rw_parameters(qu, pdf) # Solve for the parameters needed to create the target lognormal distribution
    circuit.add(rw_circuit(qu, lognormal_parameters)) # Build the probaility loading circuit with the adjusted parameters
    circuit.add(measure_probability(q)) #Circuit to test the precision of the probability loading algorithm

    return circuit, (values, pdf), (mu, mean, variance)

def run_quantum_sim(qubits, circuit, shots):
    """
    Function to execute quantum circuit for the unary representation to return an approximate probability distribution.
        This function is thought to be preceded by load_quantum_sim
    :param qubits: Number of qubits
    :param qc: quantum circuit
    :param dev: backend
    :param shots: number of measurements
    :param basis_gates: native gates of the device
    :param noise_model: noise model
    :return: approximate probability distribution
    """
    result = circuit(nshots=shots)
    frequencies = result.frequencies(binary=True, registers=False)

    prob_sim = extract_probability(qubits, frequencies, shots)

    return prob_sim

def payoff_circuit(qubits, ancilla, K, S):
    """
    Circuit that codifies the expected payoff of the option into the probability of a
    single ancilla qubit
    :param qubits: Number of qubits
    :param K: strike
    :param S: prices
    :return: quantum circuit
    """
    for i in range(qubits): #Determine the first qubit's price that
        qK = i              #surpasses the strike price
        if K<S[i]:
            break
    for i in range(qK, qubits):                              #Control-RY rotations controled by states
        angle = 2 * np.arcsin(np.sqrt((S[i]-K)/(S[qubits-1]-K))) #with higher value than the strike
        yield gates.RY(ancilla, angle).controlled_by(i)                    #targeting the ancilla qubit

def payoff_circuit_inv(qubits, ancilla, K, S):
    """
    Circuit that codifies the expected payoff of the option into the probability of a
    single ancilla qubit, INVERTED!
    :param qubits: Number of qubits
    :param K: strike
    :param S: prices
    :return: quantum circuit
    """
    for i in range(qubits): #Determine the first qubit's price that
        qK = i              #surpasses the strike price
        if K<S[i]:
            break
    for i in range(qK, qubits):                              #Control-RY rotations controled by states
        angle = 2 * np.arcsin(np.sqrt((S[i]-K)/(S[qubits-1]-K))) #with higher value than the strike
        yield gates.RY(ancilla, -angle).controlled_by(i)                     #targeting the ancilla qubit

def measure_payoff(q, ancilla):
    """
    Function to measure the expected payoff of the option into the probability of a
    single ancilla qubit
    :param qubits: number of qubits
    :return: circuit of measurements
    """
    yield gates.M(*(q+[ancilla]), register_name='payoff')

def load_payoff_quantum_sim(qu, S0, sig, r, T, K):
    """
    Function to create quantum circuit for the unary representation to return an approximate probability distribution.
        This function is thought to be the prelude of run_payoff_quantum_sim
    :param qu: Number of qubits
    :param S0: Initial price
    :param sig: Volatility
    :param r: Interest rate
    :param T: Maturity date
    :return: quantum circuit, backend, prices
    """

    mu = (r - 0.5 * sig ** 2) * T + np.log(S0)  # Define all the parameters to be used in the computation
    mean = np.exp(mu + 0.5 * T * sig ** 2)  # Set the relevant zone of study and create the mapping between qubit and option price, and
                                # generate the target lognormal distribution within the interval
    variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)
    S = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), qu)
    ln = log_normal(S, mu, sig * np.sqrt(T))
    q, ancilla, circuit = create_qc(qu)
    lognormal_parameters = rw_parameters(qu, ln)  # Solve for the parameters needed to create the target lognormal distribution
    circuit.add(rw_circuit(qu, lognormal_parameters))  # Build the probaility loading circuit with the adjusted parameters
    circuit.add(payoff_circuit(qu, ancilla, K, S))
    circuit.add(measure_payoff(q, ancilla))

    return circuit, S


def run_payoff_quantum_sim(qu, circuit, shots, S, K):
    """
    Function to execute quantum circuit for the unary representation to return an approximate payoff.
        This function is thought to be preceded by load_quantum_sim
    :param qubits: Number of qubits
    :param qc: quantum circuit
    :param dev: backend
    :param shots: number of measurements
    :param basis_gates: native gates of the device
    :param noise_model: noise model
    :return: approximate payoff
    
    """
    job_payoff_sim = circuit(nshots=shots)
    counts_payoff_sim = job_payoff_sim.frequencies(binary=True, registers=False)
    ones=0
    zeroes=0
    for key in counts_payoff_sim.keys(): # Post-selection
        unary = 0
        for i in range(0,qu):
            unary+=int(key[i])
        if unary==1:
            if int(key[qu])==0:
                zeroes+=counts_payoff_sim.get(key)
            else:
                ones+=counts_payoff_sim.get(key)

    qu_payoff_sim = ones * (S[qu - 1]-K) / (ones+zeroes)

    return qu_payoff_sim

def test_inversion_payoff(qu, S0, sig, r, T, K, shots):
    mu = (r - 0.5 * sig ** 2) * T + np.log(S0)  # Define all the parameters to be used in the computation
    mean = np.exp(
        mu + 0.5 * T * sig ** 2)  # Set the relevant zone of study and create the mapping between qubit and option price, and
    # generate the target lognormal distribution within the interval
    variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)
    S = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), qu)
    ln = log_normal(S, mu, sig * np.sqrt(T))
    lognormal_parameters = rw_parameters(qu,
                                         ln)  # Solve for the parameters needed to create the target lognormal distribution
    prob_loading = rw_circuit(qu,
                              lognormal_parameters)  # Build the probaility loading circuit with the adjusted parameters
    prob_loading_inv = rw_circuit_inv(qu,
                              lognormal_parameters)

    payoff = payoff_circuit(qu, K, S)
    payoff_inv = payoff_circuit_inv(qu, K, S)
    q, ancilla, circuit = create_qc(qu)
    circuit.add(rw_circuit(qu, lognormal_parameters))
    circuit.add(payoff_circuit(qu, K, S))
    circuit.add(payoff_circuit_inv(qu, K, S))
    circuit.add(rw_circuit_inv(qu, lognormal_parameters))
    circuit.add(measure_payoff(q, ancilla))

    job_payoff_sim = circuit(nshots=shots)
    # and sample the ancilla qubit
    counts_payoff_sim = job_payoff_sim.frequencies(binary=True, registers=False)

    return counts_payoff_sim

def diff_qu_cl(qu_payoff_sim, cl_payoff):
    """
    Comparator of quantum and classical payoff
    :param qu_payoff_sim: quantum approximation for the payoff
    :param cl_payoff: classical payoff
    :return: Relative error
    """

    error = (100 * np.abs(qu_payoff_sim - cl_payoff) / cl_payoff)

    return error

def diffusion_operator(qubits):
    if qubits%2==0:
        mid = int(qubits/2)
    else:
        mid = int((qubits-1)/2) #The random walk starts from the middle qubit
    yield gates.X(qubits)
    yield gates.H(qubits)
    yield gates.CNOT(mid, qubits)
    yield gates.H(qubits)
    yield gates.X(qubits)


def oracle_operator(qubits):
    yield gates.Z(qubits)
    
def Q(qu, ancilla, K, S, lognormal_parameters):
    yield oracle_operator(qu)
    yield payoff_circuit_inv(qu, ancilla, K, S)
    yield rw_circuit_inv(qu, lognormal_parameters, X=False)
    yield diffusion_operator(qu)
    yield rw_circuit(qu, lognormal_parameters, X=False)
    yield payoff_circuit(qu, ancilla, K, S)

def load_Q_operator(qu, iterations, S0, sig, r, T, K):
    iterations = int(iterations)
    mu = (r - 0.5 * sig ** 2) * T + np.log(S0)  # Define all the parameters to be used in the computation
    mean = np.exp(
        mu + 0.5 * T * sig ** 2)  # Set the relevant zone of study and create the mapping between qubit and option price, and
    # generate the target lognormal distribution within the interval
    variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)
    S = np.linspace(max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), qu)
    ln = log_normal(S, mu, sig * np.sqrt(T))
    lognormal_parameters = rw_parameters(qu,
                                         ln)  # Solve for the parameters needed to create the target lognormal distribution
    
    q, ancilla, circuit = create_qc(qu)
    
    circuit.add(rw_circuit(qu, lognormal_parameters))
    circuit.add(payoff_circuit(qu, ancilla, K, S))
    for i in range(iterations):
        circuit.add(Q(qu, ancilla, K, S, lognormal_parameters))
    circuit.add(measure_payoff(q, ancilla))
    return circuit # Es posible que estas dos funciones puedan unirse

def run_Q_operator(qu, circuit, shots):
    job_payoff_sim = circuit(nshots=shots)
    counts_payoff_sim = job_payoff_sim.frequencies(binary=True, registers=False)
    ones = 0
    zeroes = 0
    for key in counts_payoff_sim.keys():  # Post-selection
        unary = 0
        for i in range(0, qu):
            unary += int(key[i])
        if unary == 1:
            if int(key[qu]) == 0:
                zeroes += counts_payoff_sim.get(key)
            else:
                ones += counts_payoff_sim.get(key)

    return ones, zeroes






