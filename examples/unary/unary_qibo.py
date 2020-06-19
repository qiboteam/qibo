from aux_functions import classical_payoff
import os
from qibo.models import Circuit
from qibo import gates

class Unary():
    """
    Class to perform option pricing in a unary basis
    """

    def __init__(self, data, qubits):
        """
        Initialize
        :param data: (S0, sig, r, T, K)
        :param max_gate_error: Maximum single-qubit gate error allowed (recommended: 0.005)
        :param steps: Number of error steps (recommended: 51, 101)
        """
        self.data = data
        self.S0, self.sig, self.r, self.T, self.K = self.data
        self.cl_payoff = classical_payoff(self.S0, self.sig, self.r, self.T, self.K)
        self.qubits = qubits
        try:
            """
            Create folder with results
            """
            os.makedirs(name_folder_data(self.data))
        except:
            pass

    def distributor(self):
        values, pdf = get_pdf(self.qubits, self.S0, self.sig, self.r, self.T)
        parameters = rw_parameters(self.qubits, pdf)
        ampl_distributor = rw_circuit(self.qubits, parameters)

def rw_circuit(qubits, parameters, X=True):
    """
    Circuit that emulates the probability sharing between neighbours seen usually
    in Brownian motion and stochastic systems
    :param qubits: number of qubits of the circuit
    :param parameters: parameters for performing the circuit
    :return: Quantum circuit
    """
    C = Circuit(qubits+1)
    if qubits%2==0:
        mid1 = int(qubits/2)
        mid0 = int(mid1-1)
        C.add(gates.X(mid1))
        #SWAP-Ry gates
        #-------------------------------------------------
        C.add(gates.fSim(mid1, mid0, parameters[mid0 / 2], 0)) # Puede que el /2 no sea correcto
        #-------------------------------------------------
        for i in range(mid0):
            #SWAP-Ry gates
            #-------------------------------------------------
            C.add(gates.fSim(mid0 - i, mid0 - i - 1, parameters[mid0 - i - 1] / 2, 0))
            #-------------------------------------------------
            C.add(gates.fSim(mid1 + i, mid1 + i + 1, parameters[mid1 + i] / 2, 0))
            #-------------------------------------------------
    else:
        mid = int((qubits-1)/2) #The random walk starts from the middle qubit
        C.add(gates.X(mid))
        for i in range(mid):
            #SWAP-Ry gates
            #-------------------------------------------------
            C.add(gates.fSim(mid - i, mid - i - 1, parameters[mid - i - 1] / 2, 0))
            #-------------------------------------------------
            C.add(gates.fSim(mid + i, mid + i + 1, parameters[mid + i] / 2, 0))
            #-------------------------------------------------
    return C

def rw_parameters(qubits, pdf):
    """
    Solving for the exact angles for the random walk circuit that enables the loading of
    a desired probability distribution function
    :param qubits: number of qubits of the circuit
    :param pdf: probability distribution function to emulate
    :return: set of parameters
    """
    if qubits % 2 == 0:
        mid = qubits // 2
    else:
        mid = (qubits - 1) // 2  # Important to keep track of the centre
    last = 1
    parameters = []
    for i in range(mid - 1):
        angle = 2 * np.arctan(np.sqrt(pdf[i] / (pdf[i + 1] * last)))
        parameters.append(angle)
        last = (np.cos(angle / 2)) ** 2  # The last solution is needed to solve the next one
    angle = 2 * np.arcsin(np.sqrt(pdf[mid - 1] / last))
    parameters.append(angle)
    last = (np.cos(angle / 2)) ** 2
    for i in range(mid, qubits - 1):
        angle = 2 * np.arccos(np.sqrt(pdf[i] / last))
        parameters.append(angle)
        last *= (np.sin(angle / 2)) ** 2
    return parameters