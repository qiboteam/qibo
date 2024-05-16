import copy
import time
import argparse
import numpy as np

from scipy.optimize import minimize
import qibo
from qibo import gates, hamiltonians, models, quantum_info 

# import qcnn # CKG: strange
#from qcnn import QuantumCNN # CKG: Use uantumCNN under the local qcnn.py
from qibo.models.qcnn import QuantumCNN

class QuantumAutoencoder:
    def __init__(self, nqubits, layers, compress, lambdas, maxiter,method,encoderType,backendname):
        self.nqubits = nqubits
        self.layers = layers
        self.compress = compress
        self.lambdas = lambdas
        self.maxiter = maxiter

        # ckgan:  I. This one calls its inner function
        self.encoder = self.encoder_hamiltonian_simple()

        # ckgan:  II. This one calls its inner function
        self.ising_groundstates = self.prepare_ising_groundstates()


        self.method = method  # You can make this configurable
        self.costFunType = encoderType
        self.qc = None

        # ckgan:  III. This one calls its inner function
        self.cf_circuit = self.cost_function_setCircuit()


        self.cf_circuit_inv = None #self.cf_circuit.invert()
        self.reset_circuit = None
        self.rs = 2  # Random seed, can also be made configurable

        # ckgan:  Hard coded here!!!
        self.cfTypes =["qcnn","qcnn.ry","qcnn.real","HEA"]

        self.backendName = backendname
        qibo.set_backend(backendname) 
        

    def encoder_hamiltonian_simple(self):

        """Creates the encoding Hamiltonian.
        Args:
            nqubits (int): total number of qubits.
            ncompress (int): number of discarded/trash qubits.

        Returns:
            Encoding Hamiltonian.
        """

        # ckgan:  hamiltonians is found in qibo package
        m0 = hamiltonians.Z(self.compress).matrix

        m1 = np.eye(2 ** (self.nqubits - self.compress), dtype=m0.dtype)
        #ham = hamiltonians.Hamiltonian(self.nqubits, np.kron(m1, m0))

        # ckgan: again, hamiltonians is found in qibo package
        ham = hamiltonians.Hamiltonian(self.nqubits, np.kron(m0, m1))
        return 0.5 * (ham + self.compress)


    # ckgan: to be called by the __init__ above. 
    def prepare_ising_groundstates(self):
        groundstates = []
        for lamb in self.lambdas:
            ising_ham = -1 * hamiltonians.TFIM(self.nqubits, h=lamb)
            groundstates.append(ising_ham.eigenvectors()[0])
        return groundstates

    # ckgan: to be called by the __init__ above. 
    def cost_function_setCircuit(self):
     
        if(self.costFunType=='qcnn'):

            # ckgan: QuantumCNN module is found in the current directory qcnn/QuantumCNN !  

            self.qc = QuantumCNN(nqubits=self.nqubits, nlayers=self.layers, nclasses=2)
            circuit = self.qc._circuit
        if(self.costFunType=='qcnn.real'):
            self.qc = QuantumCNN(nqubits=self.nqubits, nlayers=self.layers, nclasses=2, ifReal=True)
            circuit = self.qc._circuit
        elif(self.costFunType=='qcnn.ry'):

            # print("ckg:  I am in qcnn.ry")
            c = models.Circuit(2)
            for i in range(2): 
                c.add(gates.RY(i,0))
            self.qc = QuantumCNN(nqubits=self.nqubits, nlayers=self.layers, nclasses=2,twoqubitansatz=c,ifReal=True)
            circuit = self.qc._circuit 
        else: #default Hardware efficient ansantz (HEA)
            circuit = models.Circuit(self.nqubits)   
            for l in range(self.layers):
                for q in range(self.nqubits):
                    circuit.add(gates.RY(q, theta=0))
                for q in range(0, self.nqubits - 1, 2):
                    circuit.add(gates.CZ(q, q + 1))
                for q in range(self.nqubits):
                    circuit.add(gates.RY(q, theta=0))
                for q in range(1, self.nqubits - 2, 2):
                    circuit.add(gates.CZ(q, q + 1))
                circuit.add(gates.CZ(0, self.nqubits - 1))
            for q in range(self.nqubits):
                circuit.add(gates.RY(q, theta=0))
        return circuit            

    def update_cf_circuit_invert(self):
        self.cf_circuit_inv = self.cf_circuit.invert()

    def cost_function(self, params, count):
        """Evaluates the cost function to be minimized.

        Args:
            params (array or list): values of the parameters.

        Returns:
            Value of the cost function.
        """
        cost = 0
        if(self.costFunType in self.cfTypes[:3]):
            #print("to compare paramters:\n")
            #print(params)
            self.qc.set_circuit_params(params)
            self.qc.set_circuit_params(params,has_bias=True)
            self.cf_circuit = self.qc._circuit 
            #print(self.cf_circuit.get_parameters())
            #print("\n")
        else:
            self.cf_circuit.set_parameters(params)
        for state in self.ising_groundstates:
            #final_state = self.cf_circuit(np.copy(state), nshots = 10000)
            final_state = self.cf_circuit(np.copy(state))
            cost += np.real(self.encoder.expectation(final_state.state()))

        if count[0] % 50 == 0:
            print(count[0], cost / len(self.ising_groundstates))
        count[0] += 1

        return cost / len(self.ising_groundstates)

    def setup_reset_circuit(self):
        self.reset_circuit = models.Circuit(self.nqubits, density_matrix=True)
        for i in range(self.nqubits - self.compress):
            resetq = gates.ResetChannel(i, [1.0, 0.])
            self.reset_circuit.add(resetq)

    def add_gates_from_circuit(self, circuit, source_circuit):
        #circuit = models.Circuit(self.nqubits, density_matrix=True)
        for gate in source_circuit.queue:
            if(gate.name == "measure"): continue
            circuit.add(copy.copy(gate))
            #circuit.add(gate)

    def run_optimization(self):
        time0 = time.time()
        np.random.seed(self.rs)
        count = [0]

        # if in qcnn.
        # WEIRD WEIRD.. printing of missing variable does not crash the code but other part of code is entered! WHY? WHY? So HOW CAN WE DEBUG??
        # print("ckgan: costFunType is ",self.contFunType)
        print("ckgan: costFunType is ",self.costFunType)
        print("ckgan:  cfTypes are:")
        print(self.cfTypes)
        print("ckgan: .....")

        if(self.costFunType in self.cfTypes[:3]): # no bug!

            nparams = self.qc.nparams_layer*self.layers  # nparams for qcnn architetures! since the parameters are correlated. 

            # bias only in QCNN. optimizing parameter by adding some constants..
            bias = np.zeros(self.qc.measured_qubits)
            initial_params = np.random.uniform(0, 2 * np.pi, nparams)
            initial_params = np.concatenate((bias,initial_params))
        else: # this is general one
            nparams = 2 * self.nqubits * self.layers + self.nqubits 
            initial_params = np.random.uniform(0, 2 * np.pi, nparams)

        result = minimize(
            lambda p: self.cost_function(p, count),
            initial_params,
            method=self.method,
            options={"maxiter": self.maxiter, "maxfun": 2.0e3},
        )
        time1 = time.time()
        print("time0 is ",time0)
        print("time1 is ",time1)
        print("time1-time0 is ",time1-time0)


        print("Final parameters: ", result.x)
        print("Final cost function: ", result.fun)
        fname="results.txt"
        listtoprint=["[backend,costFunType,str(nqubits),str(layers),str(compress),_method,str(nit),str(result.fun),str(len(data)),str(time1-time0),str(rs)]\n"]
        try:
            listtoprint+=[self.backendName, self.costFunType,str(self.nqubits),str(self.layers),str(self.compress),self.method,str(result.nit),str(result.fun),str(len(self.ising_groundstates)),str(time1-time0),str(self.rs)]
        except:
            listtoprint+=[self.backendName, self.costFunType,str(self.nqubits),str(self.layers),str(self.compress),self.method,"-1",str(result.fun),str(len(self.ising_groundstates)),str(time1-time0),str(self.rs)]

        # the printing should not be here!
        self.printInfo(fname,listtoprint)

    # ckgan: This is to be called after a successful initialization !
    def run_full_circuit(self, params=None):

        # ckgan: when we run optimization, we try to get the set of thetas
        if(params == None):
            #np.random.seed(self.rs)
            self.run_optimization()

        else: # ??? so how to run optimization once we enter this loop? We modify the circuit since we have a new set of optimized parameters?
            if(self.costFunType in self.cfTypes[:3]):
                #print("to compare paramters:\n")
                #print(params)
                #self.qc.set_circuit_params(params)
                self.qc.set_circuit_params(params,has_bias=True)
                self.cf_circuit = self.qc._circuit
                #print(self.cf_circuit.get_parameters())
                #print("\n")
            else:
                self.cf_circuit.set_parameters(params)
        self.update_cf_circuit_invert()
        self.setup_reset_circuit()

        cost = 0.
        circuit = models.Circuit(self.nqubits, density_matrix=True)
        
        #cannot add circuits with density_matrix=True and False
        '''circuit = self.cf_circuit
        circuit +=self.reset_circuit
        circuit +=self.cf_circuit_inv'''
        self.add_gates_from_circuit(circuit, self.cf_circuit) #encoder
        #print(circuit.get_parameters())
        #print(self.cf_circuit.get_parameters())
        self.add_gates_from_circuit(circuit, self.reset_circuit)
        self.add_gates_from_circuit(circuit, self.cf_circuit_inv) #decoder
    
        ''' ckgan: remove printing of circuits at this stage!
        print("encoder circuit:")
        print(self.cf_circuit.draw())
        print("decoder circuit:")
        print(self.cf_circuit_inv.draw())
        print("copied circuit:")
        print(circuit.draw())
        '''

        # ckgan: We loop through a bunch of ground states!
        # ckgan: But I do not understand...  I thought we need to evolve a state to get the final 
        # state and then compute fidelity. The same state has be evolved many times. And then we need to do the same for other states?
        for state in self.ising_groundstates:
            v_state = np.outer(np.copy(state),np.conjugate(state))
            final_state = circuit(np.copy(v_state))
            #print(v_state)
            #print(final_state.state())
            #cost += 1.-np.real(np.vdot(v_state, final_state.state()))
            #cost += quantum_info.infidelity(v_state,final_state)
            cost += 1.-quantum_info.fidelity(v_state,final_state.state())
        cost = cost/ len(self.ising_groundstates)
        print("encoder+decoder cost: ",cost)

    def printInfo(self,fname, xs):
        with open(fname, 'a') as f:
            for x in xs:
                f.write(x + '\t')
                print(x + '\t')
            f.write('\n')
    
