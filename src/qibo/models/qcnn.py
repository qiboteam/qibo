# Quantum Convolutional Neural Network (QCNN) for classification tasks.
# The QCNN model was originally proposed in: 
#   arXiv:1810.03787 <https://arxiv.org/abs/1810.03787> for the identification of quantum phases.

import numpy as np
from qibo import gates
from qibo.backends import get_backend
from qibo.models import Circuit
import sys

class QuantumCNN:
    # -------------------------------------------------------------------------
    #     Initializes the QuantumCNN object.
    #         nqubits (int): The number of qubits in the QCNN. Currently supports powers of 2.
    #         nlayers (int): The number of convolutional and pooling layers.
    #         nclasses (int): The number of classes to be classified.
    #         qcnntype:   original, real, ry (in the order of less parameters)
    #         params (np.ndarray, optional): initial list of variational parameters. 
    #           If not provided, all parameters will be initialized to zero.
    #         twoqubitansatz (qibo.models.circuit.Circuit, optional): A two-qubit ansatz for the 
    #           convolutional layers. If None, a default ansatz is used. Defaults to None.
    #         copy_init_state (bool, optional): Whether to copy the initial state for each shot 
    #           in the simulation. If None, the behavior depends on the backend. Defaults to None.
    def __init__( self, nqubits, nlayers, nclasses, qcnntype, twoqubitansatz=None, params=None, copy_init_state=None):

        self.nqubits = nqubits
        if self.nqubits <= 1:
            raise ValueError("nqubits must be larger than 1")

        self.nlayers = nlayers
        self.nclasses = nclasses
        self.qcnntype = qcnntype
        self.twoqubitansatz = twoqubitansatz

        if copy_init_state is None:
            if "qibojit" in str(get_backend()): # trigger copy states if the backend if qibojit
                self.copy_init_state = True
            else:
                self.copy_init_state = False

        if self.qcnntype == 'COMPLEX':
            self.nparams_conv = 15
            self.nparams_pool = 6
        elif self.qcnntype == 'REAL':
            self.nparams_conv = 5
            self.nparams_pool = 2
        elif self.qcnntype == 'RY':
            self.nparams_conv = len(self.twoqubitansatz.get_parameters())
            assert self.nparams_conv == 2
            self.nparams_pool = 2
        else:
            self.nparams_conv = len(self.twoqubitansatz.get_parameters())
            sys.exit('error: We are not supposed to enter here!')

        self.nparams_per_layer = self.nparams_conv + self.nparams_pool
        self.measured_qubits = int(np.ceil(np.log2(self.nclasses)))

        print('qcnntype= %s: nparams_conv= %d, nparams_pool= %d, nparams_per_layer= %d,'
               'measured_qubits= %d\n'%(self.qcnntype,self.nparams_conv,\
               self.nparams_pool,self.nparams_per_layer,self.measured_qubits))

        self._circuit = self.ansatz(nlayers, params=params)  # what is params doing here?

    # -------------------------------------------------------------------------
    #   Create a circuit implementing the QCNN variational ansatz.
    #   theta: list or numpy.array with the angles to be used in the circuit.
    #   nlayers: int number of layers of the varitional circuit ansatz.
    def ansatz(self, nlayers, params=None):

        nparams_conv = self.nparams_conv
        nparams_pool = self.nparams_pool
        nparams_per_layer = self.nparams_per_layer

        if params is None:
            irrthetas = [0 for _ in range(nlayers * nparams_per_layer)]
        else:
            # we initialize params with a predefine list
            # sys.exit('err: This loop should be not entered!')
            irrthetas = params

        nqubits = self.nqubits

        qubits = [_ for _ in range(nqubits)]
        circuit = Circuit(self.nqubits)

        # go through layer by layer, for each layer add conv and pool layers of shrinking length
        for layer in range(nlayers):
            # which qubit to insert
            cb = int(nqubits - nqubits/(2**layer) )        # conv_begin
            pb = int(nqubits - nqubits/(2**(layer + 1)) )  # pool_begin
            paramb = layer*nparams_per_layer  # param_begin
            # print('layer= %d: cb, pb, paramb = (%d, %d, %d)\n'%(layer,cb,pb,paramb))

            # print('cb=%d:pb=%d'%(cb,pb))
            # print('len(qubits[cb:pb]) = %d'%(len(qubits[cb:pb])))
            # print('len(qubits[pb:]) = %d'%(len(qubits[pb:])))
            # print('paramb + nparams_conv = %d'%(paramb + nparams_conv))
            # print('paramb + nparams_per_layer = %d'%(paramb + nparams_per_layer))
            # print('len(  irrthetas[paramb + nparams_conv : paramb + nparams_per_layer]  ) = %d'%( len(  irrthetas[paramb + nparams_conv : paramb + nparams_per_layer]   )))

            # One list from cb to pb, and another list from pb to the end
            # extract the correct segment to the calling method!
            circuit += self.quantum_conv_circuit( qubits[cb:], irrthetas[paramb : paramb + nparams_conv])
            # we pass in irreducible parameters
            circuit += self.quantum_pool_circuit( qubits[cb:pb], qubits[pb:], irrthetas[paramb + nparams_conv : paramb + nparams_per_layer],)

        # Measurements
        # print('*[nqubits - 1 - i for i in range(self.measured_qubits)] is ')
        # print(*[nqubits - 1 - i for i in range(self.measured_qubits)])
        circuit.add(gates.M(*[nqubits - 1 - i for i in range(self.measured_qubits)]))  # * here? 

        return circuit

    # -------------------------------------------------------------------------
    #   construct a single convolutional layer.
    #   bits: list qubit indices that the convolutional layer should apply to.
    #   irrthetas: list of angles to be used in the circuit.
    def quantum_conv_circuit(self, bits, irrthetas):
      circuit = Circuit(self.nqubits)

      # form pairs of (even,odd) such as (0,1), (2,3) etc
      for first, second in zip(bits[0::2], bits[1::2]):
        circuit += self.two_qubit_unitary([first, second], irrthetas)

      if len(bits) > 2: # form pairs of of (odd,even) such as (1,2), (3,4), we have must the number of qubits > 2 
        for first, second in zip(bits[1::2], bits[2::2] + [bits[0]]  ):  # the last even number is the first qubit so that we wrap around
          # print('bits[:] is ',bits[:])
          # print('(first,second)= (%d, %d)'%(first,second))
          # print('[bits[0]] = ',[bits[0]])
          circuit += self.two_qubit_unitary([first, second], irrthetas)
      return circuit

    # -------------------------------------------------------------------------
    #  construct a single pooling layer.
    #  source_bits: list of source qubits indices for the pooling layer.
    #  sink_bits: list sink qubits indices for the pooling layer.
    #  irrthetas: list angles to be used in the circuit.
    def quantum_pool_circuit(self, source_bits, sink_bits, irrthetas):
        circuit = Circuit(self.nqubits)
        for source, sink in zip(source_bits, sink_bits):
            # print('quantum_pool_circut, source= %d, sink= %d'%(source,sink))
            circuit += self.two_qubit_pool(source, sink, irrthetas)
        return circuit

    # -------------------------------------------------------------------------
    #  create a circuit consisting of two qubit unitaries added to the specified qubits.
    #  bits: the two qubits to apply the unitaries to
    #  irrthetas: length 15 array containing the parameters
    def two_qubit_unitary(self, bits, irrthetas):

        circuit = Circuit(self.nqubits)

        if self.qcnntype == 'COMPLEX':

          # print('len(irrthetas) is %d'%(len(irrthetas)))

          circuit += self.one_qubit_unitary(bits[0], irrthetas[0:3])
          circuit += self.one_qubit_unitary(bits[1], irrthetas[3:6])

          circuit.add(gates.RZZ(bits[0], bits[1], irrthetas[6]))
          circuit.add(gates.RYY(bits[0], bits[1], irrthetas[7]))
          circuit.add(gates.RXX(bits[0], bits[1], irrthetas[8]))

          circuit += self.one_qubit_unitary(bits[0], irrthetas[9:12])
          circuit += self.one_qubit_unitary(bits[1], irrthetas[12:])

        elif self.qcnntype == 'REAL': 
          # print('irrthetas is ',irrthetas)
          # print('irrthetas[0:1] is ',irrthetas[0:1])
          circuit += self.one_qubit_unitary(bits[0], irrthetas[0:1]) # Wenjun uses irrthetas[0:]
          circuit += self.one_qubit_unitary(bits[1], irrthetas[1:2])
          circuit.add(gates.RYY(bits[0], bits[1], irrthetas[2]))
          circuit += self.one_qubit_unitary(bits[0], irrthetas[3:4])
          circuit += self.one_qubit_unitary(bits[1], irrthetas[4:5])

        elif self.qcnntype == 'RY':
          circuit.add(self.twoqubitansatz.on_qubits(bits[0], bits[1]))
          circuit.set_parameters(irrthetas[0 : self.nparams_conv])
          # print('self.nparams_conv = ',self.nparams_conv)
        else:
          circuit.add(self.twoqubitansatz.on_qubits(bits[0], bits[1]))
          circuit.set_parameters(irrthetas[0 : self.nparams_conv])
          sys.exit('error: not supposed to enter here!')

        return circuit

    # -------------------------------------------------------------------------
    #     make a circuit enacting a rotation of the bloch sphere about the X,
    #     Y and Z axis, that depends on the values in `irrthetas`.
    #         bit: the qubit to apply the one-qubit unitaries to
    #         irrthetas: length 3 array containing the parameters
    # this is for the convolutional layer and also the pooling layer
    def one_qubit_unitary(self, bit, irrthetas):
        circuit = Circuit(self.nqubits)

        if self.qcnntype == 'COMPLEX':
          circuit.add(gates.RX(bit, irrthetas[0]))
          circuit.add(gates.RY(bit, irrthetas[1]))
          circuit.add(gates.RZ(bit, irrthetas[2]))
        elif self.qcnntype == 'REAL':
          circuit.add(gates.RY(bit, irrthetas[0]))  # For REAL, this is for both the convolutional layer and pooling layer
        elif self.qcnntype == 'RY':
          circuit.add(gates.RY(bit, irrthetas[0]))  # For RY, this is for the pooling layer only
        return circuit

    # -------------------------------------------------------------------------
    #     create a circuit to do a parameterized 'pooling' operation with controlled unitaries added tp the specified qubits, which
    #     attempts to reduce entanglement down from two qubits to just one.
    #         source_qubit: the control qubit.
    #         sink_qubit: the target qubit for the controlled unitaries.
    #         irrthetas: array with 6 elements containing the parameters.
    def two_qubit_pool(self, source_qubit, sink_qubit, irrthetas):
        pool_circuit = Circuit(self.nqubits)

        # seems contruct sink and source circuits and then join them.

        if self.qcnntype == 'COMPLEX':
          sink_basis_selector = self.one_qubit_unitary(sink_qubit, irrthetas[0:3])
          source_basis_selector = self.one_qubit_unitary(source_qubit, irrthetas[3:6])
        elif self.qcnntype == 'REAL':

          # print('len(irrthetas) = %d'%(len(irrthetas)))
          # sys.exit('stop here for a while!')
          sink_basis_selector = self.one_qubit_unitary(sink_qubit, irrthetas[0:1]) #  We only have two angles.
          source_basis_selector = self.one_qubit_unitary(source_qubit, irrthetas[1:2])

        elif self.qcnntype == 'RY':
          # sys.exit('ry: To be implemented!')
          sink_basis_selector = self.one_qubit_unitary(sink_qubit, irrthetas[0:1]) #  We only have two angles.
          source_basis_selector = self.one_qubit_unitary(source_qubit, irrthetas[1:2])
 
        else:
          print('Error: general case for two_qubit_pool.')
          sys.exit('We stop at two_qubit_pool')

        # can we swap the two lines below?
        pool_circuit += sink_basis_selector  # ? this makes sink takes 0, 1, 2, and source 4, 5, 6 for the original.  For 'REAL', sink takes 0, and source takes 1
        pool_circuit += source_basis_selector
        pool_circuit.add(gates.CNOT(source_qubit, sink_qubit)) # just one CNOT gate!
        pool_circuit += sink_basis_selector.invert()

        return pool_circuit

    # -------------------------------------------------------------------------
    #         theta: list or numpy.array with the biases and the angles to be used in the circuit.
    #     Returns:
    #         Circuit implementing the variational ansatz for angles "theta".
    def classifier_circuit(self, theta):
        bias = np.array(theta[0 : self.measured_qubits])
        angles = theta[self.measured_qubits :]

        # print('calling self.set_circuit_params in classifer_circuit')
        self.set_circuit_params(angles)
        return self._circuit

    # -------------------------------------------------------------------------
    #     Sets the parameters of the QCNN circuit. Can be used to load previously saved or optimized parameters.
    #         angles: the parameters to be loaded.
    #         has_bias: specify whether the list of angles contains the bias.
    def set_circuit_params(self, angles, has_bias=False):
        if not has_bias:  # has_bias=False
            params = list(angles)
        else: # has_bias=True
            self._optimal_angles = angles # why take everything, yet the following line takes part of it? Hence the object takes everything.
            params = list(angles[self.measured_qubits :])
            
        expanded_params = []
        nqubits = self.nqubits

        # fill in layer by layer...
        # print('!!! len(params) = %d'%len(params))

        # the idea is to peel from the irrthetas array and then fill in all angles in the circuit
        for layer in range(self.nlayers):

            paramb = layer * self.nparams_per_layer # we take block of nparams_per_layer each time we deal with a new layer

            # start anew for each layer
            conv_params = params[ paramb : paramb + self.nparams_conv]  # we have this new set
            pool_params = params[ paramb + self.nparams_conv : paramb + self.nparams_per_layer ]  # the rest is for pool

            # print('in set_circuit_params: self.nparams_conv = %d'%self.nparams_conv)
            # print('in set_circuit_params: self.nparams_per_layer = %d'%self.nparams_per_layer)
            # print('in set_circuit_params: paramb= %d, len(conv_params)=%d, len(pool_params)= %d'%(paramb, len(conv_params), len(pool_params)))


            if self.qcnntype == 'COMPLEX': # For original, we have 0, 1, 2 (for the bottom sink,   and the 4, 5, 6. 
                                            # (for the top source)? . Real: we have 0 (for sink), 1 (for source)
              pool_params += [-pool_params[2], -pool_params[1], -pool_params[0]]   # R^dagger (or the inverted) operations
            elif self.qcnntype == 'REAL':
              pool_params += [-pool_params[0]]  # wenjun's code. now, 0 or 1? 
            elif self.qcnntype == 'RY':
              pool_params += [-pool_params[0]]

            # first layer =0, then this nleft = nqubits, layer = 1, nleft = nqubits/2, ...
            nleft = nqubits / (2**layer)
            if nleft <= 2:
              pass
              # print('nleft = %d'%nleft)
              # sys.exit('stop here!')
            expanded_params += conv_params * int(nleft if nleft > 2 else 1)  # nleft=2 or 1, then just 1. why?
            expanded_params += pool_params * int(nleft / 2)

        # print('finally, in set_circuit_params:')

        # too many copies
        # print('len( expanded_params ) = %d'%(  len( expanded_params) ) )
        # we pass the control to the qibo's set_parameters
        self._circuit.set_parameters(expanded_params)

    # -------------------------------------------------------------------------
    #         theta: list or numpy.array with the biases to be used in the circuit.
    #         init_state: numpy.array with the quantum state to be classified.
    #         nshots: int number of runs of the circuit during the sampling process (default=10000).
    #     Returns:
    #         numpy.array() with predictions for each qubit, for the initial state.
    def predictions(self, circuit, theta, init_state, nshots=10000):
        bias = np.array(theta[0 : self.measured_qubits])
        if self.copy_init_state:
            init_state_copy = init_state.copy()
        else:
            init_state_copy = init_state
        circuit_exec = circuit(init_state_copy, nshots)
        result = circuit_exec.frequencies(binary=False)
        prediction = np.zeros(self.measured_qubits)

        for qubit in range(self.measured_qubits):
            for clase in range(self.nclasses):
                binary = bin(clase)[2:].zfill(self.measured_qubits)
                prediction[qubit] += result[clase] * (1 - 2 * int(binary[-qubit - 1]))

        return prediction / nshots + bias

    # -------------------------------------------------------------------------
    #         labels: list or numpy.array with the qubit labels of the quantum states to be classified.
    #         predictions: list or numpy.array with the qubit predictions for the quantum states to be classified.
    #     Returns:
    #         numpy.float32 with the value of the square-loss function.
    def square_loss(self, labels, predictions):
        loss = 0
        for l, p in zip(labels, predictions):
            for qubit in range(self.measured_qubits):
                loss += (l[qubit] - p[qubit]) ** 2

        return loss / len(labels)

    # -------------------------------------------------------------------------
    #         theta: list or numpy.array with the biases and the angles to be used in the circuit.
    #         data: numpy.array data[page][word]  (this is an array of kets).
    #         labels: list or numpy.array with the labels of the quantum states to be classified.
    #         nshots: int number of runs of the circuit during the sampling process (default=10000).
    #     Returns:
    #         numpy.float32 with the value of the square-loss function.
    def cost_function(self, theta, data=None, labels=None, nshots=10000):
        # UPDATE circuit!
        circ = self.classifier_circuit(theta)

        Bias = np.array(theta[0 : self.measured_qubits])
        predictions = np.zeros(shape=(len(data), self.measured_qubits))

        for i, text in enumerate(data):
            predictions[i] = self.predictions(circ, Bias, text, nshots)

        s = self.square_loss(labels, predictions)

        return s

    # -------------------------------------------------------------------------
    #         init_theta: list or numpy.array with the angles to be used in the circuit.
    #         data: the training data to be used in the minimization.
    #         labels: the corresponding ground truth for the training data.
    #         nshots: number of runs of the circuit during the sampling process (default=10000).
    #         method: str 'classical optimizer for the minimization'. All methods from qibo.optimizers.optimize are suported (default='Powell').
    #     Returns:
    #         numpy.float64 with value of the minimum found, numpy.ndarray with the optimal angles.
    def minimize( self, init_theta, data=None, labels=None, nshots=10000, method="Powell"):
        from qibo.optimizers import optimize

        # this is called one time, but cost_function will be invoked many times
        loss, optimal_angles, result = optimize( self.cost_function, init_theta, args=(data, labels, nshots), method=method)

        self._optimal_angles = optimal_angles

        # print('calling self.set_circuit_params in minimize()')
        self.set_circuit_params(optimal_angles[self.measured_qubits :])

        return loss, optimal_angles

    # -------------------------------------------------------------------------
    #         labels: numpy.array with the labels of the quantum states to be classified.
    #         predictions: numpy.array with the predictions for the quantum states classified.
    #         sign: if True, labels = np.sign(labels) and predictions = np.sign(predictions) (default=True).
    #         tolerance: float tolerance level to consider a prediction correct (default=1e-2).
    #     Returns:
    #         float with the proportion of states classified successfully.
    def accuracy(self, labels, predictions, sign=True, tolerance=1e-2):
        if sign:
            labels = [np.sign(label) for label in labels]
            predictions = [np.sign(prediction) for prediction in predictions]

        accur = 0
        for l, p in zip(labels, predictions):
            if np.allclose(l, p, rtol=0.0, atol=tolerance):
                accur += 1

        accur = accur / len(labels)

        return accur

    # -------------------------------------------------------------------------
    #     Produce predictions on new input state after the model is trained. Currently it only takes in one input data.
    #         init_state: the input state to be predicted.
    #         nshots (default=10000): number of shots.
    #     Returns:
    #         numpy.array() with predictions for each qubit, for the initial state.
    def predict(self, init_state, nshots=10000):
        return self.predictions(
            self._circuit, self._optimal_angles, init_state, nshots=nshots
        )
