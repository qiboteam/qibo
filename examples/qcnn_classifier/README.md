# Quantum Convolutional Neural Network Classifier

Code at: [https://github.com/qiboteam/qibo/tree/master/examples/qcnn_classifier]
(https://github.com/qiboteam/qibo/tree/master/examples/qcnn_classifier).

## Problem overview
This tutorial implements a simple Quantum Convolutional Neural Network (QCNN), which is a translationally invariant algorithm analogous to the classical convolutional neural network. This example demonstrates the use of the QCNN as a quantum classifier, which attempts to classify ground states of a translationally invariant quantum system, the transverse field Ising model, based on whether they are in the ordered or disordered phase. The (randomized) statevector data provided are those of a 4-qubit system. Accompanying each state is a label: +1 (ordered phase) or -1 (ordered phase).
Through the sequential reduction of entanglement, this network is able to perform classification from the final measurement of a single qubit.

Workflow of QCNN model:
![workflow](images/workflow.PNG)

Schematic of QCNN model:
![schematic](images/structure.PNG)

Convolutional layer for 4 qubits as an example:
![convolution](images/convolution_4qubits.PNG)

Pooling layer for 4 qubits as an example:
![pooling](images/pooling_4qubits.PNG)

where in the above, $R(\theta_{i,j,k}) = RZ(\theta_k) RY(\theta_j) RX(\theta_i)$:
![R](images/RxRyRz.PNG)

$U_{q_a, q_b}(\theta_{i,j,k}) = RXX(\theta_k) RYY(\theta_j) RZZ(\theta_i)$ is a two-qubit gate acting on qubits $q_a$ and $q_b$:
![U](images/U.PNG)

and $R^{\dagger}(\theta_{i,j,k}) = RX(-\theta_i) RY(-\theta_j) RZ(-\theta_k)$:
![RT](images/RT.PNG)

## How to use the QCNN class
For more details on the QuantumCNN class, please refer to the documentation. Here we recall some of the necessary arguments when instantiating a QuantumCNN object:
- `nqubits` (int): number of quantum bits. It should be larger than 2 for the model to make sense. 
- `nlayers` (int): number of layers of the QCNN variational ansatz.
- `nclasses` (int): number of classes of the training set (default=2).
- `params`: list to initialise the variational parameters (default=None).

After creating the object, one can proceed to train the model. For this, the `QuantumCNN.minimize` method can be used with the following arguments (refer to the documentation for more details)"
- `init_theta`: list or numpy.array with the angles to be used in the circuit
- `data`: the training data
- `labels`: numpy.array containing the labels for the training data
- `nshots` (int):number of runs of the circuit during the sampling process (default=10000)
- `method` (string): str 'classical optimizer for the minimization'. All methods from scipy.optimize.minmize are suported (default='Powell').

Here is how we create QuantumCNN object. The user should include necessary packages:

```python
from qibo.models.qcnn import QuantumCNN
from qibo import gates
import random
import numpy as np

import qibo
qibo.set_backend("numpy")
```

Here we provide the ground states of 4-qubit TFIM in data folder. To define data and labels:

```
data = np.load('nqubits_4_data_shuffled_no0.npy')
labels = np.load('nqubits_4_labels_shuffled_no0.npy')
labels = np.transpose(np.array([labels])) # restructure to required array format
```

Structure of data and labels are like:
![data_labels](images/data_labels.PNG)


Define circuit:
```
test = QuantumCNN(nqubits=4, nlayers=1, nclasses=2)
testcircuit = test._circuit
testcircuit.draw()
```
draw() is used to visualize the circuit construction.

Initialize model parameters:
```
testbias = np.zeros(test.measured_qubits)
testangles = [random.uniform(0,2*np.pi) for i in range(21*2)]
init_theta = np.concatenate((testbias, testangles))
```
Train model with optimize parameters:
```
result = test.minimize(init_theta, data=data, labels=labels, nshots=10000, method='Powell')
```

Generate predictions from optimized model:
```
predictions = []
for n in range(len(data)):
    predictions.append(test.Predictions(testcircuit, result[1], data[n], nshots=10000)[0])
```

Result visualization:
```
from sklearn import metrics
actual = [np.sign(label) for label in labels]
predicted = [np.sign(prediction) for prediction in predictions]
confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [1, -1])
cm_display.plot()
plt.show()

test.Accuracy(labels,predictions)
```
![result](images/result_confusion_matrix.PNG)

In this example, we achieved an accuracy of 0.925. 
