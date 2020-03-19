# QIBO

## General overview

The QIBO framework in this repository implements a
common system to deal with classical hardware and future
quantum hardware.

The code abstraction is located in `qibo/base`.
The default simulation engine is implemented using tensorflow and is located in `qibo/tensorflow`.

Other backends can be implemented by following the tensorflow example and adding a switcher in `qibo/config.py`.

Regression tests, which are run by the continous integration workflow are stored in `qibo/tests`. These tests contain several examples about how to use qibo.

The `qibo/benchmarks` folder contains benchmark code that has been implemented so far for specific applications.

## Installation

In order to install you can simply clone this repository with
```bash
git clone git@github.com:Quantum-TII/qibo.git
```

and then proceed with the installation with:
```
python setup.py install
```
if you prefer to keep changes always synchronized with the code then install using the `develop` option:
```bash
python setup.py develop
```

## Examples

Here short how to examples.

### How to write a circuit?

Here an example with 2 qubits:
```python
import numpy as np
from qibo.models import Circuit
from qibo import gates

init_state = np.ones(4) / 2.0
c = Circuit(2)
c.add(gates.X(0))
c.add(gates.X(1))
c.add(gates.CRZ(0, 1, 0.1234))
r1 = c.execute(init_state).numpy()
```

If you are planning to freeze the circuit and just query for different initial states then you can use the `Circuit.compile` method which will improve the evaluation performance, e.g.:
```python
import numpy as np
from qibo.models import Circuit
from qibo import gates

c = Circuit(2)
c.add(gates.X(0))
c.add(gates.X(1))
c.add(gates.CRZ(0, 1, 0.1234))
c.compile()

for i in range(100):
    init_state = np.ones(4) / 2.0 + i
    c(init_state)
```

### Supported gates

The following gates can be accessed as attributes of the `gates` module:
* Basic one qubit gates: `H`, `X`, `Y`, `Z`, `Iden`. Take as argument the index of the qubit they act on.
* Parametrized one qubit rotations: `RX`, `RY`, `RZ`.  Take as argument the index of the qubit they act on and the value of the parameter `theta`.
* Two qubit gates: `CNOT`, `SWAP`, `CRZ`. Take as argument the indices of the two qubits and the `theta` parameter for `CRZ`. For controlled gates, the first qubit given is the control and the second is the target.
* Three qubit gate: `TOFFOLI`. The first two qubits are controls and the third qubit is the target.
* Arbitrary unitary gate: `Unitary`. It takes as input a matrix (numpy array or Tensorflow tensor) and the target qubit ids. For example `Unitary(np.array([[0, 1], [1, 0]]), 0)` is equivalent to `X(0)`. This gate can act to arbitrary number of qubits. There is no check that the given matrix is unitary.
* The `Flatten` gate can be used to input a specific state vector. It takes as input a list/array of the amplitudes.

All gates support the `controlled_by` that allows to control them on an arbitrary number of qubits. For example 
* `gates.X(0).controlled_by(1, 2)` is equivalent to `gates.Toffoli(1, 2, 0)`,
* `gates.RY(0, np.pi).controlled_by(1, 2, 3)` applies the Y-rotation to qubit 0 or qubits 1, 2 and 3 are all 1,
* `gates.SWAP(0, 1).controlled_by(3, 4)` swaps qubits 0 and 1 if qubits 3 and 4 are both 1.

`controlled_by` cannot be used on gates that are already controlled.

### How to write a VQE?

The VQE requires an ansatz function and a Hamiltonian object.
There are examples of VQE optimization in `src/qibo/benchmarks`:
- `vqe_benchmark.py`: a simple example with the XXZ model.
- `adaptive_vqe_benchmark.py`: an adaptive example with the XXZ model.

Here a simple example using the Heisenberg XXZ model:
```python
import numpy as np
from qibo.models import Circuit, VQE
from qibo import gates
from qibo.hamiltonians import XXZ

nqubits = 6
layers  = 4

def ansatz(theta):
    c = Circuit(nqubits)
    index = 0
    for l in range(layers):
        for q in range(nqubits):
            c.add(gates.RY(q, theta[index]))
            index+=1
        for q in range(0, nqubits-1, 2):
            c.add(gates.CRZ(q, q+1, 1))
        for q in range(nqubits):
            c.add(gates.RY(q, theta[index]))
            index+=1
        for q in range(1, nqubits-2, 2):
            c.add(gates.CRZ(q, q+1, 1))
        c.add(gates.CRZ(0, nqubits-1, 1))
    for q in range(nqubits):
        c.add(gates.RY(q, theta[index]))
        index+=1
    return c()

hamiltonian = XXZ(nqubits=nqubits)
initial_parameters = np.random.uniform(0, 2*np.pi,
                                        2*nqubits*layers + nqubits)
v = VQE(ansatz, hamiltonian)
best, params = v.minimize(initial_parameters, method='BFGS', options={'maxiter': 1})

```
