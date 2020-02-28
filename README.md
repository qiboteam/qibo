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


### How to write a VQE?

The VQE requires an ansatz function and a Hamiltonian object.
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