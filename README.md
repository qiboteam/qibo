![logo](doc/source/qibo_logo.svg)

![Tests](https://github.com/Quantum-TII/qibo/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/Quantum-TII/qibo/branch/master/graph/badge.svg?token=1EKZKVEVX0)](https://codecov.io/gh/Quantum-TII/qibo)

QIBO is a Python library for classical simulation of quantum algorithms.

Some of the key features of QIBO are:
- A standard interface for the implementation and extension of quantum algorithms.
- Modular implementation on GPU and CPU with multi-threading support. 
- Availability of multi-GPU distributed approach for circuit simulation.
- Full support of double precision simulation.

## Documentation

QIBO documentation is available at [qibo.readthedocs.io](http://34.240.99.72/). (usr: qiboteam, pwd: qilimanjaro)

- [Installation](http://34.240.99.72/#installation)
- [Documentation](http://34.240.99.72)
- [Components](http://34.240.99.72/qibo.html)
- [Examples](http://34.240.99.72/examples.html)
- [Benchmarks](http://34.240.99.72/benchmarks.html)

## Minimum Working Examples

A simple [Quantum Fourier Transform (QFT)](https://en.wikipedia.org/wiki/Quantum_Fourier_transform) example to test your installation:
```python
from qibo.models import QFT

# Create a QFT circuit with 15 qubits
circuit = QFT(15)

# Simulate final state wavefunction default initial state is |00>
final_state = c()
```

Here another example with more gates and shots simulation:

```python
import numpy as np
from qibo.models import Circuit
from qibo import gates

# Construct the circuit
c = Circuit(2)

# Add some gates
c.add(gates.H(0))
c.add(gates.H(1))

# Define an initial state (optional - default initial state is |00>)
initial_state = np.ones(4) / 2.0

# Execute the circuit and obtain the final state
final_state = c.execute(initial_state) # c(initial_state) also works

# should print `np.array([1, 0, 0, 0])`
print(final_state.numpy())
```

In both cases, the simulation will run in a single device CPU or GPU in double precision `complex128`.

## Citation policy

If you use the package please cite the following references:
- https://doi.org/xx.xxxx/zenodo.xxxxx
- https://arxiv.org/abs/xxxx.xxxxx
