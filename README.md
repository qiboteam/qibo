<img src="doc/source/qibo_logo.svg" width="60%">

![Tests](https://github.com/Quantum-TII/qibo/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/Quantum-TII/qibo/branch/master/graph/badge.svg?token=1EKZKVEVX0)](https://codecov.io/gh/Quantum-TII/qibo)
[![Documentation Status](https://readthedocs.org/projects/qibo/badge/?version=latest)](https://qibo.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/241307936.svg)](https://zenodo.org/badge/latestdoi/241307936)

Qibo is a Python library for classical simulation of quantum algorithms.

Some of the key features of Qibo are:
- A standard interface for the implementation and extension of quantum algorithms.
- Modular implementation on GPU and CPU with multi-threading support.
- Availability of multi-GPU distributed approach for circuit simulation.
- Full support of double precision simulation.

## Documentation

Qibo documentation is available at [qibo.readthedocs.io](https://qibo.readthedocs.io).

- [Installation](https://qibo.readthedocs.io/en/stable/installation.html)
- [Documentation](https://qibo.readthedocs.io/)
- [Components](https://qibo.readthedocs.io/en/stable/qibo.html)
- [Examples](https://qibo.readthedocs.io/en/stable/examples.html)
- [Benchmarks](https://qibo.readthedocs.io/en/stable/benchmarks.html)

## Minimum Working Examples

A simple [Quantum Fourier Transform (QFT)](https://en.wikipedia.org/wiki/Quantum_Fourier_transform) example to test your installation:
```python
from qibo.models import QFT

# Create a QFT circuit with 15 qubits
circuit = QFT(15)

# Simulate final state wavefunction default initial state is |00>
final_state = circuit()
```

Here another example with more gates and shots simulation:

```python
import numpy as np
from qibo.models import Circuit
from qibo import gates

c = Circuit(2)
c.add(gates.X(0))

# Add a measurement register on both qubits
c.add(gates.M(0, 1))

# Execute the circuit with the default initial state |00>.
result = c(nshots=100)
```

In both cases, the simulation will run in a single device CPU or GPU in double precision `complex128`.

## Citation policy

If you use the package please cite the following references:
- https://arxiv.org/abs/2009.01845
- https://doi.org/10.5281/zenodo.3997194
