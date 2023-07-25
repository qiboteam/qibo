![](doc/source/_static/qibo_logo_dark.svg)

![Tests](https://github.com/qiboteam/qibo/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/qiboteam/qibo/branch/master/graph/badge.svg?token=1EKZKVEVX0)](https://codecov.io/gh/qiboteam/qibo)
[![Documentation Status](https://readthedocs.org/projects/qibo/badge/?version=latest)](https://qibo.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/241307936.svg)](https://zenodo.org/badge/latestdoi/241307936)

Qibo is an open-source full stack API for quantum simulation and quantum hardware control.

Some of the key features of Qibo are:
- Definition of a standard language for the construction and execution of quantum circuits with device agnostic approach to simulation and quantum hardware control based on plug and play backend drivers.
- A continuously growing code-base of quantum algorithms applications presented with examples and tutorials.
- Efficient simulation backends with GPU, multi-GPU and CPU with multi-threading support.
- Simple mechanism for the implementation of new simulation and hardware backend drivers.

## Documentation

Qibo documentation is available [here](https://qibo.science).

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
from qibo import Circuit, gates

c = Circuit(2)
c.add(gates.X(0))

# Add a measurement register on both qubits
c.add(gates.M(0, 1))

# Execute the circuit with the default initial state |00>.
result = c(nshots=100)
```

In both cases, the simulation will run in a single device CPU or GPU in double precision `complex128`.

## Citation policy

If you use the package please refer to [the documentation](https://qibo.science/qibo/stable/appendix/citing-qibo.html#publications) for citation instructions.

## Supporters and collaborators

- Quantum Research Center, Technology Innovation Institute (TII), United Arab Emirates
- Università degli Studi di Milano (UNIMI), Italy.
- Istituto Nazionale di Fisica Nucleare (INFN), Italy.
- European Organization for Nuclear research (CERN), Switzerland.
- Universitat de Barcelona (UB), Spain.
- Barcelona Supercomputing Center (BSC), Spain.
- Qilimanjaro Quantum Tech, Spain.
- Centre for Quantum Technologies (CQT), Singapore.
- Institute of High Performance Computing (IHPC), Singapore.
- National Supercomputing Centre (NSCC), Singapore.
- RIKEN Center for Computational Science (R-CCS), Japan.
- NVIDIA (cuQuantum), USA.
