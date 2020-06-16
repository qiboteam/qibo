![logo](doc/source/qibo_logo.svg)

![Tests](https://github.com/Quantum-TII/qibo/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/Quantum-TII/qibo/branch/master/graph/badge.svg?token=1EKZKVEVX0)](https://codecov.io/gh/Quantum-TII/qibo)

QIBO is an open-source high-level API, written in Python and capable of performing classical simulation of quantum algorithms.

Some of the key features of QIBO are:
- A standard interface for the implementation and extension of quantum algorithms.
- Modular implementation on single (multi-threading) CPU and GPU.
- Good performance on GPU, with emphasis on double precision, through a custom classical simulation back-end based on [TensorFlow](https://tensorflow.org/).

## Documentation

Visit the [Link to the private documentation server](http://34.240.99.72/) and use the following credentials:
```
username: qiboteam
password: qilimanjaro
```

## Installation

In order to install you can simply clone this repository with
```bash
git clone git@github.com:Quantum-TII/qibo.git
```

and then proceed with the installation with:
```
pip install .
```
if you prefer to keep changes always synchronized with the code then install using the `develop` option:
```bash
python setup.py build
python setup.py develop # or pip install -e .
```

## Examples

There are code examples in the `src/qibo/benchmarks` and `src/qibo/tests` folders.
Full detailed examples with respective explanation are available directly in the documentation.
