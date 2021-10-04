Getting started
===============

In this section we present the basic aspects of Qibo design and provide
installation instructions.

.. toctree::
    :maxdepth: 1

    overview
    installation


Quick start
-----------

However, if you are in a hurry, just open a terminal with python > 3.6 and type:

.. code-block:: bash

      pip install qibo

This will install the basic primitives to start coding quantum applications.


Here a simple of Quantum Fourier Transform (QFT) to test your installation:

.. code-block:: python

    from qibo.models import QFT

    # Create a QFT circuit with 15 qubits
    circuit = QFT(15)

    # Simulate final state wavefunction default initial state is |00>
    final_state = circuit()

Here another example with more gates and shots simulation:

.. code-block:: python

    import numpy as np
    from qibo.models import Circuit
    from qibo import gates

    c = Circuit(2)
    c.add(gates.X(0))

    # Add a measurement register on both qubits
    c.add(gates.M(0, 1))

    # Execute the circuit with the default initial state |00>.
    result = c(nshots=100)


Why use Qibo?
-------------

Qibo is an open-source full stack API for quantum simulation and quantum hardware control.

Qibo aims to contribute as a community driven quantum middleware software with

1. *Simplicity:* agnostic design to quantum primitives.
2. *Flexibility:* transparent mechanism to execute code on classical and quantum hardware.
3. *Community:* a common place where find solutions to accelerate quantum development.
4. *Documentation:* describe all steps required to support new quantum devices or simulators.
5. *Applications:* maintain a large ecosystem of applications, quantum models and algorithms.

Qibo key features:

* Definition of a standard language for the construction and execution of quantum circuits with device agnostic approach to simulation and quantum hardware control based on plug and play backend drivers.
* A continuously growing code-base of quantum algorithms applications presented with examples and tutorials.
* Efficient simulation backends with GPU, multi-GPU and CPU with multi-threading support.
* Simple mechanism for the implementation of new simulation and hardware backend drivers.
