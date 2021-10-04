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
