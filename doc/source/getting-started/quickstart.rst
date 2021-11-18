Quick start
-----------

To quickly install Qibo and a high performance simulator for CPU, open a
terminal with python >= 3.7 and type:

.. code-block:: bash

      pip install qibo qibojit

This will install the basic primitives to start coding quantum applications.

Instead, if you use `conda <https://anaconda.org/>`_ type:

.. code-block:: bash

      conda install -c conda-forge qibo qibojit

.. warning::
    The ``qibo`` package alone includes a lightweight ``numpy`` simulator for
    single-thread CPU. Please visit the `backends <backend-drivers>`_
    documentation for more details about simulation backends.

Here an example of Quantum Fourier Transform (QFT) to test your installation:

.. testcode::

    from qibo.models import QFT

    # Create a QFT circuit with 15 qubits
    circuit = QFT(15)

    # Simulate final state wavefunction default initial state is |00>
    final_state = circuit()

Here an example of adding gates and measurements:

.. testcode::

    import numpy as np
    from qibo.models import Circuit
    from qibo import gates

    c = Circuit(2)
    c.add(gates.X(0))

    # Add a measurement register on both qubits
    c.add(gates.M(0, 1))

    # Execute the circuit with the default initial state |00>.
    result = c(nshots=100)
