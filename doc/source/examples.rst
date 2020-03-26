Examples
========

Here short how to examples.

How to write a circuit?
-----------------------

Here an example with 2 qubits:

.. code-block::  python

    import numpy as np
    from qibo.models import Circuit
    from qibo import gates

    init_state = np.ones(4) / 2.0
    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.CRZ(0, 1, 0.1234))
    r1 = c.execute(init_state).numpy()

If you are planning to freeze the circuit and just query for different initial states then you can use the Circuit.compile method which will improve the evaluation performance, e.g.:

.. code-block:: python

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

How to write a VQE?
-------------------

The VQE requires an ansatz function and a ``Hamiltonian`` object. There are examples of VQE optimization in ``src/qibo/benchmarks``:

    - ``vqe_benchmark.py``: a simple example with the XXZ model.
    - ``adaptive_vqe_benchmark.py``: an adaptive example with the XXZ model.

Here a simple example using the Heisenberg XXZ model:

.. code-block:: python

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
    best, params = v.minimize(initial_parameters, method='BFGS')