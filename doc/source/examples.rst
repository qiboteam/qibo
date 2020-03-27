Examples
========

Here are a few short `how to` examples.

How to write and execute a circuit?
-----------------------------------

Here is an example of a circuit with 2 qubits:

.. code-block::  python

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
    print(final_state.numpy())
    # should print `np.array([1, 0, 0, 0])`

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

.. _measurement-examples:
How to perform measurements?
----------------------------

In order to obtain measurement results from a circuit one has to add measurement
gates (:class:`qibo.base.gates.M`) and provide a number of shots (``nshots``)
when executing the circuit. This will return a :class:`qibo.base.measurements.CircuitResult`
object which contains all the information about the measured samples.
For example

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

Measurements are now accessible using the ``samples`` and ``frequencies`` methods
on the ``result`` object. In particular

* ``result.samples(binary=True)`` will return the array ``tf.Tensor([[1, 0], [1, 0], ..., [1, 0]])`` with shape ``(100, 2)``,
* ``result.samples(binary=False)`` will return the array ``tf.Tensor([2, 2, ..., 2])``,
* ``result.frequencies(binary=True)`` will return ``collections.Counter({"10": 100})``,
* ``result.frequencies(binary=False)`` will return ``collections.Counter({2: 100})``.

In addition to the functionality described above, it is possible to collect
measurement results grouped according to registers. The registers are defined
during the addition of measurement gates in the circuit. For example

.. code-block:: python

    import numpy as np
    from qibo.models import Circuit
    from qibo import gates

    c = Circuit(5)
    c.add(gates.X(0))
    c.add(gates.X(4))
    c.add(gates.M(0, 1, register_name="A"))
    c.add(gates.M(3, 4, register_name="B"))
    result = c(nshots=100)

creates a circuit with five qubits that has two registers: ``A`` consisting of
qubits ``0`` and ``1`` and ``B`` consisting of qubits ``3`` and ``4``. Here
qubit ``2`` remains unmeasured. Measured results can now be accessed as

* ``result.samples(binary=False, registers=True)`` will return a dictionary with the measured sample tensors for each register: ``{"A": tf.Tensor([2, 2, ...]), "B": tf.Tensor([1, 1, ...])}``,
* ``result.frequencies(binary=True, registers=True)`` will return a dictionary with the frequencies for each register: ``{"A": collections.Counter({"10": 100}), "B": collections.Counter({"01": 100})}``.

Setting ``registers=False`` (default option) will ignore the registers and return the
results similarly to the previous example. For example ``result.frequencies(binary=True)``
will return ``collections.Counter({"1001": 100})``. Note that unmeasured qubits
are ignored by the measurement objects.


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
