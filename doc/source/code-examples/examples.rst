Basic examples
==============

Here are a few short basic `how to` examples.

How to write and execute a circuit?
-----------------------------------

Here is an example of a circuit with 2 qubits:

.. testcode::

    import numpy as np
    from qibo import Circuit, gates

    # Construct the circuit
    circuit = Circuit(2)
    # Add some gates
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    # Define an initial state (optional - default initial state is |00>)
    initial_state = np.ones(4) / 2.0
    # Execute the circuit and obtain the final state
    result = circuit(initial_state) # circuit.execute(initial_state) also works
    print(result.state())
    # should print `tf.Tensor([1, 0, 0, 0])`
    print(result.state())
    # should print `np.array([1, 0, 0, 0])`

.. testoutput::
    :hide:

    ...

If you are planning to freeze the circuit and just query for different initial
states then you can use the ``Circuit.compile()`` method which will improve
evaluation performance, e.g.:

.. code-block::  python

    import numpy as np
    # switch backend to "tensorflow" through the Qiboml provider
    import qibo
    qibo.set_backend(backend="qiboml", platform="tensorflow")
    from qibo import Circuit, gates

    circuit = Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.X(1))
    circuit.add(gates.CU1(0, 1, 0.1234))
    circuit.compile()

    for i in range(100):
        init_state = np.ones(4) / 2.0 + i
        circuit(init_state)

Note that compiling is only supported when the ``tensorflow`` backend is
used. This backend is much slower than ``qibojit`` which uses custom operators
to apply gates.


How to print a circuit summary?
-------------------------------

It is possible to print a summary of the circuit using ``circuit.summary()``.
This will print basic information about the circuit, including its depth, the
total number of qubits and all gates in order of the number of times they appear.
The QASM name is used as identifier of gates.
For example

.. testcode::

    from qibo import Circuit, gates

    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    circuit.add(gates.CNOT(0, 2))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.H(2))
    circuit.add(gates.TOFFOLI(0, 1, 2))
    print(circuit.summary())
    # Prints
    '''
    Circuit depth = 5
    Total number of gates = 6
    Number of qubits = 3
    Most common gates:
    h: 3
    cx: 2
    ccx: 1
    '''
.. testoutput::
    :hide:

    Circuit depth = 5
    Total number of gates = 6
    Number of qubits = 3
    Most common gates:
    h: 3
    cx: 2
    ccx: 1


The circuit property ``circuit.gate_types`` (or ``circuit.gate_names``) will return a ``collections.Counter``
that contains the gate types (or names) and the corresponding numbers of appearance. The
method ``circuit.gates_of_type()`` can be used to access gate objects of specific type or name.
For example for the circuit of the previous example:

.. testsetup::

    from qibo import Circuit, gates

    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    circuit.add(gates.CNOT(0, 2))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.H(2))
    circuit.add(gates.TOFFOLI(0, 1, 2))

.. testcode::

    common_gates = circuit.gate_names.most_common()
    # returns the list [("h", 3), ("cx", 2), ("ccx", 1)]

    most_common_gate = common_gates[0][0]
    # returns "h"

    all_h_gates = circuit.gates_of_type(gates.H)
    # returns the list [(0, ref to H(0)), (1, ref to H(1)), (4, ref to H(2))]

A circuit may contain multi-controlled or other gates that are not supported by
OpenQASM. The ``circuit.decompose(*free)`` method decomposes such gates to
others that are supported by OpenQASM. For this decomposition to work the user
has to specify which qubits can be used as free/work. For more information on
this decomposition we refer to the related publication on
`arXiv:9503016 <https://arxiv.org/abs/quant-ph/9503016>`_. Currently only the
decomposition of multi-controlled ``X`` gates is implemented.


.. _measurement-examples:

How to perform measurements?
----------------------------

In order to obtain measurement results from a circuit one has to add measurement
gates (:class:`qibo.abstractions.gates.M`) and provide a number of shots (``nshots``)
when executing the circuit. In this case the returned
:class:`qibo.abstractions.states.AbstractState` will contain all the
information about the measured samples. For example

.. testsetup::
    import qibo
    qibo.set_backend("numpy")

.. testcode::

    from qibo import Circuit, gates

    circuit = Circuit(2)
    circuit.add(gates.X(0))
    # Add a measurement register on both qubits
    circuit.add(gates.M(0, 1))
    # Execute the circuit with the default initial state |00>.
    result = circuit(nshots=100)

Measurements are now accessible using the ``samples`` and ``frequencies`` methods
on the ``result`` object. In particular

* ``result.samples(binary=True)`` will return the array ``[[1, 0], [1, 0], ..., [1, 0]]`` with shape ``(100, 2)``,
* ``result.samples(binary=False)`` will return the array ``[2, 2, ..., 2]``,
* ``result.frequencies(binary=True)`` will return ``collections.Counter({"10": 100})``,
* ``result.frequencies(binary=False)`` will return ``collections.Counter({2: 100})``.

In addition to the functionality described above, it is possible to collect
measurement results grouped according to registers. The registers are defined
during the addition of measurement gates in the circuit. For example

.. testsetup::
    import qibo
    qibo.set_backend("numpy")

.. testcode::

    from qibo import Circuit, gates

    circuit = Circuit(5)
    circuit.add(gates.X(0))
    circuit.add(gates.X(4))
    circuit.add(gates.M(0, 1, register_name="A"))
    circuit.add(gates.M(3, 4, register_name="B"))
    result = circuit(nshots=100)

creates a circuit with five qubits that has two registers: ``A`` consisting of
qubits ``0`` and ``1`` and ``B`` consisting of qubits ``3`` and ``4``. Here
qubit ``2`` remains unmeasured. Measured results can now be accessed as

* ``result.samples(binary=False, registers=True)`` will return a dictionary with the measured sample tensors for each register: ``{"A": [2, 2, ...], "B": [1, 1, ...]}``,
* ``result.frequencies(binary=True, registers=True)`` will return a dictionary with the frequencies for each register: ``{"A": collections.Counter({"10": 100}), "B": collections.Counter({"01": 100})}``.

Setting ``registers=False`` (default option) will ignore the registers and return the
results similarly to the previous example. For example ``result.frequencies(binary=True)``
will return ``collections.Counter({"1001": 100})``.

It is possible to define registers of multiple qubits by either passing
the qubit ids seperately, such as ``gates.M(0, 1, 2, 4)``, or using the ``*``
operator: ``gates.M(*[0, 1, 2, 4])``. The ``*`` operator is useful if qubit
ids are saved in an iterable. For example ``gates.M(*range(5))`` is equivalent
to ``gates.M(0, 1, 2, 3, 4)``.

Unmeasured qubits are ignored by the measurement objects. Also, the
order that qubits appear in the results is defined by the order the user added
the measurements and not the qubit ids.

The final state vector is still accessible via :meth:`qibo.measurements.CircuitResult.state`.
Note that the state vector accessed this way corresponds to the state as if no
measurements occurred, that is the state is not collapsed during the measurement.
This is because measurement gates are only used to sample bitstrings and do not
have  any effect on the state vector. There are two reasons for this choice.
First, when more than one measurement shots are used the final collapsed state
is not uniquely defined as it would be different for each measurement result.
Second the user may wish to re-sample the final state vector in order to
obtain more measurement shots without having to re-execute the full simulation.
For applications that require the state vector to be collapsed during measurements
we refer to the :ref:`How to collapse state during measurements? <collapse-examples>`

The measured shots are obtained using pseudo-random number generators of the
underlying backend. If the user has installed a custom
backend (eg. qibojit) and asks for frequencies with more than 100000 shots,
a custom Metropolis algorithm will be used to obtain the corresponding samples,
for increase performance. The user can change the threshold for which this
algorithm is used using the ``qibo.set_metropolis_threshold()`` method,
for example:

.. testcode::

    import qibo

    print(qibo.get_metropolis_threshold()) # prints 100000
    qibo.set_metropolis_threshold(int(1e8))
    print(qibo.get_metropolis_threshold()) # prints 10^8
.. testoutput::
    :hide:

    100000
    100000000


If the Metropolis algorithm is not used and the user asks for frequencies with
a high number of shots then the corresponding samples are generated in batches.
The batch size can be controlled using the ``qibo.get_batch_size()`` and
``qibo.set_batch_size()`` functions similarly to the above example.
The default batch size is 2^18.


How to write a Quantum Fourier Transform?
-----------------------------------------

A simple Quantum Fourier Transform (QFT) example to test your installation:

.. testcode::

    from qibo.models import QFT

    # Create a QFT circuit with 15 qubits
    circuit = QFT(15)

    # Simulate final state wavefunction default initial state is |00>
    final_state = circuit()


Please note that the ``QFT()`` function is simply a shorthand for the circuit
construction. For number of qubits higher than 30, the QFT can be distributed to
multiple GPUs using ``QFT(31, accelerators)``. Further details are presented in
the section :ref:`How to select hardware devices? <gpu-examples>`.


.. _dtype-example:

How to modify the simulation data type?
---------------------------------------

By default the simulation is performed in ``complex128`` precision.
We provide the ``qibo.set_dtype`` function to modify the default behaviour, *i.e.* the data type
of all arrays, matrices, and / or tensors involved in the computation(s).
Note that `qibo.set_dtype` must be called before allocating circuits:

.. testcode::

        from qibo import set_backend, set_dtype

        set_backend("numpy")  # enables the numpy backend
        set_dtype("complex64") # enables complex64

        # alternatively, it is possible to set backend and data type in the same line.
        # The following line re-enables the numpy backend but now with complex128 data type.
        set_backend("numpy", dtype="complex128")

        # ... continue with circuit creation and execution


.. _visualize-example:

How to visualize a circuit?
---------------------------

It is possible to print a schematic diagram of the circuit using ``circuit.draw()``.
This will print an unicode text based representation of the circuit, including gates,
and qubits lines.
For example

.. testcode::

    from qibo.models import QFT

    circuit = QFT(5)
    circuit.draw()
    # Prints
    '''
    0: ─H─U1─U1─U1─U1───────────────────────────x───
    1: ───o──|──|──|──H─U1─U1─U1────────────────|─x─
    2: ──────o──|──|────o──|──|──H─U1─U1────────|─|─
    3: ─────────o──|───────o──|────o──|──H─U1───|─x─
    4: ────────────o──────────o───────o────o──H─x───
    '''
.. testoutput::
    :hide:

    0: ─H─U1─U1─U1─U1───────────────────────────x───
    1: ───o──|──|──|──H─U1─U1─U1────────────────|─x─
    2: ──────o──|──|────o──|──|──H─U1─U1────────|─|─
    3: ─────────o──|───────o──|────o──|──H─U1───|─x─
    4: ────────────o──────────o───────o────o──H─x───

How to visualize a circuit with style?
--------------------------------------

Qibo is able to draw a circuit using ``matplotlib`` library by calling the function ``plot_circuit``. It also have built-in styles ready to use
and also it is possible to apply custom styles to the circuit. The function is able to cluster the gates to reduce the circuit depth.
The built-in styles are:  ``garnacha``, ``fardelejo``, ``quantumspain``, ``color-blind``, ``cachirulo`` or custom dictionary.

For example, we can draw the QFT circuit for 5-qubits:

.. testcode::

        import matplotlib.pyplot as plt
        import qibo
        from qibo import gates, models
        from qibo.models import QFT

        # new plot function based on matplotlib
        from qibo.ui import plot_circuit

        # create a 5-qubits QFT circuit
        circuit = QFT(5)
        circuit.add(gates.M(qubit) for qubit in range(2))

        # print circuit with default options (default black & white style, scale factor of 0.6 and clustered gates)
        plot_circuit(circuit);

        # print the circuit with built-int style "garnacha", clustering gates and a custom scale factor
        # built-in styles: "garnacha", "fardelejo", "quantumspain", "color-blind", "cachirulo" or custom dictionary
        plot_circuit(circuit, scale = 0.8, cluster_gates = True, style="garnacha");

        # plot the Qibo circuit with a custom style
        custom_style = {
            "facecolor" : "#6497bf",
            "edgecolor" : "#01016f",
            "linecolor" : "#01016f",
            "textcolor" : "#01016f",
            "fillcolor" : "#ffb9b9",
            "gatecolor" : "#d8031c",
            "controlcolor" : "#360000"
        }

        plot_circuit(circuit, scale = 0.8, cluster_gates = True, style=custom_style);

How to visualize the density matrix?
------------------------------------

Qibo provides a function ``plot_density_hist`` to visualize the real and imaginary parts of a quantum state's density matrix
as 3D bar plots using matplotlib. This is useful for inspecting the structure of quantum states, especially for small systems.
The function return, the figure, and axes for the real and the imaginary parts.

The function supports options to control the number of tick labels shown (for large systems), transparency, colors, and figure size.

.. testcode::

    import numpy as np
    from qibo import Circuit, gates
    from qibo.ui import plot_density_hist

    # Create the circuit
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.RY(0, theta=np.pi/3))
    circuit.add(gates.RX(1, theta=np.pi/5))

    # Plot the density matrix as a cityscape (real and imaginary parts)
    plot_density_hist(circuit);
