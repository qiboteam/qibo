Advanced examples
=================

Here are a few short advanced `how to` examples.

.. _gpu-examples:

How to select hardware devices?
-------------------------------

Qibo supports execution on different hardware configurations including CPU with
multi-threading, single GPU and multiple GPUs. Here we provide some useful
information on how to control the devices that Qibo uses for circuit execution
in order to maximize performance for the available hardware configuration.

Switching between CPU and GPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If a GPU with CUDA support is available in the system and tensorflow or qibojit (cupy)
are installed then circuits will be executed on the GPU automatically unless the user
specifies otherwise. One can change the default simulation device using ``qibo.set_device``:

.. code-block::  python

    import qibo
    qibo.set_device("/CPU:0")
    final_state = c() # circuit will now be executed on CPU

The syntax of device names follows the pattern ``'/{device type}:{device number}'``
where device type can be CPU or GPU and the device number is an integer that
distinguishes multiple devices of the same type starting from 0. For more details
we refer to `Tensorflow's tutorial <https://www.tensorflow.org/guide/gpu#manual_device_placement>`_
on manual device placement.
Alternatively, running the command ``CUDA_VISIBLE_DEVICES=""`` in a terminal
hides CUDA GPUs from this terminal session.

In most cases the GPU accelerates execution compared to CPU, however the
following limitations should be noted:

* For small circuits (less than 10 qubits) the overhead from casting tensors on
  GPU may be larger than executing the circuit on CPU, making CPU execution
  preferable. In such cases disabling CPU multi-threading may also increase
  performance (see next subsection).
* A standard GPU has 12-16GB of memory and thus can simulate up to 30 qubits on
  single-precision or 29 qubits with double-precision when Qibo's default gates
  are used. For larger circuits one should either use the CPU (assuming it has
  more memory) or a distributed circuit configuration. The latter allows splitting
  the state vector on multiple devices and is useful both when multiple GPUs are
  available in the system or even for re-using a single GPU (see relevant
  subsection bellow).

Note that if the used device runs out of memory during a circuit execution an error will be
raised prompting the user to switch the default device using ``qibo.set_device``.

Setting the number of CPU threads
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Qibo by default uses the ``qibojit`` backend which is based on
custom operators. This backend uses OpenMP instructions for parallelization.
In most cases, utilizing all available CPU threads provides better performance.
However, for small circuits the parallelization overhead may decrease
performance making single thread execution preferrable.

You can restrict the number of threads by exporting the ``OMP_NUM_THREADS``
environment variable with the requested number of threads before launching Qibo,
or programmatically, during runtime, as follows:

.. testsetup::

    import qibo
    qibo.set_backend("qibojit")

.. testcode::

    import qibo
    # set the number of threads to 1
    qibo.set_threads(1)
    # retrieve the current number of threads
    current_threads = qibo.get_threads()

For similar wariness when using a machine learning backend (such as TensorFlow or Pytorch)
please refer to the Qiboml documentation.

Using multiple GPUs
^^^^^^^^^^^^^^^^^^^

Qibo supports distributed circuit execution on multiple GPUs. This feature can
be used as follows:

.. code-block:: python

    from qibo import Circuit, gates

    # Define GPU configuration
    accelerators = {"/GPU:0": 3, "/GPU:1": 1}
    # this will use the first GPU three times and the second one time
    # leading to four total logical devices
    # construct the distributed circuit for 32 qubits
    c = Circuit(32, accelerators)

Gates can then be added normally using ``c.add`` and the circuit can be executed
using ``c()``. Note that a ``memory_device`` is passed in the distributed circuit
(if this is not passed the CPU will be used by default). This device does not perform
any gate calculations but is used to store the full state. Therefore the
distributed simulation is limited by the amount of CPU memory.

Also, note that it is possible to reuse a single GPU multiple times increasing the number of
"logical" devices in the distributed calculation. This allows users to execute
circuits with more than 30 qubits on a single GPU by reusing several times using
``accelerators = {"/GPU:0": ndevices}``. Such a simulation will be limited
by CPU memory only.

For systems without GPUs, the distributed implementation can be used with any
type of device. For example if multiple CPUs, the user can pass these CPUs in the
accelerator dictionary.

Distributed circuits are generally slower than using a single GPU due to communication
bottleneck. However for more than 30 qubits (which do not fit in single GPU) and
specific applications (such as the QFT) the multi-GPU scheme can be faster than
using only CPU.

Note that simulating a circuit using multiple GPUs partitions the state in
multiple pieces which are distributed to the different devices.
Creating the full state as a single tensor would require merging
these pieces and using twice as much memory. This is disabled by default,
however the user may create the full state as follows:

.. code-block::  python

    # Create distributed circuits for two GPUs
    c = Circuit(32, {"/GPU:0": 1, "/GPU:1": 1})
    # Add gates
    c.add(...)
    # Execute (``result`` will be a ``DistributedState``)
    result = c()

    # ``DistributedState`` supports indexing and slicing
    print(result[40])
    # will print the 40th component of the final state vector
    print(result[20:25])
    # will print the components from 20 to 24 (inclusive)

    # Access the full state (will double memory usage)
    final_state = result.state()
    # ``final_state`` is a ``tf.Tensor``


How to use callbacks?
---------------------

Callbacks allow the user to apply additional functions on the state vector
during circuit execution. An example use case of this is the calculation of
entanglement entropy as the state propagates through a circuit. This can be
implemented easily using :class:`qibo.callbacks.EntanglementEntropy`
and the :class:`qibo.gates.CallbackGate` gate. For example:

.. testcode::

    from qibo import models, gates, callbacks

    # create entropy callback where qubit 0 is the first subsystem
    entropy = callbacks.EntanglementEntropy([0])

    # initialize circuit with 2 qubits and add gates
    c = models.Circuit(2) # state is |00> (entropy = 0)
    c.add(gates.CallbackGate(entropy)) # performs entropy calculation in the initial state
    c.add(gates.H(0)) # state is |+0> (entropy = 0)
    c.add(gates.CallbackGate(entropy)) # performs entropy calculation after H
    c.add(gates.CNOT(0, 1)) # state is |00> + |11> (entropy = 1))
    c.add(gates.CallbackGate(entropy)) # performs entropy calculation after CNOT

    # execute the circuit using the callback
    final_state = c()

The results can be accessed using indexing on the callback objects. In this
example ``entropy[:]`` will return ``[0, 0, 1]`` which are the
values of entropy after every gate in the circuit.

The same callback object can be used in a second execution of this or a different
circuit. For example

.. testsetup::

    from qibo import models, gates, callbacks

    # create entropy callback where qubit 0 is the first subsystem
    entropy = callbacks.EntanglementEntropy([0])

    # initialize circuit with 2 qubits and add gates
    c = models.Circuit(2) # state is |00> (entropy = 0)
    c.add(gates.CallbackGate(entropy)) # performs entropy calculation in the initial state
    c.add(gates.H(0)) # state is |+0> (entropy = 0)
    c.add(gates.CallbackGate(entropy)) # performs entropy calculation after H
    c.add(gates.CNOT(0, 1)) # state is |00> + |11> (entropy = 1))
    c.add(gates.CallbackGate(entropy)) # performs entropy calculation after CNOT

    # execute the circuit using the callback
    final_state = c()

.. testcode::

    # c is the same circuit as above
    # execute the circuit
    final_state = c()
    # execute the circuit a second time
    final_state = c()

    # print result
    print(entropy[:]) # [0, 0, 1, 0, 0, 1]
.. testoutput::
    :hide:

    ...

The callback for entanglement entropy can also be used on state vectors directly.
For example


.. _params-examples:

How to use parametrized gates?
------------------------------

Some Qibo gates such as rotations accept values for their free parameter. Once
such gates are added in a circuit their parameters can be updated using the
:meth:`qibo.models.circuit.Circuit.set_parameters` method. For example:

.. testcode::

    from qibo import Circuit, gates
    # create a circuit with all parameters set to 0.
    c = Circuit(3)
    c.add(gates.RX(0, theta=0))
    c.add(gates.RY(1, theta=0))
    c.add(gates.CZ(1, 2))
    c.add(gates.fSim(0, 2, theta=0, phi=0))
    c.add(gates.H(2))

    # set new values to the circuit's parameters
    params = [0.123, 0.456, (0.789, 0.321)]
    c.set_parameters(params)

initializes a circuit with all gate parameters set to 0 and then updates the
values of these parameters according to the ``params`` list. Alternatively the
user can use ``circuit.set_parameters()`` with a dictionary or a flat list.
The keys of the dictionary should be references to the gate objects of
the circuit. For example:

.. testsetup::

    from qibo import Circuit, gates

.. testcode::

    c = Circuit(3)
    g0 = gates.RX(0, theta=0)
    g1 = gates.RY(1, theta=0)
    g2 = gates.fSim(0, 2, theta=0, phi=0)
    c.add([g0, g1, gates.CZ(1, 2), g2, gates.H(2)])

    # set new values to the circuit's parameters using a dictionary
    params = {g0: 0.123, g1: 0.456, g2: (0.789, 0.321)}
    c.set_parameters(params)
    # equivalently the parameter's can be update with a list as
    params = [0.123, 0.456, (0.789, 0.321)]
    c.set_parameters(params)
    # or with a flat list as
    params = [0.123, 0.456, 0.789, 0.321]
    c.set_parameters(params)

If a list is given then its length and elements should be compatible with the
parametrized gates contained in the circuit. If a dictionary is given then its
keys should be all the parametrized gates in the circuit.

The following gates support parameter setting:

* ``RX``, ``RY``, ``RZ``, ``U1``, ``CU1``: Accept a single ``theta`` parameter.
* :class:`qibo.gates.fSim`: Accepts a tuple of two parameters ``(theta, phi)``.
* :class:`qibo.gates.GeneralizedfSim`: Accepts a tuple of two parameters
  ``(unitary, phi)``. Here ``unitary`` should be a unitary matrix given as an
  array or ``tf.Tensor`` of shape ``(2, 2)``. A ``torch.Tensor`` is required when using the pytorch backend.
* :class:`qibo.gates.Unitary`: Accepts a single ``unitary`` parameter. This
  should be an array or ``tf.Tensor`` of shape ``(2, 2)``. A ``torch.Tensor`` is required when using the pytorch backend.

Note that a ``np.ndarray`` or a ``tf.Tensor`` may also be used in the place of
a flat list (``torch.Tensor`` is required when using the pytorch backend).
Using :meth:`qibo.models.circuit.Circuit.set_parameters` is more
efficient than recreating a new circuit with new parameter values. The inverse
method :meth:`qibo.models.circuit.Circuit.get_parameters` is also available
and returns a list, dictionary or flat list with the current parameter values
of all parametrized gates in the circuit.

It is possible to hide a parametrized gate from the action of
:meth:`qibo.models.circuit.Circuit.get_parameters` and
:meth:`qibo.models.circuit.Circuit.set_parameters` by setting
the ``trainable=False`` during gate creation. For example:

.. testsetup::

    from qibo import Circuit, gates

.. testcode::

    c = Circuit(3)
    c.add(gates.RX(0, theta=0.123))
    c.add(gates.RY(1, theta=0.456, trainable=False))
    c.add(gates.fSim(0, 2, theta=0.789, phi=0.567))

    print(c.get_parameters())
    # prints [(0.123,), (0.789, 0.567)] ignoring the parameters of the RY gate

.. testoutput::

    [(0.123,), (0.789, 0.567)]


This is useful when the user wants to freeze the parameters of specific
gates during optimization.
Note that ``trainable`` defaults to ``True`` for all parametrized gates.


.. _collapse-examples:

How to collapse state during measurements?
------------------------------------------

As mentioned in the :ref:`How to perform measurements? <measurement-examples>`
measurement can by default be used only in the end of the circuit and they do
not have any effect on the state. In this section we describe how to collapse
the state during measurements and re-use measured qubits in the circuit.
Collapsing the state means projecting to the ``|0>`` or ``|1>`` subspace according to
the sampled result for each measured qubit.

The state is collapsed when the ``collapse=True`` is used during instantiation
of the :class:`qibo.gates.M` gate. For example

.. testcode::

    from qibo import Circuit, gates

    c = Circuit(1, density_matrix=True)
    c.add(gates.H(0))
    output = c.add(gates.M(0, collapse=True))
    c.add(gates.H(0))
    result = c(nshots=1)
    print(result)
    # prints |+><+| if 0 is measured
    # or |-><-| if 1 is measured
.. testoutput::
    :hide:

    ...

In this example the single qubit is measured while in the state (``|0> + |1>``) and
is collapsed to either ``|0>`` or ``|1>``. The qubit can then be re-used by adding more
gates that act to this. The outcomes of ``collapse=True`` measurements is not
contained in the final result object but is accessible from the `output` object
returned when adding the gate to the circuit. ``output`` supports the
``output.samples()`` and ``output.frequencies()`` functionality as described
in :ref:`How to perform measurements? <measurement-examples>`.

Collapse gates are single-shot by default because the state collapse is not
well-defined for more than one shots. If the user passes the ``nshots`` arguments
during the circuit execution (eg. ``result = c(nshots=100)`` in the above
example), then the circuit execution will be repeated ``nshots`` times using
a loop:

.. testsetup::

    from qibo import Circuit, gates

    c = Circuit(1, density_matrix=True)
    c.add(gates.H(0))
    output = c.add(gates.M(0, collapse=True))
    c.add(gates.H(0))
    nshots = 100

.. testcode::

    for _ in range(nshots):
        result = c()

Note that this will be more time-consuming compared to multi-shot simulation
of standard (non-collapse) measurements where the circuit is simulated once and
the final state vector is sampled ``nshots`` times. For multi-shot simulation
the outcomes are still accessible using ``output.samples()`` and
``output.frequencies()``.

Using normal measurements and collapse measurements in the same circuit is
also possible:

.. testcode::

    from qibo import Circuit, gates

    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.H(1))
    output = c.add(gates.M(0, collapse=True))
    c.add(gates.H(0))
    c.add(gates.M(0, 1))
    result = c(nshots=100)

In this case ``output`` will contain the results of the first ``collapse=True``
measurement while ``result`` will contain the results of the standard measurement.

Conditioning gates on measurement outcomes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The output of ``collapse=True`` measurements can be used as a parameter in
any parametrized gate as follows:

.. testcode::

    import numpy as np
    from qibo import Circuit, gates

    c = Circuit(2, density_matrix=True)
    c.add(gates.H(0))
    output = c.add(gates.M(0, collapse=True))
    c.add(gates.RX(1, theta=np.pi * output.symbols[0] / 4))
    result = c()

In this case the first qubit will be measured and if 1 is found a pi/4 X-rotation
will be applied to the second qubit, otherwise no rotation. Qibo allows to
use ``output`` as a parameter during circuit creation through the use of
``sympy.Symbol`` objects. These symbols can be accessed through the ``output.symbols``
list and they acquire a numerical value during execution when the measurement
is performed. As explained above, if ``nshots > 1`` is given during circuit
execution the execution is repeated using a loop.

If more than one qubits are used in a ``collapse=True`` measurement gate the
``output.symbols`` list can be indexed accordingly:

.. testcode::

    import numpy as np
    from qibo import Circuit, gates

    c = Circuit(3, density_matrix=True)
    c.add(gates.H(0))
    output = c.add(gates.M(0, 1, collapse=True))
    c.add(gates.RX(1, theta=np.pi * output.symbols[0] / 4))
    c.add(gates.RY(2, theta=np.pi * (output.symbols[0] + output.symbols[1]) / 5))
    result = c()


How to invert a circuit?
------------------------

Many quantum algorithms require using a specific subroutine and its inverse
in the same circuit. Qibo simplifies this implementation via the
:meth:`qibo.models.circuit.Circuit.invert` method. This method produces
the inverse of a circuit by taking the dagger of all gates in reverse order. It
can be used with circuit addition to simplify the construction of algorithms,
for example:

.. testcode::

    from qibo import Circuit, gates

    # Create a subroutine
    subroutine = Circuit(6)
    subroutine.add([gates.RX(i, theta=0.1) for i in range(5)])
    subroutine.add([gates.CZ(i, i + 1) for i in range(0, 5, 2)])

    # Create the middle part of the circuit
    middle = Circuit(6)
    middle.add([gates.CU2(i, i + 1, phi=0.1, lam=0.2) for i in range(0, 5, 2)])

    # Create the total circuit as subroutine + middle + subroutine^{-1}
    circuit = subroutine + middle + subroutine.invert()


Note that circuit addition works only between circuits that act on the same number
of qubits. It is often useful to add subroutines only on a subset of qubits of the
large circuit. This is possible using the :meth:`qibo.models.circuit.Circuit.on_qubits`
method. For example:

.. testcode::

    from qibo import models, gates

    # Create a small circuit of 4 qubits
    smallc = models.Circuit(4)
    smallc.add((gates.RX(i, theta=0.1) for i in range(4)))
    smallc.add((gates.CNOT(0, 1), gates.CNOT(2, 3)))

    # Create a large circuit on 8 qubits
    largec = models.Circuit(8)
    # Add the small circuit on even qubits
    largec.add(smallc.on_qubits(*range(0, 8, 2)))
    # Add a QFT on odd qubits
    largec.add(models.QFT(4).on_qubits(*range(1, 8, 2)))
    # Add an inverse QFT on first 6 qubits
    largec.add(models.QFT(6).invert().on_qubits(*range(6)))


.. _vqe-example:

How to write a VQE?
-------------------

The VQE requires an ansatz function and a ``Hamiltonian`` object.
There are examples of VQE optimization in ``examples/benchmarks``:

    - ``vqe.py``: a simple example with the XXZ model.

Here is a simple example using the Heisenberg XXZ model Hamiltonian:

.. testcode::

    import numpy as np
    from qibo import models, gates, hamiltonians

    nqubits = 6
    nlayers  = 4

    # Create variational circuit
    circuit = models.Circuit(nqubits)
    for l in range(nlayers):
        circuit.add((gates.RY(q, theta=0) for q in range(nqubits)))
        circuit.add((gates.CZ(q, q+1) for q in range(0, nqubits-1, 2)))
        circuit.add((gates.RY(q, theta=0) for q in range(nqubits)))
        circuit.add((gates.CZ(q, q+1) for q in range(1, nqubits-2, 2)))
        circuit.add(gates.CZ(0, nqubits-1))
    circuit.add((gates.RY(q, theta=0) for q in range(nqubits)))

    # Create XXZ Hamiltonian
    hamiltonian = hamiltonians.XXZ(nqubits=nqubits)
    # Create VQE model
    vqe = models.VQE(circuit, hamiltonian)

    # Optimize starting from a random guess for the variational parameters
    initial_parameters = np.random.uniform(0, 2*np.pi,
                                            2*nqubits*nlayers + nqubits)
    best, params, extra = vqe.minimize(initial_parameters, method='BFGS', compile=False)



For more information on the available options of the ``vqe.minimize`` call we
refer to the :ref:`Optimizers <Optimizers>` section of the documentation.
Note that if the Stochastic Gradient Descent optimizer is used then the user
has to use a backend based on tensorflow or pytorch primitives and not the default custom
backend, as custom operators currently do not support automatic differentiation.
To switch the backend one can do ``qibo.set_backend("tensorflow")`` or ``qibo.set_backend("pytorch")``.
Check the :ref:`How to use automatic differentiation? <autodiff-example>`
section for more details.

When using a VQE with more than 12 qubits, it may be useful to fuse the circit implementing
the ansatz using :meth:`qibo.models.Circuit.fuse`.
This optimizes performance by fusing the layer of one-qubit parametrized gates with
the layer of two-qubit entangling gates and applying both as a single layer of
general two-qubit gates (as 4x4 matrices).

.. testsetup::

    import numpy as np
    from qibo import models, gates, hamiltonians

.. testcode::

    circuit = models.Circuit(nqubits)
    for l in range(nlayers):
        circuit.add((gates.RY(q, theta=0) for q in range(nqubits)))
        circuit.add((gates.CZ(q, q+1) for q in range(0, nqubits-1, 2)))
        circuit.add((gates.RY(q, theta=0) for q in range(nqubits)))
        circuit.add((gates.CZ(q, q+1) for q in range(1, nqubits-2, 2)))
        circuit.add(gates.CZ(0, nqubits-1))
    circuit.add((gates.RY(q, theta=0) for q in range(nqubits)))
    circuit = circuit.fuse()

.. _vqc-example:

How to write a custom variational circuit optimization?
-------------------------------------------------------

Similarly to the VQE, a custom implementation of a Variational Quantum Circuit
(VQC) model can be achieved by defining a custom loss function and calling the
:meth:`qibo.optimizers.optimize` method.

Here is a simple example using a custom loss function:

.. testcode::

    import numpy as np
    from qibo import models, gates
    from qibo.optimizers import optimize

    # custom loss function, computes fidelity
    def myloss(parameters, circuit, target):
        circuit.set_parameters(parameters)
        final_state = circuit().state()
        return 1 - np.abs(np.conj(target).dot(final_state))

    nqubits = 6
    nlayers  = 2

    # Create variational circuit
    c = models.Circuit(nqubits)
    for l in range(nlayers):
        c.add((gates.RY(q, theta=0) for q in range(nqubits)))
        c.add((gates.CZ(q, q+1) for q in range(0, nqubits-1, 2)))
        c.add((gates.RY(q, theta=0) for q in range(nqubits)))
        c.add((gates.CZ(q, q+1) for q in range(1, nqubits-2, 2)))
        c.add(gates.CZ(0, nqubits-1))
    c.add((gates.RY(q, theta=0) for q in range(nqubits)))

    # Optimize starting from a random guess for the variational parameters
    x0 = np.random.uniform(0, 2*np.pi, 2*nqubits*nlayers + nqubits)
    data = np.random.normal(0, 1, size=2**nqubits)

    # perform optimization
    best, params, extra = optimize(myloss, x0, args=(c, data), method='BFGS')

    # set final solution to circuit instance
    c.set_parameters(params)


.. _qaoa-example:

How to use the QAOA?
--------------------

The quantum approximate optimization algorithm (QAOA) was introduced in
`arXiv:1411.4028 <https://arxiv.org/abs/1411.4028>`_ and is a prominent
algorithm for solving hard optimization problems using the circuit-based model
of quantum computation. Qibo provides an implementation of the QAOA as a model
that can be defined using a :class:`qibo.hamiltonians.Hamiltonian`. When
properly optimized, the QAOA ansatz will approximate the ground state of this
Hamiltonian. Here is a simple example using the Heisenberg XXZ Hamiltonian:

.. testcode::

    import numpy as np
    from qibo import models, hamiltonians

    # Create XXZ Hamiltonian for six qubits
    hamiltonian = hamiltonians.XXZ(6)
    # Create QAOA model
    qaoa = models.QAOA(hamiltonian)

    # Optimize starting from a random guess for the variational parameters
    initial_parameters = 0.01 * np.random.uniform(0,1,4)
    best_energy, final_parameters, extra = qaoa.minimize(initial_parameters, method="BFGS")

In the above example the initial guess for parameters has length four and
therefore the QAOA ansatz consists of four operators, two using the
``hamiltonian`` and two using the mixer Hamiltonian. The user may specify the
mixer Hamiltonian when defining the QAOA model, otherwise
:class:`qibo.hamiltonians.X` will be used by default.
Note that the user may set the values of the variational parameters explicitly
using :meth:`qibo.models.QAOA.set_parameters`.
Similarly to the VQE, we refer to :ref:`Optimizers <Optimizers>` for more
information on the available options of the ``qaoa.minimize``.

QAOA uses the ``|++...+>`` as the default initial state on which the variational
operators are applied. The user may specify a different initial state when
executing or optimizing by passing the ``initial_state`` argument.

The QAOA model uses :ref:`Solvers <Solvers>` to apply the exponential operators
to the state vector. For more information on how solvers work we refer to the
:ref:`How to simulate time evolution? <timeevol-example>` section.
When a :class:`qibo.hamiltonians.Hamiltonian` is used then solvers will
exponentiate it using its full matrix. Alternatively, if a
:class:`qibo.hamiltonians.SymbolicHamiltonian` is used then solvers
will fall back to traditional Qibo circuits that perform Trotter steps. For
more information on how the Trotter decomposition is implemented in Qibo we
refer to the :ref:`Using Trotter decomposition <trotterdecomp-example>` example.

When Trotter decomposition is used, it is possible to execute the QAOA circuit
on multiple devices, by passing an ``accelerators`` dictionary when defining
the model. For example the previous example would have to be modified as:

.. code-block:: python

    from qibo import models, hamiltonians

    hamiltonian = hamiltonians.XXZ(6, dense=False)
    qaoa = models.QAOA(hamiltonian, accelerators={"/GPU:0": 1, "/GPU:1": 1})


.. _autodiff-example:

How to use automatic differentiation?
-------------------------------------

The parameters of variational circuits can be optimized using the frameworks of
Tensorflow or Pytorch.

As a deep learning framework, Tensorflow supports
`automatic differentiation <https://www.tensorflow.org/tutorials/customization/autodiff>`_.
The following script optimizes the parameters of two rotations so that the
circuit output matches a target state using the fidelity as the corresponding loss function.

Note that, as in the following example, the rotation angles have to assume real values
to ensure the rotational gates are representing unitary operators.

Qibo doesn't provide Tensorflow and Pytorch as native backends; Qiboml has to be
installed and used as provider of these quantum machine learning backends.

.. code-block:: python

    import qibo
    qibo.set_backend(backend="qiboml", platform="tensorflow")
    from qibo import gates, models

    backend = qibo.get_backend()
    tf = backend.tf

    # Optimization parameters
    nepochs = 1000
    optimizer = tf.keras.optimizers.Adam()
    target_state = tf.ones(4, dtype=tf.complex128) / 2.0

    # Define circuit ansatz
    params = tf.Variable(
        tf.random.uniform((2,), dtype=tf.float64)
    )
    c = models.Circuit(2)
    c.add(gates.RX(0, params[0]))
    c.add(gates.RY(1, params[1]))

    for _ in range(nepochs):
        with tf.GradientTape() as tape:
            c.set_parameters(params)
            final_state = c().state()
            fidelity = tf.math.abs(tf.reduce_sum(tf.math.conj(target_state) * final_state))
            loss = 1 - fidelity
        grads = tape.gradient(loss, params)
        optimizer.apply_gradients(zip([grads], [params]))


Note that the ``"tensorflow"`` backend has to be used here since it provides
automatic differentiation tools. To be constructed, the Qiboml package has to be
installed and used.

The optimization procedure may also be compiled, however in this case it is not
possible to use :meth:`qibo.circuit.Circuit.set_parameters` as the
circuit needs to be defined inside the compiled ``tf.GradientTape()``.
For example:

.. code-block:: python

    import qibo
    qibo.set_backend(backend="qiboml", platform="tensorflow")
    from qibo import gates, models

    backend = qibo.get_backend()
    tf = backend.tf

    nepochs = 1000
    optimizer = tf.keras.optimizers.Adam()
    target_state = tf.ones(4, dtype=tf.complex128) / 2.0
    params = tf.Variable(tf.random.uniform((2,), dtype=tf.float64))

    @tf.function
    def optimize(params):
        with tf.GradientTape() as tape:
            c = models.Circuit(2)
            c.add(gates.RX(0, theta=params[0]))
            c.add(gates.RY(1, theta=params[1]))
            final_state = c().state()
            fidelity = tf.math.abs(tf.reduce_sum(tf.math.conj(target_state) * final_state))
            loss = 1 - fidelity
        grads = tape.gradient(loss, params)
        optimizer.apply_gradients(zip([grads], [params]))

    for _ in range(nepochs):
        optimize(params)


The user may also use ``tf.Variable`` and parametrized gates in any other way
that is supported by Tensorflow, such as defining
`custom Keras layers <https://www.tensorflow.org/guide/keras/custom_layers_and_models>`_
and using the `Sequential model API <https://www.tensorflow.org/api_docs/python/tf/keras/Sequential>`_
to train them.

Similarly, Pytorch supports `automatic differentiation <https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorFor%20example%20tial.html>`_.
The following script optimizes the parameters of the variational circuit of the first example using the Pytorch framework.

.. code-block:: python

    import qibo
    qibo.set_backend("pytorch")
    import torch
    from qibo import gates, models

    # Optimization parameters
    nepochs = 1000
    optimizer = torch.optim.Adam
    target_state = torch.ones(4, dtype=torch.complex128) / 2.0

    # Define circuit ansatz
    params = torch.tensor(
        torch.rand(2, dtype=torch.float64), requires_grad=True
    )
    c = models.Circuit(2)
    c.add(gates.RX(0, params[0]))
    c.add(gates.RY(1, params[1]))

    optimizer = optimizer([params])

    for _ in range(nepochs):
        optimizer.zero_grad()
        c.set_parameters(params)
        final_state = c().state()
        fidelity = torch.abs(torch.sum(torch.conj(target_state) * final_state))
        loss = 1 - fidelity
        loss.backward()
        optimizer.step()


.. _noisy-example:

How to perform noisy simulation?
--------------------------------

Qibo can perform noisy simulation with two different methods: by repeating the
circuit execution multiple times and applying noise gates probabilistically
or by using density matrices and applying noise channels. The two methods
are analyzed in the following sections.

Moreover, Qibo provides functionality to add bit-flip errors to measurements
after the simulation is completed. This is analyzed in
:ref:`Measurement errors <measurementbitflips-example>`.



.. _densitymatrix-example:

Using density matrices
^^^^^^^^^^^^^^^^^^^^^^

Qibo circuits can evolve density matrices if they are initialized using the
``density_matrix=True`` flag, for example:

.. testcode::

    import qibo
    qibo.set_backend("qibojit")

    from qibo import models, gates

    # Define circuit
    c = models.Circuit(2, density_matrix=True)
    c.add(gates.H(0))
    c.add(gates.H(1))
    # execute using the default initial state |00><00|
    result = c() # will be |++><++|

will perform the transformation

.. math::
    |00 \rangle \langle 00| \rightarrow (H_1 \otimes H_2)|00 \rangle \langle 00|(H_1 \otimes H_2)^\dagger = |++ \rangle \langle ++|

Similarly to state vector circuit simulation, the user may specify a custom
initial density matrix by passing the corresponding array when executing the
circuit. If a state vector is passed as an initial state in a density matrix
circuit, it will be transformed automatically to the equivalent density matrix.

Additionally, Qibo provides several gates that represent channels which can
be used during a density matrix simulation. We refer to the
:ref:`Channels <Channels>` section of the documentation for a complete list of
the available channels. Noise can be simulated using these channels,
for example:

.. testcode::

    from qibo import models, gates

    c = models.Circuit(2, density_matrix=True) # starts with state |00><00|
    c.add(gates.X(1))
    # transforms |00><00| -> |01><01|
    c.add(gates.PauliNoiseChannel(0, [("X", 0.3)]))
    # transforms |01><01| -> (1 - px)|01><01| + px |11><11|
    result = c()
    # result.state() will be tf.Tensor(diag([0, 0.7, 0, 0.3]))

will perform the transformation

.. math::
    |00\rangle \langle 00|& \rightarrow (I \otimes X)|00\rangle \langle 00|(I \otimes X)
    = |01\rangle \langle 01|
    \\& \rightarrow 0.7|01\rangle \langle 01| + 0.3(X\otimes I)|01\rangle \langle 01|(X\otimes I)^\dagger
    \\& = 0.7|01\rangle \langle 01| + 0.3|11\rangle \langle 11|

Measurements and callbacks can be used with density matrices exactly as in the
case of state vector simulation.


.. _repeatedexec-example:

Using repeated execution
^^^^^^^^^^^^^^^^^^^^^^^^

Simulating noise with density matrices is memory intensive as it effectively
doubles the number of qubits. Qibo provides an alternative way of simulating
the effect of channels without using density matrices, which relies on state
vectors and repeated circuit execution with sampling. Noise can be simulated
by creating a normal (non-density matrix) circuit and repeating its execution
as follows:

.. testcode::

    import numpy as np
    from qibo import models, gates

    # Define circuit
    c = models.Circuit(5)
    thetas = np.random.random(5)
    c.add((gates.RX(i, theta=t) for i, t in enumerate(thetas)))
    # Add noise channels to all qubits
    c.add((gates.PauliNoiseChannel(i, [("X", 0.2), ("Y", 0.0), ("Z", 0.3)])
           for i in range(5)))
    # Add measurement of all qubits
    c.add(gates.M(*range(5)))

    # Repeat execution 1000 times
    result = c(nshots=1000)

In this example the simulation is repeated 1000 times and the action of the
:class:`qibo.gates.PauliNoiseChannel` gate differs each time, because
the error ``X``, ``Y`` and ``Z`` gates are sampled according to the given
probabilities. Note that when a channel is used, the command ``c(nshots=1000)``
has a different behavior than what is described in
:ref:`How to perform measurements? <measurement-examples>`.
Normally ``c(nshots=1000)`` would execute the circuit once and would then
sample 1000 bit-strings from the final state. When channels are used, the full
is executed 1000 times because the behavior of channels is probabilistic and
different in each execution. Note that now the simulation time required will
increase linearly with the number of repetitions (``nshots``).

Note that executing a circuit with channels only once is possible, however,
since the channel acts probabilistically, the results of a single execution
are random and usually not useful on their own.
It is possible also to use repeated execution with noise channels even without
the presence of measurements. If ``c(nshots=1000)`` is called for a circuit
that contains channels but no measurements measurements then the circuit will
be executed 1000 times and the final 1000 state vectors will be returned as
a tensor of shape ``(nshots, 2 ^ nqubits)``.
Note that this tensor is usually large and may lead to memory errors,
therefore this usage is not advised.

Unlike the density matrix approach, it is not possible to use every channel
with sampling and repeated execution. Specifically,
:class:`qibo.gates.UnitaryChannel` and
:class:`qibo.gates.PauliNoiseChannel` can be used with sampling, while
:class:`qibo.gates.KrausChannel` requires density matrices.


Adding noise after every gate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In practical applications noise typically occurs after every gate.
Qibo provides the :meth:`qibo.models.circuit.Circuit.with_pauli_noise` method
which automatically creates a new circuit that contains a
:class:`qibo.gates.PauliNoiseChannel` after every gate.
The user can control the probabilities of the noise channel using a noise map,
which is a dictionary that maps qubits to the corresponding probability
triplets. For example, the following script

.. testcode::

      from qibo import models, gates

      c = models.Circuit(2)
      c.add([gates.H(0), gates.H(1), gates.CNOT(0, 1)])

      # Define a noise map that maps qubit IDs to noise probabilities
      noise_map = {0: list(zip(["X", "Z"], [0.1, 0.2])), 1: list(zip(["Y", "Z"], [0.2, 0.1]))}
      noisy_c = c.with_pauli_noise(noise_map)

will create a new circuit ``noisy_c`` that is equivalent to:

.. testcode::

      noisy_c2 = models.Circuit(2)
      noisy_c2.add(gates.H(0))
      noisy_c2.add(gates.PauliNoiseChannel(0, [("X", 0.1), ("Y", 0.0), ("Z", 0.2)]))
      noisy_c2.add(gates.H(1))
      noisy_c2.add(gates.PauliNoiseChannel(1, [("X", 0.0), ("Y", 0.2), ("Z", 0.1)]))
      noisy_c2.add(gates.CNOT(0, 1))
      noisy_c2.add(gates.PauliNoiseChannel(0, [("X", 0.1), ("Y", 0.0), ("Z", 0.2)]))
      noisy_c2.add(gates.PauliNoiseChannel(1, [("X", 0.0), ("Y", 0.2), ("Z", 0.1)]))

Note that ``noisy_c`` uses the gate objects of the original circuit ``c``
(it is not a deep copy), while in ``noisy_c2`` each gate was created as
a new object.

The user may use a single tuple instead of a dictionary as the noise map
In this case the same probabilities will be applied to all qubits.
That is ``noise_map = list(zip(["X", "Z"], [0.1, 0.1]))`` is equivalent to
``noise_map = {0: list(zip(["X", "Z"], [0.1, 0.1])), 1: list(zip(["X", "Z"], [0.1, 0.1])), ...}``.

As described in the previous sections, if
:meth:`qibo.models.circuit.Circuit.with_pauli_noise` is used in a circuit
that uses state vectors then noise will be simulated with repeated execution.
If the user wishes to use density matrices instead, this is possible by
passing the ``density_matrix=True`` flag during the circuit initialization and call
``.with_pauli_noise`` on the new circuit.

.. _noisemodel-example:

Using a noise model
^^^^^^^^^^^^^^^^^^^

In a real quantum circuit some gates can be highly faulty and introduce errors.
In order to simulate this behavior Qibo provides the :class:`qibo.noise.NoiseModel`
class which can store errors that are gate-dependent using the
:meth:`qibo.noise.NoiseModel.add` method and generate the corresponding noisy circuit
with :meth:`qibo.noise.NoiseModel.apply`. The corresponding noise is applied after
every instance of the gate in the circuit. It is also possible to specify on which qubits
the noise will be added.

The current quantum errors available to build a custom noise model are:
:class:`qibo.noise.PauliError`, :class:`qibo.noise.ThermalRelaxationError` and
:class:`qibo.noise.ResetError`.

Here is an example on how to use a noise model:

.. testcode::

      import numpy as np
      from qibo import models, gates
      from qibo.noise import NoiseModel, PauliError

      # Build specific noise model with 3 quantum errors:
      # - Pauli error on H only for qubit 1.
      # - Pauli error on CNOT for all the qubits.
      # - Pauli error on RX(pi/2) for qubit 0.
      noise = NoiseModel()
      noise.add(PauliError([("X", 0.5)]), gates.H, 1)
      noise.add(PauliError([("Y", 0.5)]), gates.CNOT)
      is_sqrt_x = (lambda g: np.pi/2 in g.parameters)
      noise.add(PauliError([("X", 0.5)]), gates.RX, qubits=0, conditions=is_sqrt_x)

      # Generate noiseless circuit.
      c = models.Circuit(2)
      c.add([gates.H(0), gates.H(1), gates.CNOT(0, 1), gates.RX(0, np.pi/2),  gates.RX(0, 3*np.pi/2), gates.RX(1, np.pi/2)])

      # Apply noise to the circuit according to the noise model.
      noisy_c = noise.apply(c)

The noisy circuit defined above will be equivalent to the following circuit:

.. testcode::

      noisy_c2 = models.Circuit(2)
      noisy_c2.add(gates.H(0))
      noisy_c2.add(gates.H(1))
      noisy_c2.add(gates.PauliNoiseChannel(1, [("X", 0.5)]))
      noisy_c2.add(gates.CNOT(0, 1))
      noisy_c2.add(gates.PauliNoiseChannel(0, [("Y", 0.5)]))
      noisy_c2.add(gates.PauliNoiseChannel(1, [("Y", 0.5)]))
      noisy_c2.add(gates.RX(0, np.pi/2))
      noisy_c2.add(gates.PauliNoiseChannel(0, [("X", 0.5)]))
      noisy_c2.add(gates.RX(0, 3*np.pi/2))
      noisy_c2.add(gates.RX(1, np.pi/2))


The :class:`qibo.noise.NoiseModel` class supports also density matrices,
it is sufficient to pass a circuit which was initialized with ``density_matrix=True``.


.. _measurementbitflips-example:

Measurement errors
^^^^^^^^^^^^^^^^^^

:class:`qibo.measurements.CircuitResult` provides :meth:`qibo.measurements.CircuitResult.apply_bitflips`
which allows adding bit-flip errors to the sampled bit-strings without having to
re-execute the simulation. For example:

.. testcode::

      import numpy as np
      from qibo import models, gates

      thetas = np.random.random(4)
      c = models.Circuit(4)
      c.add((gates.RX(i, theta=t) for i, t in enumerate(thetas)))
      c.add([gates.M(0, 1), gates.M(2, 3)])
      result = c(nshots=100)
      # add bit-flip errors with probability 0.2 for all qubits
      result.apply_bitflips(0.2)
      # add bit-flip errors with different probabilities for each qubit
      error_map = {0: 0.2, 1: 0.1, 2: 0.3, 3: 0.1}
      result.apply_bitflips(error_map)

The corresponding noisy samples and frequencies can then be obtained as described
in the :ref:`How to perform measurements? <measurement-examples>` example.

Note that :meth:`qibo.measurements.CircuitResult.apply_bitflips` modifies
the measurement samples contained in the corresponding state and therefore the
original noiseless measurement samples are no longer accessible. It is possible
to keep the original samples by creating a copy of the states before applying
the bitflips:

.. testcode::

      import numpy as np
      from qibo import models, gates

      thetas = np.random.random(4)
      c = models.Circuit(4)
      c.add((gates.RX(i, theta=t) for i, t in enumerate(thetas)))
      c.add([gates.M(0, 1), gates.M(2, 3)])
      result = c(nshots=100)
      # add bit-flip errors with probability 0.2 for all qubits
      result.apply_bitflips(0.2)
      # add bit-flip errors with different probabilities for each qubit
      error_map = {0: 0.2, 1: 0.1, 2: 0.3, 3: 0.1}
      result.apply_bitflips(error_map)


Alternatively, the user may specify a bit-flip error map when defining
measurement gates:

.. testcode::

      import numpy as np
      from qibo import models, gates

      thetas = np.random.random(6)
      c = models.Circuit(6)
      c.add((gates.RX(i, theta=t) for i, t in enumerate(thetas)))
      c.add(gates.M(0, 1, p0=0.2))
      c.add(gates.M(2, 3, p0={2: 0.1, 3: 0.0}))
      c.add(gates.M(4, 5, p0=[0.4, 0.3]))
      result = c(nshots=100)

In this case ``result`` will contain noisy samples according to the given
bit-flip probabilities. The probabilities can be given as a
dictionary (must contain all measured qubits as keys),
a list (must have the sample as the measured qubits) or
a single float number (to be used on all measured qubits).
Note that, unlike the previous code example, when bit-flip errors are
incorporated as part of measurement gates it is not possible to access the
noiseless samples.

Moreover, it is possible to simulate asymmetric bit-flips using the ``p1``
argument as ``result.apply_bitflips(p0=0.2, p1=0.1)``. In this case a
probability of 0.2 will be used for 0->1 errors but 0.1 for 1->0 errors.
Similarly to ``p0``, ``p1`` can be a single float number or a dictionary and
can be used both in :meth:`qibo.measurements.CircuitResult.apply_bitflips`
and the measurement gate. If ``p1`` is not specified the value of ``p0`` will
be used for both errors.

.. _noise-hardware-example:

Simulating IBMQ's quantum hardware
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Qibo can perform a simulation of a real quantum computer using the
:class:`qibo.noise.IBMQNoiseModel` class.
It is possible by passing the circuit instance that we want to simulate
and the noise channels' parameters as a dictionary.
In this model, the user must set the relaxation times ``t1`` and ``t2`` for each qubit,
an approximated `gate times`, and depolarizing errors for each one-qubit (`depolarizing_one_qubit`)
and two-qubit (`depolarizing_two_qubit`) gates.
Additionally, one can also pass single-qubit readout error probabilities (`readout_one_qubit`).

.. testcode::

    from qibo import Circuit, gates
    from qibo.noise import IBMQNoiseModel

    nqubits = 2
    circuit = Circuit(2, density_matrix=True)
    circuit.add(
        [
            gates.H(0),
            gates.X(1),
            gates.Z(0),
            gates.X(0),
            gates.CNOT(0,1),
            gates.CNOT(1, 0),
            gates.X(1),
            gates.Z(1),
            gates.M(0),
            gates.M(1),
        ]
    )

    print("raw circuit:")
    circuit.draw()

    parameters = {
        "t1": {"0": 250*1e-06, "1": 240*1e-06},
        "t2": {"0": 150*1e-06, "1": 160*1e-06},
        "gate_times" : (200*1e-9, 400*1e-9),
        "excited_population" : 0,
        "depolarizing_one_qubit" : 4.000e-4,
        "depolarizing_two_qubit": 1.500e-4,
        "readout_one_qubit" : {"0": (0.022, 0.034), "1": (0.015, 0.041)},
        }

    noise_model = IBMQNoiseModel()
    noise_model.from_dict(parameters)
    noisy_circuit = noise_model.apply(circuit)

    print("noisy circuit:")
    noisy_circuit.draw()

.. testoutput::
   :hide:

   ...

``noisy_circuit`` is the new circuit containing the error gate channels.

.. #TODO: rewrite this optimization example after the fit function is moded to `qibo.optimizers`
.. It is possible to learn the parameters of the noise model that best describe a frequency distribution obtained by running a circuit on quantum hardware. To do this,
.. assuming we have a ``result`` object after running a circuit with a certain number of shots,

.. .. testcode::

..       noise = NoiseModel()
..       params = {"idle_qubits" : True}
..       noise.composite(params)

..       result =  noisy_circ(nshots=1000)

..       noise.noise_model.fit(c, result)

..       print(noise.noise_model.params)
..       print(noise.noise_model.hellinger)

.. .. testoutput::
..    :hide:

..    ...

.. where ``noise.params`` is a dictionary with the parameters obatined after the optimization and ``noise.hellinger`` is the corresponding Hellinger fidelity.


How to perform error mitigation?
--------------------------------

Noise and errors in circuits are one of the biggest obstacles to face in quantum computing.
Say that you have a circuit :math:`C` and you want to measure an observable :math:`A` at the end of it,
in general you are going to obtain an expected value :math:`\langle A \rangle_{noisy}` that
can lie quiet far from the true one :math:`\langle A \rangle_{exact}`.
In Qibo, different methods are implemented for mitigating errors in circuits and obtaining
a better estimate of the noise-free expected value :math:`\langle A \rangle_{exact}`.


Let's see how to use them. For starters, let's define a dummy circuit with some RZ, RX and CNOT gates:

.. testcode::

   import numpy as np

   from qibo import Circuit, gates

   # Define the circuit
   nqubits = 3
   hz = 0.5
   hx = 0.5
   dt = 0.25
   circ = Circuit(nqubits, density_matrix=True)
   circ.add(gates.RZ(q, theta=-2 * hz * dt - np.pi / 2) for q in range(nqubits))
   circ.add(gates.RX(q, theta=np.pi / 2) for q in range(nqubits))
   circ.add(gates.RZ(q, theta=-2 * hx * dt + np.pi) for q in range(nqubits))
   circ.add(gates.RX(q, theta=np.pi / 2) for q in range(nqubits))
   circ.add(gates.RZ(q, theta=-np.pi / 2) for q in range(nqubits))
   circ.add(gates.CNOT(q, q + 1) for q in range(0, nqubits - 1, 2))
   circ.add(gates.RZ(q + 1, theta=-2 * dt) for q in range(0, nqubits - 1, 2))
   circ.add(gates.CNOT(q, q + 1) for q in range(0, nqubits - 1, 2))
   circ.add(gates.CNOT(q, q + 1) for q in range(1, nqubits, 2))
   circ.add(gates.RZ(q + 1, theta=-2 * dt) for q in range(1, nqubits, 2))
   circ.add(gates.CNOT(q, q + 1) for q in range(1, nqubits, 2))
   # Include the measurements
   circ.add(gates.M(*range(nqubits)))

   # visualize the circuit
   circ.draw()

   #  q0: ─RZ─RX─RZ─RX─RZ─o────o────────M─
   #  q1: ─RZ─RX─RZ─RX─RZ─X─RZ─X─o────o─M─
   #  q2: ─RZ─RX─RZ─RX─RZ────────X─RZ─X─M─

.. testoutput::
   :hide:

   ...

remember to initialize the circuit with ``density_matrix=True`` and to include the measuerement gates at the end for expectation value calculation.

As observable we can simply take :math:`Z_0 Z_1 Z_2` :

.. testcode::

   from qibo.symbols import Z
   from qibo.hamiltonians import SymbolicHamiltonian

   backend = qibo.get_backend()

   # Define the observable
   obs = np.prod([Z(i) for i in range(nqubits)])
   obs = SymbolicHamiltonian(obs, backend=backend)

We can obtain the exact expected value by running the circuit on any simulation ``backend``. To mimic the execution on
the real quantum hardware, instead, we can use a noise model:

.. testcode::

   # Noise-free expected value
   exact = obs.expectation(backend.execute_circuit(circ).state())
   print(exact)
   # 0.9096065335014379

   from qibo.noise import DepolarizingError, ReadoutError, NoiseModel
   from qibo.quantum_info import random_stochastic_matrix

   # Define the noise model
   noise =  NoiseModel()
   # depolarizing error after each CNOT
   noise.add(DepolarizingError(0.1), gates.CNOT)
   # readout error
   # randomly initialize the bitflip probabilities
   prob = random_stochastic_matrix(
       2**nqubits, diagonally_dominant=True, seed=2, backend=backend
   )
   noise.add(ReadoutError(probabilities=prob), gate=gates.M)
   # Noisy expected value without mitigation
   noisy = obs.expectation(backend.execute_circuit(noise.apply(circ)).state())
   print(noisy)
   # 0.5647937721701448

.. testoutput::
   :hide:

   ...

Note that when running on the quantum hardware, you won't need to use a noise model
anymore, you will just have to change the backend to the appropriate one.

Now let's check that error mitigation produces better estimates of the exact expected value.

Readout Mitigation
^^^^^^^^^^^^^^^^^^
Firstly, let's try to mitigate the readout errors. To do this, we can either compute the
response matrix and use it modify the final state after the circuit execution:

.. testcode::

   from qibo.models.error_mitigation import get_expectation_val_with_readout_mitigation, get_response_matrix

   nshots = 10000
   # compute the response matrix
   response_matrix = get_response_matrix(
       nqubits, backend=backend, noise_model=noise, nshots=nshots
   )
   # define mitigation options
   readout = {"response_matrix": response_matrix}
   # mitigate the readout errors
   mit_val = get_expectation_val_with_readout_mitigation(circ, obs, noise, readout=readout)
   print(mit_val)
   # 0.5945794816381054

.. testoutput::
   :hide:

   ...

Or use the randomized readout mitigation:

.. testcode::

   from qibo.models.error_mitigation import apply_randomized_readout_mitigation

   # define mitigation options
   readout = {"ncircuits": 10}
   # mitigate the readout errors
   mit_val = get_expectation_val_with_readout_mitigation(circ, obs, noise, readout=readout)
   print(mit_val)
   # 0.5860884499785314

.. testoutput::
   :hide:

   ...

Alright, the expected value is improving, but we are still far from the ideal one.
Readout mitigation alone is not enough, let's try to use some more advanced methods
to get rid of the depolarizing error we introduced in the CNOT gates.

Zero Noise Extrapolation (ZNE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run ZNE, we just need to define the noise levels to use. Each level corresponds to the
number of CNOT or RX pairs (depending on the value of ``insertion_gate``) inserted in the
circuit in correspondence to the original ones. Since we decided to simulate noisy CNOTs::

   Level 1
   q0: ─X─  -->  q0: ─X───X──X─
   q1: ─o─  -->  q1: ─o───o──o─

   Level 2
   q0: ─X─  -->  q0: ─X───X──X───X──X─
   q1: ─o─  -->  q1: ─o───o──o───o──o─

   .
   .
   .

For example if we use the five levels ``[0,1,2,3,4]`` :

.. testcode::

   from qibo.models.error_mitigation import ZNE

   # Mitigated expected value
   estimate = ZNE(
       circuit=circ,
       observable=obs,
       noise_levels=np.arange(5),
       noise_model=noise,
       nshots=10000,
       insertion_gate='CNOT',
       backend=backend,
   )
   print(estimate)
   # 0.8332843749999996

.. testoutput::
   :hide:

   ...

we get an expected value closer to the exact one. We can further improve by using ZNE
combined with the readout mitigation:

.. testcode::

   # we can either use
   # the response matrix computed earlier
   readout = {'response_matrix': response_matrix}
   # or the randomized readout
   readout = {'ncircuits': 10}

   # Mitigated expected value
   estimate = ZNE(
       circuit=circ,
       observable=obs,
       backend=backend,
       noise_levels=np.arange(5),
       noise_model=noise,
       nshots=10000,
       insertion_gate='CNOT',
       readout=readout,
   )
   print(estimate)
   # 0.8979124892467807

.. testoutput::
   :hide:

   ...


Clifford Data Regression (CDR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For CDR instead, you don't need to define anything additional. However, keep in mind that the input
circuit is expected to be decomposed in the set of primitive gates :math:`RX(\frac{\pi}{2}), CNOT, X` and :math:`RZ(\theta)`.

.. testcode::

   from qibo.models.error_mitigation import CDR

   # Mitigated expected value
   estimate = CDR(
       circuit=circ,
       observable=obs,
       n_training_samples=10,
       backend=backend,
       noise_model=noise,
       nshots=10000,
       readout=readout,
   )
   print(estimate)
   # 0.8983676333969615

.. testoutput::
   :hide:

   ...

Again, the mitigated expected value improves over the noisy one and is also slightly better compared to ZNE.


Variable Noise CDR (vnCDR)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Being a combination of ZNE and CDR, vnCDR requires you to define the noise levels as done in ZNE, and the same
caveat about the input circuit for CDR is valid here as well.

.. testcode::

   from qibo.models.error_mitigation import vnCDR

   # Mitigated expected value
   estimate = vnCDR(
       circuit=circ,
       observable=obs,
       n_training_samples=10,
       backend=backend,
       noise_levels=np.arange(3),
       noise_model=noise,
       nshots=10000,
       insertion_gate='CNOT',
       readout=readout,
   )
   print(estimate)
   # 0.8998376314644383

.. testoutput::
   :hide:

   ...

The result is similar to the one obtained by CDR. Usually, one would expect slightly better results for vnCDR,
however, this can substantially vary depending on the circuit and the observable considered and, therefore, it is hard to tell
a priori.


Importance Clifford Sampling (ICS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The use of iCS is straightforward, analogous to CDR and vnCDR.

.. testcode::

   from qibo.models.error_mitigation import ICS

   # Mitigated expected value
   estimate = ICS(
       circuit=circ,
       observable=obs,
       n_training_samples=10,
       backend=backend,
       noise_model=noise,
       nshots=10000,
       readout=readout,
   )
   print(estimate)
   # 0.9183495097398502

.. testoutput::
   :hide:

   ...

Again, the mitigated expected value improves over the noisy one and is also slightly better compared to ZNE.
This was just a basic example usage of the three methods, for all the details about them you should check the API-reference page :ref:`Error Mitigation <error-mitigation>`.

.. _timeevol-example:

How to simulate time evolution?
-------------------------------

Simulating the unitary time evolution of quantum states is useful in many
physics applications including the simulation of adiabatic quantum computation.
Qibo provides the :class:`qibo.models.StateEvolution` model that simulates
unitary evolution using the full state vector. For example:

.. testcode::

    import numpy as np
    from qibo import hamiltonians, models

    # Define evolution model under the non-interacting sum(Z) Hamiltonian
    # with time step dt=1e-1
    nqubits = 4
    evolve = models.StateEvolution(hamiltonians.Z(nqubits), dt=1e-1)
    # Define initial state as |++++>
    initial_state = np.ones(2 ** nqubits) / np.sqrt(2 ** nqubits)
    # Get the final state after time t=2
    final_state = evolve(final_time=2, initial_state=initial_state)


When studying dynamics people are usually interested not only in the final state
vector but also in observing how physical quantities change during the time
evolution. This is possible using callbacks. For example, in the above case we
can track how <X> changes as follows:

.. testcode::

    import numpy as np
    from qibo import hamiltonians, models, callbacks

    nqubits = 4
    # Define a callback that calculates the energy (expectation value) of the X Hamiltonian
    observable = callbacks.Energy(hamiltonians.X(nqubits))
    # Create evolution object using the above callback and a time step of dt=1e-3
    evolve = models.StateEvolution(hamiltonians.Z(nqubits), dt=1e-3,
                                   callbacks=[observable])
    # Evolve for total time t=1
    initial_state = np.ones(2 ** nqubits) / np.sqrt(2 ** nqubits)
    final_state = evolve(final_time=1, initial_state=initial_state)

    print(observable[:])
    # will print an array of shape ``(1001,)`` that holds <X>(t) values
.. testoutput::
    :hide:

    ...


Note that the time step ``dt=1e-3`` defines how often we calculate <X> during
the evolution.

In the above cases the exact time evolution operator (exponential of the Hamiltonian)
was used to evolve the state vector. Because the evolution Hamiltonian is
time-independent, the matrix exponentiation happens only once. It is possible to
simulate time-dependent Hamiltonians by passing a function of time instead of
a :class:`qibo.hamiltonians.Hamiltonian` in the
:class:`qibo.models.StateEvolution` model. For example:

.. testcode::

    import numpy as np
    from qibo import hamiltonians, models

    # Defina a time dependent Hamiltonian
    nqubits = 4
    ham = lambda t: np.cos(t) * hamiltonians.Z(nqubits)
    # and pass it to the evolution model
    evolve = models.StateEvolution(ham, dt=1e-3)
    initial_state = np.ones(2 ** nqubits) / np.sqrt(2 ** nqubits)
    final_state = evolve(final_time=1, initial_state=initial_state)


The above script will still use the exact time evolution operator with the
exponentiation repeated for each time step. The integration method can
be changed using the ``solver`` argument when executing. The solvers that are
currently implemented are the default exponential solver (``"exp"``) and two
Runge-Kutta solvers: fourth-order (``"rk4"``) and fifth-order (``"rk45"``).
For more information we refer to the :ref:`Solvers <Solvers>` section.


.. _trotterdecomp-example:

Using Trotter decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Trotter decomposition provides a way to represent the unitary evolution of
quantum states as a sequence of local unitaries. This allows to represent
the physical process of time evolution as a quantum circuit. Qibo provides
functionality to perform this transformation automatically, if the underlying
Hamiltonian object is defined as a sum of commuting parts that consist of terms
that can be exponentiated efficiently.
Such Hamiltonian can be implemented in Qibo using
:class:`qibo.hamiltonians.SymbolicHamiltonian`.
The implementation of Trotter decomposition is based on Sec.
4.1 of `arXiv:1901.05824 <https://arxiv.org/abs/1901.05824>`_.
Below is an example of how to use this object in practice:

.. testcode::

    from qibo import hamiltonians

    # Define TFIM model as a non-dense ``SymbolicHamiltonian``
    ham = hamiltonians.TFIM(nqubits=5, dense=False)
    # This object can be used to create the circuit that
    # implements a single Trotter time step ``dt``
    circuit = ham.circuit(dt=1e-2)


This is a standard :class:`qibo.core.circuit.Circuit` that
contains :class:`qibo.gates.Unitary` gates corresponding to the
exponentials of the Trotter decomposition and can be executed on any state.

Note that in the transverse field Ising model (TFIM) that was used in this
example is among the pre-coded Hamiltonians in Qibo and could be created as
a :class:`qibo.hamiltonians.SymbolicHamiltonian` simply using the
``dense=False`` flag. For more information on the difference between dense
and non-dense Hamiltonians we refer to the :ref:`Hamiltonians <Hamiltonians>`
section. Note that only non-dense Hamiltonians created using ``dense=False``
or through the :class:`qibo.hamiltonians.SymbolicHamiltonian` object
can be used for evolution using Trotter decomposition. If a dense Hamiltonian
is used then evolution will be done by exponentiating the full Hamiltonian
matrix.

Defining custom Hamiltonians from terms can be more complicated,
however Qibo simplifies this process by providing the option
to define Hamiltonians symbolically through the use of ``sympy``.
For more information on this we refer to the
:ref:`How to define custom Hamiltonians using symbols? <symbolicham-example>`
example.

A :class:`qibo.hamiltonians.SymbolicHamiltonian` can also be used to
simulate time evolution. This can be done by passing the Hamiltonian to a
:class:`qibo.models.StateEvolution` model and using the exponential solver.
For example:

.. testcode::

    import numpy as np
    from qibo import models, hamiltonians

    nqubits = 5
    # Create a critical TFIM Hamiltonian as ``SymbolicHamiltonian``
    ham = hamiltonians.TFIM(nqubits=nqubits, h=1.0, dense=False)
    # Define the |+++++> initial state
    initial_state = np.ones(2 ** nqubits) / np.sqrt(2 ** nqubits)
    # Define the evolution model
    evolve = models.StateEvolution(ham, dt=1e-3)
    # Evolve for total time T=1
    final_state = evolve(final_time=1, initial_state=initial_state)

This script creates the Trotter circuit for ``dt=1e-3`` and applies it
repeatedly to the given initial state T / dt = 1000 times to obtain the
final state of the evolution.

Since Trotter evolution is based on Qibo circuits, it also supports distributed
execution on multiple devices (GPUs). This can be enabled by passing an
``accelerators`` dictionary when defining the
:class:`qibo.models.StateEvolution` model. We refer to the
:ref:`How to select hardware devices? <gpu-examples>` example for more details
on how the ``accelerators`` dictionary can be used.


How to simulate adiabatic time evolution?
-----------------------------------------

Qibo provides the :class:`qibo.models.AdiabaticEvolution` model to simulate
adiabatic time evolution. This is a special case of the
:class:`qibo.models.StateEvolution` model analyzed in the previous example
where the evolution Hamiltonian is interpolated between an initial "easy"
Hamiltonian and a "hard" Hamiltonian that usually solves an optimization problem.
Here is an example of adiabatic evolution simulation:

.. testcode::

    import numpy as np
    from qibo import hamiltonians, models

    nqubits = 4
    T = 1 # total evolution time
    # Define the easy and hard Hamiltonians
    h0 = hamiltonians.X(nqubits)
    h1 = hamiltonians.TFIM(nqubits, h=0)
    # Define the interpolation scheduling
    s = lambda t: t
    # Define evolution model
    evolve = models.AdiabaticEvolution(h0, h1, s, dt=1e-2)
    # Get the final state of the evolution
    final_state = evolve(final_time=T)


According to the adiabatic theorem, for proper scheduling and total evolution
time the ``final_state`` should approximate the ground state of the "hard"
Hamiltonian.

If the initial state is not specified, the ground state of the easy Hamiltonian
will be used, which is common for adiabatic evolution. A distributed execution
can be used by passing an ``accelerators`` dictionary during the initialization
of the ``AdiabaticEvolution`` model. In this case the default initial state is
``|++...+>`` (full superposition in the computational basis).

Callbacks may also be used as in the previous example. An additional callback
(:class:`qibo.callbacks.Gap`) is available for calculating the
energies and the gap of the adiabatic evolution Hamiltonian. Its usage is
similar to other callbacks:

.. testcode::

    import numpy as np
    from qibo import hamiltonians, models, callbacks

    nqubits = 4
    h0 = hamiltonians.X(nqubits)
    h1 = hamiltonians.TFIM(nqubits, h=0)

    ground = callbacks.Gap(mode=0)
    # define a callback for calculating the gap
    gap = callbacks.Gap()
    # define and execute the ``AdiabaticEvolution`` model
    evolution = models.AdiabaticEvolution(h0, h1, lambda t: t, dt=1e-1,
                                          callbacks=[gap, ground])

    final_state = evolution(final_time=1.0)
    # print the values of the gap at each evolution time step
    print(gap[:])
.. testoutput::
    :hide:

    ...


The scheduling function ``s`` should be a callable that accepts one (s(t)) or
two (s(t, p)) arguments. The first argument accepts values in [0, 1] and
corresponds to the ratio ``t / final_time`` during evolution. The second
optional argument is a vector of free parameters that can be optimized. The
function should, by definition, satisfy the properties s(0, p) = 0 and
s(1, p) = 1 for any p, otherwise errors will be raised.

All state evolution functionality described in the previous example can also be
used for simulating adiabatic evolution. The solver can be specified during the
initialization of the :class:`qibo.models.AdiabaticEvolution` model and a
Trotter decomposition may be used with the exponential solver. The Trotter
decomposition will be used automatically if ``h0`` and ``h1`` are defined
using as :class:`qibo.hamiltonians.SymbolicHamiltonian` objects. For
pre-coded Hamiltonians this can be done simply as:

.. testcode::

    from qibo import hamiltonians, models

    nqubits = 4
    # Define ``SymolicHamiltonian``s
    h0 = hamiltonians.X(nqubits, dense=False)
    h1 = hamiltonians.TFIM(nqubits, h=0, dense=False)
    # Perform adiabatic evolution using the Trotter decomposition
    evolution = models.AdiabaticEvolution(h0, h1, lambda t: t, dt=1e-1)
    final_state = evolution(final_time=1.0)


When Trotter evolution is used, it is also possible to execute on multiple
devices by passing an ``accelerators`` dictionary in the creation of the
:class:`qibo.models.AdiabaticEvolution` model.

Note that ``h0`` and ``h1`` should have the same type, either both
:class:`qibo.hamiltonians.Hamiltonian` or both
:class:`qibo.hamiltonians.SymbolicHamiltonian`.


Optimizing the scheduling function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The free parameters ``p`` of the scheduling function can be optimized using
the :meth:`qibo.models.AdiabaticEvolution.minimize` method. The parameters
are optimized so that the final state of the adiabatic evolution approximates
the ground state of the "hard" Hamiltonian. Optimization is similar to what is
described in the :ref:`How to write a VQE? <vqe-example>` example and can be
done as follows:

.. testcode::

    import numpy as np
    from qibo import hamiltonians, models

    # Define Hamiltonians
    h0 = hamiltonians.X(3)
    h1 = hamiltonians.TFIM(3)
    # Define scheduling function with a free variational parameter ``p``
    sp = lambda t, p: (1 - p) * np.sqrt(t) + p * t
    # Define an evolution model with dt=1e-2
    evolution = models.AdiabaticEvolution(h0, h1, sp, dt=1e-2)
    # Find the optimal value for ``p`` starting from ``p = 0.5`` and ``T=1``.
    initial_guess = [0.5, 1]
    # best, params, extra = evolution.minimize(initial_guess, method="BFGS", options={'disp': True})
    print(best) # prints the best energy <H1> found from the final state
    print(params) # prints the optimal values for the parameters.
.. testoutput::
    :hide:

    ...

Note that the ``minimize`` method optimizes both the free parameters ``p`` of
the scheduling function as well as the total evolution time. The initial guess
for the total evolution time is the last value of the given ``initial_guess``
array. For a list of the available optimizers we refer to
:ref:`Optimizers <Optimizers>`.


.. _symbolicham-example:

How to define custom Hamiltonians using symbols?
------------------------------------------------

In order to use the VQE, QAOA and time evolution models in Qibo the user has to
define Hamiltonians based on :class:`qibo.hamiltonians.Hamiltonian` which
uses the full matrix representation of the corresponding operator or
:class:`qibo.hamiltonians.SymbolicHamiltonian` which uses a more efficient
term representation. Qibo provides pre-coded Hamiltonians for some common models,
such as the transverse field Ising model (TFIM) and the Heisenberg model
(see :ref:`Hamiltonians <Hamiltonians>` for a complete list of the pre-coded models).
In order to explore other problems the user needs to define the Hamiltonian
objects from scratch.

A standard way to define Hamiltonians is through their full matrix
representation. For example the following code generates the TFIM Hamiltonian
with periodic boundary conditions for four qubits by constructing the
corresponding 16x16 matrix:

.. testcode::

    import numpy as np
    from qibo import hamiltonians, matrices

    # ZZ terms
    matrix = np.kron(np.kron(matrices.Z, matrices.Z), np.kron(matrices.I, matrices.I))
    matrix += np.kron(np.kron(matrices.I, matrices.Z), np.kron(matrices.Z, matrices.I))
    matrix += np.kron(np.kron(matrices.I, matrices.I), np.kron(matrices.Z, matrices.Z))
    matrix += np.kron(np.kron(matrices.Z, matrices.I), np.kron(matrices.I, matrices.Z))
    # X terms
    matrix += np.kron(np.kron(matrices.X, matrices.I), np.kron(matrices.I, matrices.I))
    matrix += np.kron(np.kron(matrices.I, matrices.X), np.kron(matrices.I, matrices.I))
    matrix += np.kron(np.kron(matrices.I, matrices.I), np.kron(matrices.X, matrices.I))
    matrix += np.kron(np.kron(matrices.I, matrices.I), np.kron(matrices.I, matrices.X))
    # Create Hamiltonian object
    ham = hamiltonians.Hamiltonian(4, matrix)


Although it is possible to generalize the above construction to arbitrary number
of qubits this procedure may be more complex for other Hamiltonians. Moreover
constructing the full matrix does not scale well with increasing the number of
qubits. This makes the use of :class:`qibo.hamiltonians.SymbolicHamiltonian`
preferrable as the qubit number increases, as this Hamiltonians is not based
in the full matrix representation.

To simplify the construction of Hamiltonians, Qibo provides the
:class:`qibo.hamiltonians.SymbolicHamiltonian` object which
allows the user to construct Hamiltonian objects by writing their symbolic
form using ``sympy`` symbols. Moreover Qibo provides quantum-computation specific
symbols (:class:`qibo.symbols.Symbol`) such as the Pauli operators.
For example, the TFIM on four qubits could be constructed as:

.. testcode::

    import numpy as np
    from qibo import hamiltonians
    from qibo.symbols import X, Z

    # Define Hamiltonian using Qibo symbols
    # ZZ terms
    symbolic_ham = sum(Z(i) * Z(i + 1) for i in range(3))
    # periodic boundary condition term
    symbolic_ham += Z(0) * Z(3)
    # X terms
    symbolic_ham += sum(X(i) for i in range(4))

    # Define a Hamiltonian using the above form
    ham = hamiltonians.SymbolicHamiltonian(symbolic_ham)
    # This Hamiltonian is memory efficient as it does not construct the full matrix

    # The corresponding dense Hamiltonian which contains the full matrix can
    # be constructed easily as
    dense_ham = ham.dense
    # and the matrix is accessed as ``dense_ham.matrix`` or ``ham.matrix``.


Defining Hamiltonians from symbols is usually a simple process as the symbolic
form is very close to the form of the Hamiltonian on paper. Note that when a
:class:`qibo.hamiltonians.SymbolicHamiltonian` is used for time evolution,
Qibo handles automatically automatically the Trotter decomposition by splitting
to the appropriate terms.

Qibo symbols support an additional ``commutative`` argument which is set to
``False`` by default since quantum operators are non-commuting objects.
When the user knows that the Hamiltonian consists of commuting terms only, such
as products of Z operators, switching ``commutative`` to ``True`` may speed-up
some symbolic calculations, such as the ``sympy.expand`` used when calculating
the Trotter decomposition for the Hamiltonian. This option can be used when
constructing each symbol:


.. testcode::

    from qibo import hamiltonians
    from qibo.symbols import Z

    form = Z(0, commutative=True) * Z(1, commutative=True) + Z(1, commutative=True) * Z(2, commutative=True)
    ham = hamiltonians.SymbolicHamiltonian(form)


.. _hamexpectation-example:

How to calculate expectation values using samples?
--------------------------------------------------

It is possible to calculate the expectation value of a :class:`qibo.hamiltonians.Hamiltonian`
on a given state using the :meth:`qibo.hamiltonians.Hamiltonian.expectation` method,
which can be called on a state or density matrix. For example


.. testcode::

    from qibo import Circuit, gates
    from qibo.hamiltonians import XXZ

    circuit = Circuit(4)
    circuit.add(gates.H(i) for i in range(4))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.CNOT(2, 3))

    hamiltonian = XXZ(4)

    result = circuit()
    expectation_value = hamiltonian.expectation(result.state())

In this example, the circuit will be simulated to obtain the final state vector
and the corresponding expectation value will be calculated through exact matrix
multiplication with the Hamiltonian matrix.
If a :class:`qibo.hamiltonians.SymbolicHamiltonian` is used instead, the expectation
value will be calculated as a sum of expectation values of local terms, allowing
calculations of more qubits with lower memory consumption. The calculation of each
local term still requires the state vector.

When executing a circuit on real hardware, usually only measurements of the state are
available, not the state vector. Qibo provides :meth:`qibo.hamiltonians.Hamiltonian.expectation_from_samples`
to allow calculation of expectation values directly from such samples:


.. testcode::

    from qibo import Circuit, gates
    from qibo import hamiltonians

    circuit = Circuit(4)
    circuit.add(gates.H(i) for i in range(4))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.CNOT(2, 3))
    circuit.add(gates.M(*range(4)))

    hamiltonian = hamiltonians.Z(4)

    result = circuit(nshots=1024)
    expectation_value = hamiltonian.expectation_from_samples(result.frequencies())


This example simulates the circuit similarly to the previous one but calculates
the expectation value using the frequencies of shots, instead of the exact state vector.
This can also be invoked directly from the ``result`` object:

.. testcode::

    expectation_value = result.expectation_from_samples(hamiltonian)


For Hamiltonians that are not diagonal in the computational basis, or that are sum of terms that cannot be
diagonalised simultaneously, one has to calculate the expectation value starting from the circuit:

.. testcode::

   from qibo.symbols import X, Y, Z
   from qibo.hamiltonians import SymbolicHamiltonian

   # build the circuit as before
   circuit = Circuit(4)
   circuit.add(gates.H(i) for i in range(4))
   circuit.add(gates.CNOT(0, 1))
   circuit.add(gates.CNOT(1, 2))
   circuit.add(gates.CNOT(2, 3))
   # but don't add any measurement at the end!
   # they will be automatically added with the proper basis
   # while calculating the expectation value

   hamiltonian = SymbolicHamiltonian(3 * Z(2) * (1 - X(1)) ** 2 - (Y(0) * X(3)) / 2, nqubits=4)
   expectation_value = hamiltonian.expectation_from_circuit(circuit)

What is happening under the hood in this case, is that the expectation value is calculated for each term
individually by measuring the circuit in the correct (rotated) basis. All the contributions are then
summed to recover the global expectation value. This means, in particular, that several copies of the
circuit are parallely executed, one for each term of the Hamiltonian. Note that, at the moment, no
check is performed to verify whether a subset of the terms could be diagonalised simultaneously, but
rather each term is treated separately every time.


.. _tutorials_transpiler:

How to modify the transpiler?
-----------------------------

Logical quantum circuits for quantum algorithms are hardware agnostic. Usually an all-to-all qubit connectivity
is assumed while most current hardware only allows the execution of two-qubit gates on a restricted subset of qubit
pairs. Moreover, quantum devices are restricted to executing a subset of gates, referred to as native.
This means that, in order to execute circuits on a real quantum chip, they must be transformed into an equivalent,
hardware specific, circuit. The transformation of the circuit is carried out by the transpiler through the resolution
of two key steps: connectivity matching and native gates decomposition.
In order to execute a gate between two qubits that are not directly connected SWAP gates are required. This procedure is called routing.
As on NISQ devices two-qubit gates are a large source of noise, this procedure generates an overall noisier circuit.
Therefore, the goal of an efficient routing algorithm is to minimize the number of SWAP gates introduced.
An important step to ease the connectivity problem, is finding anoptimal initial mapping between logical and physical qubits.
This step is called placement.
The native gates decomposition in the transpiling procedure is performed by the unroller. An optimal decomposition uses the least amount
of two-qubit native gates. It is also possible to reduce the number of gates of the resulting circuit by exploiting
commutation relations, KAK decomposition or machine learning techniques.
Qibo implements a built-in transpiler with customizable options for each step. The main algorithms that can
be used at each transpiler step are reported below with a short description.

The initial placement can be found with one of the following procedures:
- Trivial: logical-physical qubit mapping is an identity.
- Custom: custom logical-physical qubit mapping.
- Random greedy: the best mapping is found within a set of random layouts based on a greedy policy.
- Subgraph isomorphism: the initial mapping is the one that guarantees the execution of most gates at
the beginning of the circuit without introducing any SWAP.
- Reverse traversal: this technique uses one or more reverse routing passes to find an optimal mapping by
starting from a trivial layout.

The routing problem can be solved with the following algorithms:
- Shortest paths: when unconnected logical qubits have to interact, they are moved on the chip on
the shortest path connecting them. When multiple shortest paths are present, the one that also matches
the largest number of the following two-qubit gates is chosen.
- Sabre: this heuristic routing technique uses a customizable cost function to add SWAP gates
that reduce the distance between unconnected qubits involved in two-qubit gates.

Qibolab unroller applies recursively a set of hard-coded gates decompositions in order to translate any gate into
single and two-qubit native gates. Single qubit gates are translated into U3, RX, RZ, X and Z gates. It is possible to
fuse multiple single qubit gates acting on the same qubit into a single U3 gate. For the two-qubit native gates it
is possible to use CZ and/or iSWAP. When both CZ and iSWAP gates are available the chosen decomposition is the
one that minimizes the use of two-qubit gates.

Multiple transpilation steps can be implemented using the :class:`qibo.transpiler.pipeline.Pipeline`:

.. testcode:: python

    import networkx as nx

    from qibo import gates
    from qibo.models import Circuit
    from qibo.transpiler.pipeline import Passes, assert_transpiling
    from qibo.transpiler.optimizer import Preprocessing
    from qibo.transpiler.router import ShortestPaths
    from qibo.transpiler.unroller import Unroller, NativeGates
    from qibo.transpiler.placer import Random

    # Define connectivity as nx.Graph
    def star_connectivity():
        chip = nx.Graph()
        chip.add_nodes_from(list(range(5)))
        graph_list = [(i, 2) for i in range(5) if i != 2]
        chip.add_edges_from(graph_list)
        return chip

    # Define the circuit
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CZ(0, 1))

    # Define custom passes as a list
    custom_passes = []
    # Preprocessing adds qubits in the original circuit to match the number of qubits in the chip
    custom_passes.append(Preprocessing(connectivity=star_connectivity()))
    # Placement step
    custom_passes.append(Random(connectivity=star_connectivity()))
    # Routing step
    custom_passes.append(ShortestPaths(connectivity=star_connectivity()))
    # Gate decomposition step
    custom_passes.append(Unroller(native_gates=NativeGates.default()))

    # Define the general pipeline
    custom_pipeline = Passes(custom_passes, connectivity=star_connectivity(), native_gates=NativeGates.default())

    # Call the transpiler pipeline on the circuit
    transpiled_circ, final_layout = custom_pipeline(circuit)

    # Optinally call assert_transpiling to check that the final circuit can be executed on hardware
    # For this test it is necessary to get the initial layout
    initial_layout = custom_pipeline.get_initial_layout()
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circ,
        connectivity=star_connectivity(),
        initial_layout=initial_layout,
        final_layout=final_layout,
        native_gates=NativeGates.default()
    )

In this case circuits will first be transpiled to respect the 5-qubit star connectivity, with qubit 2 as the middle qubit. This will potentially add some SWAP gates.
Then all gates will be converted to native. The :class:`qibo.transpiler.unroller.Unroller` transpiler used in this example assumes Z, RZ, GPI2 or U3 as
the single-qubit native gates, and supports CZ and iSWAP as two-qubit natives. In this case we restricted the two-qubit gate set to CZ only.
The final_layout contains the final logical-physical qubit mapping.

.. _gst_example:

How to perform Gate Set Tomography?
-----------------------------------

In order to obtain an estimated representation of a set of quantum gates in a particular noisy environment, qibo provides a GST routine in its tomography module.

Let's first define the set of gates we want to estimate:

.. testcode::

   from qibo import gates

   gate_set = {gates.X, gates.H, gates.CZ}

For simulation purposes we can define a noise model. Naturally this is not needed when running on real quantum hardware, which is intrinsically noisy. For example, we can suppose that the three gates we want to estimate are going to be noisy:

.. testcode::

   from qibo.noise import NoiseModel, DepolarizingError

   noise_model = NoiseModel()
   noise_model.add(DepolarizingError(1e-3), gates.X)
   noise_model.add(DepolarizingError(1e-2), gates.H)
   noise_model.add(DepolarizingError(3e-2), gates.CZ)

Then the estimated representation of the gates in this noisy environment can be extracted by running the GST:

.. testcode::

   from qibo.tomography import GST

   estimated_gates = GST(
       gate_set = gate_set,
       nshots = 10000,
       noise_model = noise_model
   )

In some cases the empty circuit matrix :math:`E` can also be useful, and can be returned by setting the ``include_empty`` argument to ``True``:

.. testcode::

   empty_1q, empty_2q, *estimated_gates = GST(
       gate_set = gate_set,
       nshots = 10000,
       noise_model = noise_model,
       include_empty = True,
   )

where ``empty_1q`` and ``empty_2q`` correspond to the single and two qubits empty matrices respectively.
Similarly, the Pauli-Liouville representation of the gates can be directly returned as well:

.. testcode::

   estimated_gates = GST(
       gate_set = gate_set,
       nshots = 10000,
       noise_model = noise_model,
       pauli_liouville = True,
   )
