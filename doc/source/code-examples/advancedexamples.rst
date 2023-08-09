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

If a GPU with CUDA support is available in the system and Tensorflow is installed
for CUDA then circuits will be executed on the GPU automatically unless the user
specifies otherwise. In order to force the device a circuit will be executed
one can use:

.. code-block::  python

    with tf.device("/CPU:0"):
        # execute circuit on CPU with default initial state |000...0>.
        final_state = c()

or switch the default Qibo device using ``qibo.set_device`` as:

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
hides GPUs from tensorflow. As a result, any program executed from the same
terminal will run on CPU even if ``tf.device`` is not used.

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

On the other hand, when using the ``tensorflow`` backend Qibo inherits
Tensorflow's defaults for CPU thread configuration.
Tensorflow allows restricting the number of threads as follows:

.. code-block:: python

    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    import qibo

Note that this should be run during Tensorflow initialization in the beginning
of the script and before any circuit or gate allocation.

Using multiple GPUs
^^^^^^^^^^^^^^^^^^^

Qibo supports distributed circuit execution on multiple GPUs. This feature can
be used as follows:

.. code-block:: python

    from qibo.models import Circuit
    from qibo import gates

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

For more details in the distributed implementation one can look in the related
code: :class:`qibo.core.distcircuit.DistributedCircuit`. When
``models.Circuit`` is called then this distributed implementation is used automatically
if the ``accelerators`` dictionary is passed, otherwise the standard single device
:class:`qibo.core.circuit.Circuit` is used.

Note that executing a:class:`qibo.core.distcircuit.DistributedCircuit`
will return a :class:`qibo.core.distutils.DistributedState` which holds
the state vector partitioned in multiple pieces which are distributed to the
different devices during the simulation.
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


Note that indexing a class:`qibo.core.distutils.DistributedState` uses
the state pieces without requiring to merge and create the single state vector
tensor.


How to use callbacks?
---------------------

Callbacks allow the user to apply additional functions on the state vector
during circuit execution. An example use case of this is the calculation of
entanglement entropy as the state propagates through a circuit. This can be
implemented easily using :class:`qibo.abstractions.callbacks.EntanglementEntropy`
and the :class:`qibo.abstractions.gates.CallbackGate` gate. For example:

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
example ``entropy[:]`` will return ``tf.Tensor([0, 0, 1])`` which are the
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
    print(entropy[:]) # tf.Tensor([0, 0, 1, 0, 0, 1])
.. testoutput::
    :hide:

    ...

The callback for entanglement entropy can also be used on state vectors directly.
For example

.. testcode::

    import numpy as np
    from qibo import callbacks
    # import the backend to access math ops directly
    from qibo import K
    # create a singlet state vector
    state = K.zeros(4)
    state[0], state[3] = 1 / K.sqrt(2), 1 / K.sqrt(2)

    # create an `EntanglementEntropy` callback object
    entropy = callbacks.EntanglementEntropy([0])
    # call the object on the state
    print(entropy(state))
.. testoutput::
    :hide:

    ...

will print ``tf.Tensor(1.0)``.

.. _params-examples:

How to use parametrized gates?
------------------------------

Some Qibo gates such as rotations accept values for their free parameter. Once
such gates are added in a circuit their parameters can be updated using the
:meth:`qibo.abstractions.circuit.AbstractCircuit.set_parameters` method. For example:

.. testcode::

    from qibo.models import Circuit
    from qibo import gates
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

    from qibo.models import Circuit
    from qibo import gates

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
* :class:`qibo.abstractions.gates.fSim`: Accepts a tuple of two parameters ``(theta, phi)``.
* :class:`qibo.abstractions.gates.GeneralizedfSim`: Accepts a tuple of two parameters
  ``(unitary, phi)``. Here ``unitary`` should be a unitary matrix given as an
  array or ``tf.Tensor`` of shape ``(2, 2)``.
* :class:`qibo.abstractions.gates.Unitary`: Accepts a single ``unitary`` parameter. This
  should be an array or ``tf.Tensor`` of shape ``(2, 2)``.
* :class:`qibo.abstractions.gates.VariationalLayer`: Accepts a list of ``float``
  parameters with length compatible to the number of one qubit rotations implemented
  by the layer, for example:

.. testcode::

    import numpy as np
    from qibo.models import Circuit
    from qibo import gates

    nqubits = 5
    c = Circuit(nqubits)
    pairs = [(i, i + 1) for i in range(0, 4, 2)]
    c.add(gates.VariationalLayer(range(nqubits), pairs,
                                 gates.RY, gates.CZ,
                                 params=np.zeros(5)))
    c.add((gates.RX(i, theta=0) for i in range(5)))

    # set random parameters to all rotations in the circuit
    c.set_parameters(np.random.random(10))
    # note that 10 numbers are used as the VariationalLayer contains five
    # rotations and five additional RX rotations are added afterwards.

Note that a ``np.ndarray`` or a ``tf.Tensor`` may also be used in the place of
a flat list. Using :meth:`qibo.abstractions.circuit.AbstractCircuit.set_parameters` is more
efficient than recreating a new circuit with new parameter values. The inverse
method :meth:`qibo.abstractions.circuit.AbstractCircuit.get_parameters` is also available
and returns a list, dictionary or flat list with the current parameter values
of all parametrized gates in the circuit.

It is possible to hide a parametrized gate from the action of
:meth:`qibo.abstractions.circuit.AbstractCircuit.get_parameters` and
:meth:`qibo.abstractions.circuit.AbstractCircuit.set_parameters` by setting
the ``trainable=False`` during gate creation. For example:

.. testsetup::

    from qibo.models import Circuit
    from qibo import gates

.. testcode::

    c = Circuit(3)
    c.add(gates.RX(0, theta=0.123))
    c.add(gates.RY(1, theta=0.456, trainable=False))
    c.add(gates.fSim(0, 2, theta=0.789, phi=0.567))

    print(c.get_parameters())
    # prints [0.123, (0.789, 0.567)] ignoring the parameters of the RY gate

.. testoutput::

    [0.123, (0.789, 0.567)]


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
of the :class:`qibo.abstractions.gates.M` gate. For example

.. testcode::

    from qibo.models import Circuit
    from qibo import gates

    c = Circuit(1)
    c.add(gates.H(0))
    output = c.add(gates.M(0, collapse=True))
    c.add(gates.H(0))
    result = c()
    print(result.state())
    # prints [0.7071, 0.7071] if 0 is measured
    # or [0.7071, -0.7071] if 1 is measured
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

    from qibo.models import Circuit
    from qibo import gates

    c = Circuit(1)
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

    from qibo.models import Circuit
    from qibo import gates

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
    from qibo.models import Circuit
    from qibo import gates

    c = Circuit(2)
    c.add(gates.H(0))
    output = c.add(gates.M(0, collapse=True))
    c.add(gates.RX(1, theta=np.pi * output / 4))
    result = c()

In this case the first qubit will be measured and if 1 is found a pi/4 X-rotation
will be applied to the second qubit, otherwise no rotation. Qibo allows to
use ``output`` as a parameter during circuit creation by representing it using
a ``sympy.Symbol``. The symbol acquires a numerical value later during execution
when the measurement is performed. As explained above, if a ``nshots > 1`` is
given during circuit execution the execution is repeated using a loop.

If more than one qubits are used in a ``collapse=True`` measurement gate the
``output`` can be indexed accordingly:

.. testcode::

    import numpy as np
    from qibo.models import Circuit
    from qibo import gates

    c = Circuit(3)
    c.add(gates.H(0))
    output = c.add(gates.M(0, 1, collapse=True))
    c.add(gates.RX(1, theta=np.pi * output[0] / 4))
    c.add(gates.RY(2, theta=np.pi * (output[0] + output[1]) / 5))
    result = c()


How to invert a circuit?
------------------------

Many quantum algorithms require using a specific subroutine and its inverse
in the same circuit. Qibo simplifies this implementation via the
:meth:`qibo.abstractions.circuit.AbstractCircuit.invert` method. This method produces
the inverse of a circuit by taking the dagger of all gates in reverse order. It
can be used with circuit addition to simplify the construction of algorithms,
for example:

.. testcode::

    from qibo.models import Circuit
    from qibo import gates

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
large circuit. This is possible using the :meth:`qibo.abstractions.circuit.AbstractCircuit.on_qubits`
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
has to use a backend based on tensorflow primitives and not the default custom
backend, as custom operators currently do not support automatic differentiation.
To switch the backend one can do ``qibo.set_backend("tensorflow")``.
Check the :ref:`How to use automatic differentiation? <autodiff-example>`
section for more details.

A useful gate for defining the ansatz of the VQE is :class:`qibo.abstractions.gates.VariationalLayer`.
This optimizes performance by fusing the layer of one-qubit parametrized gates with
the layer of two-qubit entangling gates and applying both as a single layer of
general two-qubit gates (as 4x4 matrices). The ansatz from the above example can
be written using :class:`qibo.abstractions.gates.VariationalLayer` as follows:

.. testsetup::

    import numpy as np
    from qibo import models, gates, hamiltonians

.. testcode::

    circuit = models.Circuit(nqubits)
    pairs = [(i, i + 1) for i in range(0, nqubits - 1, 2)]
    theta = np.zeros(nqubits)
    for l in range(nlayers):
        circuit.add(gates.VariationalLayer(range(nqubits), pairs,
                                           gates.RY, gates.CZ,
                                           theta, theta))
        circuit.add((gates.CZ(i, i + 1) for i in range(1, nqubits - 2, 2)))
        circuit.add(gates.CZ(0, nqubits - 1))
    circuit.add((gates.RY(i, theta) for i in range(nqubits)))

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
        final_state = circuit().numpy()
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
that can be defined using a :class:`qibo.abstractions.hamiltonians.Hamiltonian`. When
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
When a :class:`qibo.abstractions.hamiltonians.Hamiltonian` is used then solvers will
exponentiate it using its full matrix. Alternatively, if a
:class:`qibo.core.hamiltonians.SymbolicHamiltonian` is used then solvers
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

As a deep learning framework, Tensorflow supports
`automatic differentiation <https://www.tensorflow.org/tutorials/customization/autodiff>`_.
This can be used to optimize the parameters of variational circuits. For example
the following script optimizes the parameters of two rotations so that the circuit
output matches a target state using the fidelity as the corresponding loss
function.

.. testcode::

    import qibo
    qibo.set_backend("tensorflow")
    import tensorflow as tf
    from qibo import gates, models

    # Optimization parameters
    nepochs = 1000
    optimizer = tf.keras.optimizers.Adam()
    target_state = tf.ones(4, dtype=tf.complex128) / 2.0

    # Define circuit ansatz
    params = tf.Variable(tf.random.uniform((2,), dtype=tf.float64))
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


Note that the ``"tensorflow"`` backend has to be used here because other custom
backends do not support automatic differentiation.

The optimization procedure may also be compiled, however in this case it is not
possible to use :meth:`qibo.abstractions.circuit.AbstractCircuit.set_parameters` as the
circuit needs to be defined inside the compiled ``tf.GradientTape()``.
For example:

.. code-block:: python

    import qibo
    qibo.set_backend("tensorflow")
    import tensorflow as tf
    from qibo import gates, models

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

.. testsetup::

    import qibo
    qibo.set_backend("qibojit")

.. testcode::

    from qibo import models, gates

    # Define circuit
    c = models.Circuit(2, density_matrix=True)
    c.add(gates.H(0))
    c.add(gates.H(1))
    # execute using the default initial state |00><00|
    result = c()
    # result.state() will be tf.ones(4) / 4 which corresponds to |++><++|

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
    c.add(gates.PauliNoiseChannel(0, px=0.3))
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
    c.add((gates.PauliNoiseChannel(i, px=0.2, py=0.0, pz=0.3)
           for i in range(5)))
    # Add measurement of all qubits
    c.add(gates.M(*range(5)))

    # Repeat execution 1000 times
    result = c(nshots=1000)

In this example the simulation is repeated 1000 times and the action of the
:class:`qibo.abstractions.gates.PauliNoiseChannel` gate differs each time, because
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
:class:`qibo.abstractions.gates.UnitaryChannel` and
:class:`qibo.abstractions.gates.PauliNoiseChannel` can be used with sampling, while
:class:`qibo.abstractions.gates.KrausChannel` requires density matrices.


Adding noise after every gate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In practical applications noise typically occurs after every gate.
Qibo provides the :meth:`qibo.abstractions.circuit.AbstractCircuit.with_noise()` method
which automatically creates a new circuit that contains a
:class:`qibo.abstractions.gates.PauliNoiseChannel` after every gate.
The user can control the probabilities of the noise channel using a noise map,
which is a dictionary that maps qubits to the corresponding probability
triplets. For example, the following script

.. testcode::

      from qibo import models, gates

      c = models.Circuit(2)
      c.add([gates.H(0), gates.H(1), gates.CNOT(0, 1)])

      # Define a noise map that maps qubit IDs to noise probabilities
      noise_map = {0: (0.1, 0.0, 0.2), 1: (0.0, 0.2, 0.1)}
      noisy_c = c.with_noise(noise_map)

will create a new circuit ``noisy_c`` that is equivalent to:

.. testcode::

      noisy_c2 = models.Circuit(2)
      noisy_c2.add(gates.H(0))
      noisy_c2.add(gates.PauliNoiseChannel(0, 0.1, 0.0, 0.2))
      noisy_c2.add(gates.H(1))
      noisy_c2.add(gates.PauliNoiseChannel(1, 0.0, 0.2, 0.1))
      noisy_c2.add(gates.CNOT(0, 1))
      noisy_c2.add(gates.PauliNoiseChannel(0, 0.1, 0.0, 0.2))
      noisy_c2.add(gates.PauliNoiseChannel(1, 0.0, 0.2, 0.1))

Note that ``noisy_c`` uses the gate objects of the original circuit ``c``
(it is not a deep copy), while in ``noisy_c2`` each gate was created as
a new object.

The user may use a single tuple instead of a dictionary as the noise map
In this case the same probabilities will be applied to all qubits.
That is ``noise_map = (0.1, 0.0, 0.1)`` is equivalent to
``noise_map = {0: (0.1, 0.0, 0.1), 1: (0.1, 0.0, 0.1), ...}``.

As described in the previous sections, if
:meth:`qibo.abstractions.circuit.AbstractCircuit.with_noise()` is used in a circuit
that uses state vectors then noise will be simulated with repeated execution.
If the user wishes to use density matrices instead, this is possible by
initializing a :class:`qibo.core.circuit.DensityMatrixCircuit`
using the ``density_matrix=True`` flag during initialization and call
``.with_noise`` on this circuit.

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

      from qibo import models, gates
      from qibo.noise import NoiseModel, PauliError

      # Build specific noise model with 2 quantum errors:
      # - Pauli error on H only for qubit 1.
      # - Pauli error on CNOT for all the qubits.
      noise = NoiseModel()
      noise.add(PauliError(px = 0.5), gates.H, 1)
      noise.add(PauliError(py = 0.5), gates.CNOT)

      # Generate noiseless circuit.
      c = models.Circuit(2)
      c.add([gates.H(0), gates.H(1), gates.CNOT(0, 1)])

      # Apply noise to the circuit according to the noise model.
      noisy_c = noise.apply(c)

The noisy circuit defined above will be equivalent to the following circuit:

.. testcode::

      noisy_c2 = models.Circuit(2)
      noisy_c2.add(gates.H(0))
      noisy_c2.add(gates.H(1))
      noisy_c2.add(gates.PauliNoiseChannel(1, px=0.5))
      noisy_c2.add(gates.CNOT(0, 1))
      noisy_c2.add(gates.PauliNoiseChannel(0, py=0.5))
      noisy_c2.add(gates.PauliNoiseChannel(1, py=0.5))


The :class:`qibo.noise.NoiseModel` class supports also density matrices,
it is sufficient to pass a circuit which was initialized with ``density_matrix=True``
to generate the correspoding :class:`qibo.core.circuit.DensityMatrixCircuit`.



.. _measurementbitflips-example:

Measurement errors
^^^^^^^^^^^^^^^^^^

:class:`qibo.abstractions.states.AbstractState` provides
the :meth:`qibo.abstractions.states.AbstractState.apply_bitflips` method which
allows adding bit-flip errors to the sampled bit-strings without having to
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

Note that :meth:`qibo.abstractions.states.AbstractState.apply_bitflips` modifies
the measurement samples contained in the corresponding state and therefore the
original noiseless measurement samples are no longer accessible. It is possible
to keep the original samples by creating a copy of the states before applying
the bitflips:

.. testsetup::

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

.. testcode::

      # create a copy of the state containing the noiseless samples
      noisy_result = result.copy()
      # perform bitflips in the copy
      noisy_result.apply_bitflips(0.2)

Creating a copy as shown in the above example does not duplicate the state
vector for memory efficiency reasons. All copies of the state point to the
same tensor in memory.

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
can be used both in :meth:`qibo.abstractions.states.AbstractState.apply_bitflips`
and the measurement gate. If ``p1`` is not specified the value of ``p0`` will
be used for both errors.


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
    # will print a ``tf.Tensor`` of shape ``(1001,)`` that holds <X>(t) values
.. testoutput::
    :hide:

    ...


Note that the time step ``dt=1e-3`` defines how often we calculate <X> during
the evolution.

In the above cases the exact time evolution operator (exponential of the Hamiltonian)
was used to evolve the state vector. Because the evolution Hamiltonian is
time-independent, the matrix exponentiation happens only once. It is possible to
simulate time-dependent Hamiltonians by passing a function of time instead of
a :class:`qibo.abstractions.hamiltonians.Hamiltonian` in the
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
:class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
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
contains :class:`qibo.abstractions.gates.Unitary` gates corresponding to the
exponentials of the Trotter decomposition and can be executed on any state.

Note that in the transverse field Ising model (TFIM) that was used in this
example is among the pre-coded Hamiltonians in Qibo and could be created as
a :class:`qibo.core.hamiltonians.SymbolicHamiltonian` simply using the
``dense=False`` flag. For more information on the difference between dense
and non-dense Hamiltonians we refer to the :ref:`Hamiltonians <Hamiltonians>`
section. Note that only non-dense Hamiltonians created using ``dense=False``
or through the :class:`qibo.core.hamiltonians.SymbolicHamiltonian` object
can be used for evolution using Trotter decomposition. If a dense Hamiltonian
is used then evolution will be done by exponentiating the full Hamiltonian
matrix.

Defining custom Hamiltonians from terms can be more complicated,
however Qibo simplifies this process by providing the option
to define Hamiltonians symbolically through the use of ``sympy``.
For more information on this we refer to the
:ref:`How to define custom Hamiltonians using symbols? <symbolicham-example>`
example.

A :class:`qibo.core.hamiltonians.SymbolicHamiltonian` can also be used to
simulate time evolution. This can be done by passing the Hamiltonian to a
:class:`qibo.evolution.StateEvolution` model and using the exponential solver.
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
:class:`qibo.evolution.StateEvolution` model. We refer to the
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
(:class:`qibo.abstractions.callbacks.Gap`) is available for calculating the
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
using as :class:`qibo.core.hamiltonians.SymbolicHamiltonian` objects. For
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
:class:`qibo.evolution.AdiabaticEvolution` model.

Note that ``h0`` and ``h1`` should have the same type, either both
:class:`qibo.core.hamiltonians.Hamiltonian` or both
:class:`qibo.core.hamiltonians.SymbolicHamiltonian`.


Optimizing the scheduling function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The free parameters ``p`` of the scheduling function can be optimized using
the :meth:`qibo.evolution.AdiabaticEvolution.minimize` method. The parameters
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
define Hamiltonians based on :class:`qibo.core.hamiltonians.Hamiltonian` which
uses the full matrix representation of the corresponding operator or
:class:`qibo.core.hamiltonians.SymbolicHamiltonian` which uses a more efficient
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
qubits. This makes the use of :class:`qibo.abstractions.hamiltonians.SymbolicHamiltonian`
preferrable as the qubit number increases, as this Hamiltonians is not based
in the full matrix representation.

To simplify the construction of Hamiltonians, Qibo provides the
:class:`qibo.abstractions.hamiltonians.SymbolicHamiltonian` object which
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
:class:`qibo.core.hamiltonians.SymbolicHamiltonian` is used for time evolution,
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
