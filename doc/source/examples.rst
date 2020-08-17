Basic examples
==============

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
    final_state = c(initial_state) # c.execute(initial_state) also works
    print(final_state.numpy())
    # should print `np.array([1, 0, 0, 0])`

If you are planning to freeze the circuit and just query for different initial
states then you can use the ``Circuit.compile`` method which will improve
evaluation performance, e.g.:

.. code-block:: python

    import numpy as np
    # switch backend to "matmuleinsum" or "defaulteinsum"
    # (slower than default "custom" backend)
    import qibo
    qibo.set_backend("matmuleinsum")
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

Note that compiling is only supported when native tensorflow gates are used.
This happens when the calculation backend is switched to ``"matmuleinsum"``
or ``"defaulteinsum"``. This backend is much slower than the default ``"custom"``
backend which uses custom tensorflow operators to apply gates.

How to print a circuit summary?
-------------------------------

It is possible to print a summary of the circuit using ``circuit.summary()``.
This will print basic information about the circuit, including its depth, the
total number of qubits and all gates in order of the number of times they appear.
The QASM name is used as identifier of gates.
For example

.. code-block:: python

    c = Circuit(3)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.CNOT(0, 2))
    c.add(gates.CNOT(1, 2))
    c.add(gates.H(2))
    c.add(gates.TOFFOLI(0, 1, 2))
    print(c.summary())
    # Prints
    '''
    Circuit depth = 7
    Number of qubits = 3
    Most common gates:
    h: 3
    cx: 2
    ccx: 1
    '''

The circuit property ``circuit.gate_types`` will also return a ``collections.Counter``
that contains the gate types and the corresponding numbers of appearance. The
method ``circuit.gates_of_type`` can be used to access gate objects of specific type.
For example for the circuit of the previous example:

.. code-block:: python

    common_gates = c.gate_types.most_common()
    # returns the list [("h", 3), ("cx", 2), ("ccx", 1)]

    most_common_gate = common_gates[0][0]
    # returns "h"

    all_h_gates = c.gates_of_type("h")
    # returns the list [(0, ref to H(0)), (1, ref to H(1)), (4, H(2))]

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
will return ``collections.Counter({"1001": 100})``.

It is possible to define registers of multiple qubits by either passing
the qubit ids seperately, such as ``gates.M(0, 1, 2, 4)``, or using the ``*``
operator: ``gates.M(*[0, 1, 2, 4])``. The ``*`` operator is useful if qubit
ids are saved in an iterable. For example ``gates.M(*range(5))`` is equivalent
to ``gates.M(0, 1, 2, 3, 4)``.

Unmeasured qubits are ignored by the measurement objects. Also, the
order that qubits appear in the results is defined by the order the user added
the measurements and not the qubit ids.


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
  * For small circuits (less than 10 qubits) the overhead from casting tensors
    on GPU may be larger than executing the circuit on CPU, making CPU execution
    preferrable. In such cases disabling CPU multi-threading may also increase
    performance (see next subsection).
  * A standard GPU has 12-16GB of memory and thus can simulate up to 30 qubits on
    single-precision or 29 qubits with double-precision when Qibo's default gates
    are used. For larger circuits one should either use the CPU (assuming it has
    more memory) or a distributed circuit configuration. The latter allows splitting
    the state vector on multiple devices and is useful both when multiple GPUs
    are available in the system or even for re-using a single GPU
    (see relevant subsection bellow).

Note that if the used device runs out of memory during a circuit execution an error will be
raised prompting the user to switch the default device using ``qibo.set_device``.

Setting the number of CPU threads
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Qibo inherits Tensorflow's defaults for CPU thread configuration and in most cases
will utilize all available CPU threads. For small circuits the parallelization
overhead may decrease performance making single thread execution preferrable.
Tensorflow allows restricting the number of threads as follows:

.. code-block::  python

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

.. code-block::  python

    from qibo.models import Circuit
    from qibo import gates

    # Define GPU configuration
    accelerators = {"/GPU:0": 3, "/GPU:1": 1}
    # this will use the first GPU three times and the second one time
    # leading to four total logical devices
    # construct the distributed circuit for 32 qubits
    c = Circuit(32, accelerators, memory_device="/CPU:0")

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
code: :class:`qibo.tensorflow.distcircuit.TensorflowDistributedCircuit`. When
``models.Circuit`` is called then this distributed implementation is used automatically
if the ``accelerators`` dictionary is passed, otherwise the standard single device
:class:`qibo.tensorflow.circuit.TensorflowCircuit` is used.

Unlike the standard circuit, executing a
:class:`qibo.tensorflow.distcircuit.TensorflowDistributedCircuit` without
measurements will return a
:class:`qibo.tensorflow.distutils.DistributedState` instead of the final
state vector as a ``tf.Tensor``. This is done because the distributed circuit
uses the state partitioned in multiple pieces that are distributed to the
different devices. Creating the full state as a tensor would require merging
these pieces and using twice as much memory. This is disabled by default,
however the user may create the full state as follows:

.. code-block::  python

    # Create distributed circuits for two GPUs
    c = Circuit(32, {"/GPU:0": 1, "/GPU:1": 1})
    # Add gates
    c.add(...)
    # Execute (``final_state`` will be a ``DistributedState``)
    final_state = c()

    # Access the full state (will double memory usage)
    full_final_state = final_state.vector
    # ``full_final_state`` is a ``tf.Tensor``

    # ``DistributedState`` supports indexing and slicing
    print(final_state[40])
    # will print the 40th component of the final state vector
    print(final_state[20:25])
    # will print the components from 20 to 24 (inclusive)


How to use callbacks?
---------------------

Callbacks allow the user to apply additional functions on the state vector
during circuit execution. An example use case of this is the calculation of
entanglement entropy as the state propagates through a circuit. This can be
implemented easily using :class:`qibo.tensorflow.callbacks.EntanglementEntropy`
and the :class:`qibo.base.gates.CallbackGate` gate. For example:

.. code-block::  python

    from qibo import models, gates, callbacks
    # initialize circuit with 2 qubits and add gates
    c = models.Circuit(2) # state is |00> (entropy = 0)
    c.add(gates.CallbackGate(entropy)) # performs entropy calculation in the initial state
    c.add(gates.H(0)) # state is |+0> (entropy = 0)
    c.add(gates.CallbackGate(entropy)) # performs entropy calculation after H
    c.add(gates.CNOT(0, 1)) # state is |00> + |11> (entropy = 1))
    c.add(gates.CallbackGate(entropy)) # performs entropy calculation after CNOT

    # create entropy callback where qubit 0 is the first subsystem
    entropy = callbacks.EntanglementEntropy([0])
    # execute the circuit using the callback
    final_state = c()

The results can be accessed using indexing on the callback objects. In this
example ``entropy[:]`` will return ``tf.Tensor([0, 0, 1])`` which are the
values of entropy after every gate in the circuit.

The same callback object can be used in a second execution of this or a different
circuit. For example

.. code-block::  python

    # c is the same circuit as above
    # execute the circuit
    final_state = c()
    # execute the circuit a second time
    final_state = c()

    # print result
    print(entropy[:]) # tf.Tensor([0, 0, 1, 0, 0, 1])

The callback for entanglement entropy can also be used on state vectors directly.
For example

.. code-block::  python

    import numpy as np
    # create a singlet state vector
    state = np.zeros(4)
    state[0], state[3] = 1 / np.sqrt(2), 1 / np.sqrt(2)

    # create an `EntanglementEntropy` callback object
    entropy = callbacks.EntanglementEntropy([0])
    # call the object on the state
    print(entropy(state))

will print ``tf.Tensor(1.0)``.

.. _params-examples:
How to use parametrized gates?
------------------------------

Some Qibo gates such as rotations accept values for their free parameter. Once
such gates are added in a circuit their parameters can be updated using the
:meth:`qibo.base.circuit.BaseCircuit.set_parameters` method. For example:

.. code-block::  python

    from qibo.models import Circuit
    from qibo import gates
    # create a circuit with all parameters set to 0.
    c = Circuit(3, accelerators)
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

.. code-block::  python

    c = Circuit(3, accelerators)
    g0 = gates.RX(0, theta=0)
    g1 = gates.RY(1, theta=0)
    g2 = gates.fSim(0, 2, theta=0, phi=0)
    c.add([g0, g1, gates.CZ(1, 2), g2, gates.H(2)])

    # set new values to the circuit's parameters using a dictionary
    params = {g0: 0.123, g1: 0.456, g2: (0.789, 0.321)]
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

* ``RX``, ``RY``, ``RZ``, ``ZPow``, ``CZPow``: Accept a single ``theta`` parameter.
* :class:`qibo.base.gates.fSim`: Accepts a tuple of two parameters ``(theta, phi)``.
* :class:`qibo.base.gates.GeneralizedfSim`: Accepts a tuple of two parameters
  ``(unitary, phi)``. Here ``unitary`` should be a unitary matrix given as an
  array or ``tf.Tensor`` of shape ``(2, 2)``.
* :class:`qibo.base.gates.Unitary`: Accepts a single ``unitary`` parameter. This
  should be an array or ``tf.Tensor`` of shape ``(2, 2)``.
* :class:`qibo.base.gates.VariationalLayer`: Accepts a list of ``float``
  parameters with length compatible to the number of one qubit rotations implemented
  by the layer, for example:

.. code-block:: python

    c = Circuit(5)
    pairs = list((i, i + 1) for i in range(0, 4, 2))
    c.add(gates.VariationalLayer(range(nqubits), pairs,
                                 gates.RY, gates.CZ,
                                 params=np.zeros(5)))
    c.add((gates.RX(i, theta=0) for i in range(5)))

    # set random parameters to all rotations in the circuit
    c.set_parameters(np.random.random(10))
    # note that 10 numbers are used as the VariationalLayer contains five
    # rotations and five additional RX rotations are added afterwards.

Note that a ``np.ndarray`` or a ``tf.Tensor`` may also be used in the place of
a flat list.

Using :meth:`qibo.base.circuit.BaseCircuit.set_parameters` is more efficient than
recreating a new circuit with new parameter values.


How to write a Quantum Fourier Transform?
-----------------------------------------

A simple Quantum Fourier Transform (QFT) example to test your installation:

.. code-block:: python

    from qibo.models import QFT

    # Create a QFT circuit with 15 qubits
    circuit = QFT(15)

    # Simulate final state wavefunction default initial state is |00>
    final_state = c()


Please note that the ``QFT()`` function is simply a shorthand for the circuit
construction. For number of qubits higher than 30, the QFT can be distributed to
multiple GPUs using ``QFT(31, accelerators)``. Further details are presented in
the section :ref:`How to select hardware devices? <gpu-examples>`.


.. vqe-example:

How to write a VQE?
-------------------

The VQE requires an ansatz function and a ``Hamiltonian`` object. There are examples of VQE optimization in ``examples/benchmarks``:

    - ``vqe.py``: a simple example with the XXZ model.

Here is a simple example using the Heisenberg XXZ model Hamiltonian:

.. code-block:: python

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
    return circuit

    # Create XXZ Hamiltonian
    hamiltonian = hamiltonians.XXZ(nqubits=nqubits)
    # Create VQE model
    vqe = models.VQE(circuit, hamiltonian)

    # Optimize starting from a random guess for the variational parameters
    initial_parameters = np.random.uniform(0, 2*np.pi,
                                            2*nqubits*nlayers + nqubits)
    best, params = vqe.minimize(initial_parameters, method='BFGS')

The user can choose one of the following methods for minimization:

    - ``"cma"``: Genetic optimizer,
    - ``"sgd"``: Gradient descent using Tensorflow's automatic differentiation and built-in `Adagrad <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adagrad>`_ optimizer,
    - All methods supported by `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.

Note that if ``"sgd"`` is used then the user has to use a backend based on
tensorflow primitives and not the default custom backend because custom operators
currently do not support automatic differentiation. To switch the backend one
can do ``qibo.set_backend("matmuleinsum")``.
Check the next example on automatic differentiation for more details.

A useful gate for defining the ansatz of the VQE is :class:`qibo.base.gates.VariationalLayer`.
This optimizes performance by fusing the layer of one-qubit parametrized gates with
the layer of two-qubit entangling gates and applying both as a single layer of
general two-qubit gates (as 4x4 matrices). The ansatz from the above example can
be written using :class:`qibo.base.gates.VariationalLayer` as follows:

.. code-block:: python

    circuit = models.Circuit(nqubits)
    pairs = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    theta = np.zeros(nqubits)
    for l in range(nlayers):
        circuit.add(gates.VariationalLayer(range(nqubits), pairs,
                                           gates.RY, gates.CZ,
                                           theta, theta))
        circuit.add((gates.CZ(i, i + 1) for i in range(1, nqubits - 2, 2)))
        circuit.add(gates.CZ(0, nqubits - 1))
    circuit.add((gates.RY(i, theta) for i in range(nqubits)))
    return circuit


How to use automatic differentiation?
-------------------------------------

As a deep learning framework, Tensorflow supports
`automatic differentiation <https://www.tensorflow.org/tutorials/customization/autodiff>`_.
This can be used to optimize the parameters of variational circuits. For example
the following script optimizes the parameters of two rotations so that the circuit
output matches a target state, using the fidelity as figure of merit.

.. code-block:: python

    import tensorflow as tf
    # switch backend to "matmuleinsum" or "defaulteinsum"
    import qibo
    qibo.set_backend("matmuleinsum")
    from qibo.models import Circuit
    from qibo import gates

    nepochs = 100
    params = tf.Variable(np.zeros(2), dtype=tf.float64)
    optimizer = tf.keras.optimizers.Adam()
    target_state = tf.ones(4, dtype=tf.complex128) / 2.0

    for _ in range(nepochs):
        with tf.GradientTape() as tape:
            c = Circuit(2)
            c.add(RX(0, params[0]))
            c.add(RY(0, params[1]))
            fidelity = tf.math.real(tf.reduce_sum(tf.math.conj(target_state) * c()))
            loss = 1 - fidelity

        grads = tape.gradient(loss, params)
        optimizer.apply_gradients(zip(grads, params))


Note that the circuit has to be defined inside the ``tf.GradientTape()`` otherwise
the calculated gradients will be ``None``. Also, a backend that uses tensorflow
primitives gates (either ``"matmuleinsum"`` or ``"defaulteinsum"``) has to be
used because currently the default ``"custom"`` backend does not support automatic
differentiation.

The optimization procedure can also be compiled as follows:

.. code-block:: python

    nepochs = 100
    params = tf.Variable(np.zeros(2), dtype=tf.float64)
    optimizer = tf.keras.optimizers.Adam()
    target_state = tf.ones(4, dtype=tf.complex128) / 2.0

    @tf.function
    def optimize(params):
        with tf.GradientTape() as tape:
            c = Circuit(2)
            c.add(RX(0, params[0]))
            c.add(RY(0, params[1]))
            fidelity = tf.math.real(tf.reduce_sum(tf.math.conj(target_state) * c()))
            loss = 1 - fidelity

        grads = tape.gradient(loss, params)
        optimizer.apply_gradients(zip(grads, params))

    for _ in range(nepochs):
        optimize(params)

The user may also use ``tf.Variable`` and parametrized gates in any other way
that is supported by Tensorflow, such as defining
`custom Keras layers <https://www.tensorflow.org/guide/keras/custom_layers_and_models>`_
and using the `Sequential model API <https://www.tensorflow.org/api_docs/python/tf/keras/Sequential>`_
to train them.


How to perform noisy simulation?
--------------------------------

Qibo can perform noisy simulation using density matrices. ``Circuit`` objects can
evolve density matrices in a similar manner to state vectors. In order to use
density matrices the user should execute the circuit passing a density matrix as
the initial state. For example

.. code-block:: python

    import qibo
    # switch backend to "matmuleinsum" or "defaulteinsum"
    qibo.set_backend("matmuleinsum")
    from qibo import models, gates

    # Define circuit
    c = models.Circuit(2)
    c.add(gates.H(0))
    c.add(gates.H(1))

    # Define initial density matrix as `rho = |00><00|`
    state = np.zeros(4)
    state[0] = 1
    initial_rho = np.outer(state, state.conj())

    # Call circuit on the density matrix
    final_rho = c(initial_rho)
    # final_rho will be tf.eye(4) / 4 which corresponds to |++><++|

will perform the transformation

.. math::
    |00 \rangle \langle 00| \rightarrow (H_1 \otimes H_2)|00 \rangle \langle 00|(H_1 \otimes H_2)^\dagger = |++ \rangle \langle ++|

Note that the calculation backend was switched to ``"matmuleinsum"`` because the
default ``"custom"`` backend does not support density matrices.

The user can simulate noise using :class:`qibo.base.gates.NoiseChannel`.
If this or any other channel is used in a ``Circuit``, then the execution will automatically
switch to density matrices. For example

.. code-block:: python

    from qibo import models, gates

    c = models.Circuit(2) # starts with state |00>
    c.add(gates.X(1)) # transforms |00> -> |01>
    c.add(gates.NoiseChannel(0, px=0.3)) # transforms |01> -> (1 - px)|01><01| + px |11><11|
    final_state = c()
    # will return tf.Tensor(diag([0, 0.7, 0, 0.3]))

will perform the transformation

.. math::
    |00\rangle & \rightarrow (I \otimes X)|00\rangle = |01\rangle
    \\& \rightarrow 0.7|01\rangle \langle 01| + 0.3(X\otimes I)|01\rangle \langle 01|(X\otimes I)^\dagger
    \\& = 0.7|01\rangle \langle 01| + 0.3|11\rangle \langle 11|

Note that ``Circuit`` will use state vectors until the first channel is found and will
switch to density matrices for the rest of the simulation. Measurements and
callbacks can be used exactly as in the pure state vector case.

In practical applications noise typically occurs after every gate. For this reason,
:class:`qibo.base.circuit.BaseCircuit` provides a ``.with_noise()`` method
which automatically creates a new circuit that contains a noise channel after
every normal gate. The user can control the probabilities of the noise channel
using a noise map, which is a dictionary that maps qubits to the corresponding
noise probability triplets.

For example, the following script

.. code-block:: python

      from qibo.models import Circuit
      from qibo import gates

      c = Circuit(2)
      c.add([gates.H(0), gates.H(1), gates.CNOT(0, 1)])

      # Define a noise map that maps qubit IDs to noise probabilities
      noise_map = {0: (0.1, 0.0, 0.2), 1: (0.0, 0.2, 0.1)}
      noisy_c = c.with_noise(noise_map)

will create a new circuit ``noisy_c`` that is equivalent to:

.. code-block:: python

      noisy_c2 = Circuit(2)
      noisy_c2.add(gates.H(0))
      noisy_c2.add(gates.NoiseChannel(0, 0.1, 0.0, 0.2))
      noisy_c2.add(gates.NoiseChannel(1, 0.0, 0.2, 0.1))
      noisy_c2.add(gates.H(1))
      noisy_c2.add(gates.NoiseChannel(0, 0.1, 0.0, 0.2))
      noisy_c2.add(gates.NoiseChannel(1, 0.0, 0.2, 0.1))
      noisy_c2.add(gates.CNOT(0, 1))
      noisy_c2.add(gates.NoiseChannel(0, 0.1, 0.0, 0.2))
      noisy_c2.add(gates.NoiseChannel(1, 0.0, 0.2, 0.1))

Note however that the circuit ``noisy_c`` that was created using the
``with_noise`` method uses the gate objects of the original circuit ``c``
(it is not a deep copy), unlike ``noisy_c2`` where each gate was created as
a new object.

The user may use a single tuple instead of a dictionary as the noise map
In this case the same probabilities will be applied to all qubits.
That is ``noise_map = (0.1, 0.0, 0.1)`` is equivalent to
``noise_map = {0: (0.1, 0.0, 0.1), 1: (0.1, 0.0, 0.1), ...}``.

Moreover, ``with_noise`` supports an additional optional argument ``measurement_noise``
which allows the user to explicitly specify the noise probabilities.
before measurement gates. These may be different from the typical noise probabilities
depending on the experimental realization of measurements. For example:

.. code-block:: python

      from qibo.models import Circuit
      from qibo import gates

      c = Circuit(2)
      c.add([gates.H(0), gates.H(1)])
      c.add(gates.M(0))

      # Define a noise map that maps qubit IDs to noise probabilities
      noise_map = {0: (0.1, 0.0, 0.2), 1: (0.0, 0.2, 0.1)}
      measurement_noise = (0.4, 0.0, 0.0)
      noisy_c = c.with_noise(noise_map, measurement_noise=measurement_noise)

is equivalent to the following:

.. code-block:: python

      noisy_c = Circuit(2)
      noisy_c.add(gates.H(0))
      noisy_c.add(gates.NoiseChannel(0, 0.1, 0.0, 0.2))
      noisy_c.add(gates.NoiseChannel(1, 0.0, 0.2, 0.1))
      noisy_c.add(gates.H(1))
      noisy_c.add(gates.NoiseChannel(0, 0.4, 0.0, 0.0))
      noisy_c.add(gates.NoiseChannel(1, 0.0, 0.2, 0.1))
      noisy_c.add(gates.M(0))

Note that ``measurement_noise`` does not affect qubits that are not measured
and the default ``noise_map`` will be used for those.

Similarly to ``noise_map``, ``measurement_noise`` can either be either a
dictionary that maps each qubit to the corresponding probability triplet or
a tuple if the same triplet shall be used on all measured qubits.


How to simulate time evolution?
-------------------------------

Simulating the unitary time evolution of quantum states is useful in many
physics applications including the simulation of adiabatic quantum computation.
Qibo provides the :class:`qibo.models.StateEvolution` model that simulates
unitary evolution using the full state vector. For example:

.. code-block::  python

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
vector but also observing how physical quantities change during the time
evolution. This is possible using callbacks. For example, in the above case we
can track how <X> changes as follows:

.. code-block::  python

    from qibo import callbacks
    # Define a callback that calculates the energy (expectation value) of the X Hamiltonian
    observable = callbacks.Energy(hamiltonians.X(nqubits))
    # Create evolution object using the above callback and a time step of dt=1e-3
    evolve = models.StateEvolution(hamiltonians.Z(nqubits), dt=1e-3,
                                   callbacks=[observable])
    # Evolve for total time t=1
    final_state = evolve(final_time=1, initial_state=initial_state)

    print(observable[:])
    # will print a ``tf.Tensor`` of shape ``(1001,)`` that holds <X>(t) values


Note that the time step ``dt=1e-3`` defines how often we calculate <X> during
the evolution.

In the above cases the exact time evolution operator (exponential of the Hamiltonian)
was used to evolve the state vector. Because the evolution Hamiltonian is
time-independent, the matrix exponentiation happens only once. It is possible to
simulate time-dependent Hamiltonians by passing a function of time instead of
a :class:`qibo.hamiltonians.Hamiltonian` in the
:class:`qibo.models.StateEvolution` model. For example:

.. code-block::  python

    import numpy as np
    from qibo import hamiltonians, models

    # Defina a time dependent Hamiltonian
    nqubits = 4
    ham = lambda t: np.cos(t) * hamiltonians.Z(nqubits)
    # and pass it to the evolution model
    evolve = models.StateEvolution(ham, dt=1e-3)
    final_state = evolve(final_time=1, initial_state=initial_state)


The above script will still use the exact time evolution operator with the
exponentiation repeated for each time step. The integration method can
be changed using the ``solver`` argument when executing. Currently the default
exponential solver (``"exp"``) and a fourth-order Runge-Kutta solver (``"rk4"``)
are implemented.


How to simulate adiabatic time evolution?
-----------------------------------------

Qibo provides the :class:`qibo.models.AdiabaticEvolution` model to simulate
adiabatic time evolution. This is a special case of the
:class:`qibo.models.StateEvolution` model analyzed in the previous example
where the evolution Hamiltonian is interpolated between an initial "easy"
Hamiltonian and a "hard" Hamiltonian that usually solves an optimization problem.
Here is an example of adiabatic evolution simulation:

.. code-block::  python

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
    evolve = models.AdiabaticEvolution(h0, h1, s, solver="rk4")
    # Evolve using the Runge-Kutta solver to get the final state
    final_state = evolve(final_time=T)


If the initial state is not specified the ground state of the easy Hamiltonian
will be used, as it is common for adiabatic evolution. For proper scheduling
and total evolution time the ``final_state`` should approximate the ground state
of the "hard" Hamiltonian. Callbacks may also be used as in the previous example.

The scheduling function ``s`` should be a callable that accepts one (s(t)) or
two (s(t, p)) arguments. The first argument accepts values in [0, 1] and
corresponds to the ratio ``t / final_time`` during evolution. The second
optional argument is a vector of free parameters that can be optimized. The
function should, by definition, satisfy the properties s(0, p) = 0 and
s(1, p) = 1 for any p, otherwise errors will be raised.

The free parameters ``p`` of the scheduling function can be optimized using
the :meth:`qibo.evolution.AdiabaticEvolution.minimize` method. The parameters
are optimized so that the final state of the adiabatic evolution approximates
the ground state of the "hard" Hamiltonian. Optimization is similar to what is
described in the :ref:`How to write a VQE? <vqe-example>` example and can be
done as follows:

.. code-block::  python

    # Define Hamiltonians
    h0 = hamiltonians.X(3)
    h1 = hamiltonians.TFIM(3)
    # Define scheduling function with a free variational parameter ``p``
    sp = lambda t, p: (1 - p) * np.sqrt(t) + p * t
    # Define an evolution model with dt=1e-2
    evolution = models.AdiabaticEvolution(h0, h1, sp, dt=1e-2)
    # Find the optimal value for ``p`` starting from ``p = 0.5`` and ``T=1``.
    initial_guess = [0.5, 1]
    best, params = evolution.minimize(initial_guess, method="BFGS", options={'disp': True})
    print(best) # prints the best energy <H1> found from the final state
    print(params) # prints the optimal values for the parameters.

Note that the ``minimize`` method optimizes both the free parameters ``p`` of
the scheduling function as well as the total evolution time. The initial guess
for the total evolution time should be the last value of the given
``initial_guess`` array.


How to modify the simulation precision?
---------------------------------------

By default the simulation is performed in ``double`` precision (``complex128``).
We provide the ``qibo.set_precision`` function to modify the default behaviour.
Note that `qibo.set_precision` must be called before allocating circuits:

.. code-block:: python

        import qibo
        qibo.set_precision("single") # enables complex64
        # or
        qibo.set_precision("double") # re-enables complex128

        # ... continue with circuit creation and execution
