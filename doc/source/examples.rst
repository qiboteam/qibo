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

If you are planning to freeze the circuit and just query for different initial states then you can use the ``Circuit.compile`` method which will improve the evaluation performance, e.g.:

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
will return ``collections.Counter({"1001": 100})``.

Note that unmeasured qubits are ignored by the measurement objects. Also, the
order that qubits appear in the results is defined by the order the user added
the measurements and not the qubit ids.


How to use callbacks?
-----------------------------------

Callbacks allow the user to apply additional functions on the state vector
during circuit execution. An example use case of this is the calculation of
entanglement entropy as the state propagates through a circuit. This can be
implemented easily using :class:`qibo.tensorflow.callbacks.EntanglementEntropy`
as follows:

.. code-block::  python

    from qibo import models, gates, callbacks
    # initialize circuit with 2 qubits and add gates
    c = models.Circuit(2) # state is |00> (entropy = 0)
    c.add(gates.H(0)) # state is |+0> (entropy = 0)
    c.add(gates.CNOT(0, 1)) # state is |00> + |11> (entropy = 1))

    # create entropy callback where qubit 0 is the first subsystem
    entropy = callbacks.EntanglementEntropy([0])
    # execute the circuit using the callback
    final_state = c(callback=entropy)

The results can be accessed using indexing on the callback objects. In this
example ``entropy[0]`` will return ``tf.Tensor([0, 0, 1])`` which are the
values of entropy after every gate in the circuit.

The same callback object can be used in a second execution of this or a different
circuit. For example

.. code-block::  python

    # c is the same circuit as above
    entropy = callbacks.EntanglementEntropy([0])
    # execute the circuit using the callback
    final_state = c(callback=entropy)
    # execute the circuit again using the same callback
    final_state = c(callback=entropy)

    # print result of first execution
    print(entropy[0]) # tf.Tensor([0, 0, 1])
    # print result of second execution
    print(entropy[1]) # tf.Tensor([0, 0, 1])
    # print result of all executions
    print(entropy[:]) # tf.Tensor([[0, 0, 1], [0, 0, 1]])

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
        return c

    hamiltonian = XXZ(nqubits=nqubits)
    initial_parameters = np.random.uniform(0, 2*np.pi,
                                            2*nqubits*layers + nqubits)
    v = VQE(ansatz, hamiltonian)
    best, params = v.minimize(initial_parameters, method='BFGS')

The user can choose one of the following methods for minimization:

    - ``"cma"``: Genetic optimizer,
    - ``"sgd"``: Gradient descent using Tensorflow's automatic differentiation and built-in `Adagrad <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adagrad>`_ optimizer,
    - All methods supported by `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.

If ``"sgd"`` is used then the user has to switch to the :class:`qibo.tensorflow.einsum.MatmulEinsum`
backend when defining the ansatz in order to get correct gradients,
because of an issue with the automatic differentiation of ``tf.einsum``.
This can be done easily by calling ``.with_backend("MatmulEinsum")`` on each
gate when defining the ansatz.
Check the next example on automatic differentiation for more details.


How to use automatic differentiation?
-------------------------------------

As a deep learning framework, Tensorflow supports
`automatic differentiation <https://www.tensorflow.org/tutorials/customization/autodiff>`_.
This can be used to optimize the parameters of variational circuits. For example
the following script optimizes the parameters of two rotations so that the circuit
output matches a target state, using the fidelity as figure of merit.

.. code-block:: python

    import tensorflow as tf

    nepochs = 100
    params = tf.Variable(np.zeros(2), dtype=tf.float64)
    optimizer = tf.keras.optimizers.Adam()
    target_state = tf.ones(4, dtype=tf.complex128) / 2.0

    for _ in range(nepochs):
        with tf.GradientTape() as tape:
            c = Circuit(2)
            c.add(RX(0, params[0]).with_backend("MatmulEinsum"))
            c.add(RY(0, params[1]).with_backend("MatmulEinsum"))
            fidelity = tf.math.real(tf.reduce_sum(tf.math.conj(target_state) * c()))
            loss = 1 - fidelity

        grads = tape.gradient(loss, params)
        optimizer.apply_gradients(zip(grads, params))


Note that the circuit has to be defined inside the ``tf.GradientTape()`` otherwise
the calculated gradients will be ``None``. Also, our custom einsum backend
:class:`qibo.tensorflow.einsum.MatmulEinsum` has to be used for gates that
contain parameters with respect to which we calculate gradients.
This is because the gradients of the original ``tf.einsum`` do not work properly
with complex numbers
(see `related issue <https://github.com/tensorflow/tensorflow/issues/37307>`_).

The optimization procedure can also be compiled as follows:

.. code-block:: python

    import tensorflow as tf

    nepochs = 100
    params = tf.Variable(np.zeros(2), dtype=tf.float64)
    optimizer = tf.keras.optimizers.Adam()
    target_state = tf.ones(4, dtype=tf.complex128) / 2.0

    @tf.function
    def optimize(params):
        with tf.GradientTape() as tape:
            c = Circuit(2)
            c.add(RX(0, params[0]).with_backend("MatmulEinsum"))
            c.add(RY(0, params[1]).with_backend("MatmulEinsum"))
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

QIBO can perform noisy simulation using density matrices. ``Circuit`` objects can
evolve density matrices in a similar manner to state vectors. In order to use
density matrices the user should execute the circuit passing a density matrix as
the initial state. For example

.. code-block:: python

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
:class:`qibo.tensorflow.TensorflowCircuit` provides a `.with_noise()` method
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

will create a new circuit ``noisy_c`` that is equivalent to running the
following:

.. code-block:: python

      noisy_c = Circuit(2)
      noisy_c.add(gates.H(0))
      noisy_c.add(gates.NoiseChannel(0, 0.1, 0.0, 0.2))
      noisy_c.add(gates.NoiseChannel(1, 0.0, 0.2, 0.1))
      noisy_c.add(gates.H(1))
      noisy_c.add(gates.NoiseChannel(0, 0.1, 0.0, 0.2))
      noisy_c.add(gates.NoiseChannel(1, 0.0, 0.2, 0.1))
      noisy_c.add(gates.CNOT(0, 1))
      noisy_c.add(gates.NoiseChannel(0, 0.1, 0.0, 0.2))
      noisy_c.add(gates.NoiseChannel(1, 0.0, 0.2, 0.1))


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
