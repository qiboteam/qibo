Noise
=====

In Qibo it is possible to create a custom noise model using the
class :class:`qibo.noise.NoiseModel`. This enables the user to create
circuits where the noise is gate and qubit dependent.

For more information on the use of :class:`qibo.noise.NoiseModel` see
:ref:`How to perform noisy simulation? <noisemodel-example>`

.. autoclass:: qibo.noise.NoiseModel
    :members:
    :member-order: bysource

Quantum errors
--------------

The quantum errors available to build a noise model are the following:

.. autoclass:: qibo.noise.KrausError
    :members:
    :member-order: bysource

.. autoclass:: qibo.noise.UnitaryError
    :members:
    :member-order: bysource

.. autoclass:: qibo.noise.PauliError
    :members:
    :member-order: bysource

.. autoclass:: qibo.noise.DepolarizingError
    :members:
    :member-order: bysource

.. autoclass:: qibo.noise.ThermalRelaxationError
    :members:
    :member-order: bysource

.. autoclass:: qibo.noise.AmplitudeDampingError
    :members:
    :member-order: bysource

.. autoclass:: qibo.noise.PhaseDampingError
    :members:
    :member-order: bysource

.. autoclass:: qibo.noise.ReadoutError
    :members:
    :member-order: bysource

.. autoclass:: qibo.noise.ResetError
    :members:
    :member-order: bysource

.. autoclass:: qibo.noise.CustomError
    :members:
    :member-order: bysource


IBMQ noise model
----------------

In Qibo, it is possible to build noisy circuits based on IBMQ's reported noise model of
for its quantum computer by using the :class:`qibo.noise.IBMQNoiseModel` class.
The noise model is built using a combination of the
:class:`qibo.gates.ThermalRelaxationChannel` and :class:`qibo.gates.DepolarizingChannel`
channels. . At the end of the circuit, if the qubit is measured,
bitflips errors are set. Moreover, the model handles idle qubits by applying a thermal
relaxation channel for the duration of the idle-time.

For more information on the :class:`qibo.noise.IBMQNoiseModel` class, see the
example on :ref:`Simulating quantum hardware <noise-hardware-example>`.


.. autoclass:: qibo.noise.IBMQNoiseModel
    :members:
    :member-order: bysource
