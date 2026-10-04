.. _Channels:

Channels
========

Channels are implemented in Qibo as additional gates and can be accessed from
the ``qibo.gates`` module. Channels can be used on density matrices to perform
noisy simulations. Channels that inherit :class:`qibo.gates.UnitaryChannel`
can also be applied to state vectors using sampling and repeated execution.
For more information on the use of channels to simulate noise we refer to
:ref:`How to perform noisy simulation? <noisy-example>`
The following channels are currently implemented:

Kraus channel
-------------

.. autoclass:: qibo.gates.KrausChannel
    :members:
    :member-order: bysource

Unitary channel
---------------

.. autoclass:: qibo.gates.UnitaryChannel
    :members:
    :member-order: bysource


Pauli noise channel
-------------------

.. autoclass:: qibo.gates.PauliNoiseChannel
    :members:
    :member-order: bysource

Depolarizing channel
--------------------

.. autoclass:: qibo.gates.DepolarizingChannel
    :members:
    :member-order: bysource

Thermal relaxation channel
--------------------------

.. autoclass:: qibo.gates.ThermalRelaxationChannel
    :members:
    :member-order: bysource

Amplitude damping channel
-------------------------

.. autoclass:: qibo.gates.AmplitudeDampingChannel
    :members:
    :member-order: bysource

Phase damping channel
-------------------------

.. autoclass:: qibo.gates.PhaseDampingChannel
    :members:
    :member-order: bysource

Readout error channel
---------------------

.. autoclass:: qibo.gates.ReadoutErrorChannel
    :members:
    :member-order: bysource

Reset channel
-------------

.. autoclass:: qibo.gates.ResetChannel
    :members:
    :member-order: bysource
