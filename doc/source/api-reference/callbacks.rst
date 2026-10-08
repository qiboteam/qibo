.. _Callbacks:

Callbacks
=========

Callbacks provide a way to calculate quantities on the state vector as it
propagates through the circuit. Example of such quantity is the entanglement
entropy, which is currently the only callback implemented in
:class:`qibo.callbacks.EntanglementEntropy`.
The user can create custom callbacks by inheriting the
:class:`qibo.callbacks.Callback` class. The point each callback is
calculated inside the circuit is defined by adding a :class:`qibo.gates.CallbackGate`.
This can be added similarly to a standard gate and does not affect the state vector.

.. autoclass:: qibo.callbacks.Callback
   :members:
   :member-order: bysource

Entanglement entropy
--------------------

.. autoclass:: qibo.callbacks.EntanglementEntropy
   :members:
   :member-order: bysource

Norm
----

.. autoclass:: qibo.callbacks.Norm
   :members:
   :member-order: bysource

Overlap
-------

.. autoclass:: qibo.callbacks.Overlap
    :members:
    :member-order: bysource

Energy
------

.. autoclass:: qibo.callbacks.Energy
    :members:
    :member-order: bysource

Gap
---

.. autoclass:: qibo.callbacks.Gap
    :members:
    :member-order: bysource
