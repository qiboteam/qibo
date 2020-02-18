Components
==========

The Qibo package comes with the following modules:

* Models_ (:code:`qibo.models`)
* Gates_ (:code:`qibo.gates`)
* Backends_ (:code:`qibo.backends`)
* Running_ (:code:`qibo.run`)

These modules provide all the required components to ...

_______________________

.. _Models:

Models
------

Qibo provides a ``Circuit`` class which holds the gates and
measurements in a common format for all backends.

.. autoclass:: qibo.models.Circuit
   :members:
   :inherited-members:
   :member-order: bysource

_______________________

.. _Gates:

Gates
-----

Qibo provides the following quantum gates:

* CNOT: the controlled-not gate.
* H: the Hadamard gate.
* X, Y, Z: the Pauli X/Y/Z gate.
* Barrier: the barrier gate.
* S: the swap gate.
* T: the toffoli gate.
* Iden: the identity gate.
* MX, MY, MZ: measures X/Y/Z gate.
* RX, RY, RZ: rotation X/Y/Z-axis.

All gate implementations are based on the ``Gate`` class.

.. automodule:: qibo.gates
   :members:
   :member-order: bysource

_______________________

.. _Backends:

Backends
--------

All backends inherits from the ``Backend`` abstract class:

.. automodule:: qibo.backends.common
   :members:
   :inherited-members:
   :member-order: bysource


_______________________

.. _Running:

Running the code
----------------

In order to perform the computation Qibo provides the ``run()`` function
presented below:

.. autofunction:: qibo.run.run

