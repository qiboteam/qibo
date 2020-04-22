Components
==========

The QIBO package comes with the following modules:

* Models_
* Gates_
* Hamiltonians_
* Callbacks_

These modules provide all the required components to ...

_______________________

.. _Models:

Models
------

QIBO provides both :ref:`generalpurpose` and :ref:`applicationspecific`.

The general purpose model is called ``Circuit`` and holds the list of gates
that are applied to the state vector or density matrix. All ``Circuit`` models
inherit the :class:`qibo.base.circuit.BaseCircuit` which implements basic
properties of the circuit, such as the list of gates and the number of qubits.

In order to perform calculations and apply gates to a state vector a backend
has to be used. Our current backend of choice is `Tensorflow <http://tensorflow.org/>`_
and the corresponding ``Circuit`` model is :class:`qibo.tensorflow.circuit.TensorflowCircuit`.

Currently there are two application specific models implemented,
the Quantum Fourier Transform (:class:`qibo.models.QFT`) and
the Variational Quantum Eigensolver (:class:`qibo.models.VQE`).

.. _generalpurpose:

General purpose models
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.base.circuit.BaseCircuit
    :members:
    :member-order: bysource
.. autoclass:: qibo.tensorflow.circuit.TensorflowCircuit
    :members:
    :member-order: bysource


.. _applicationspecific:

Application specific models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: qibo.models
    :members:
    :member-order: bysource


.. _circuitaddition:

Circuit addition
^^^^^^^^^^^^^^^^

``Circuit`` objects also support addition. For example

.. code-block::  python

    from qibo import models
    from qibo import gates

    c1 = models.QFT(4)

    c2 = models.Circuit(4)
    c2.add(gates.RZ(0, 0.1234))
    c2.add(gates.RZ(1, 0.1234))
    c2.add(gates.RZ(2, 0.1234))
    c2.add(gates.RZ(3, 0.1234))

    c = c1 + c2

will create a circuit that performs the Quantum Fourier Transform on four qubits
followed by Rotation-Z gates.


_______________________

.. _Gates:

Gates
-----

All supported gates can be accessed from the `qibo.gates` module and inherit
the base gate object :class:`qibo.base.gates.Gate`. Read bellow for a complete
list of supported gates.

All gates support the ``controlled_by`` method that allows to control
the gate on an arbitrary number of qubits. For example

* ``gates.X(0).controlled_by(1, 2)`` is equivalent to ``gates.TOFFOLI(1, 2, 0)``,
* ``gates.RY(0, np.pi).controlled_by(1, 2, 3)`` applies the Y-rotation to qubit 0 when qubits 1, 2 and 3 are in the |111> state.
* ``gates.SWAP(0, 1).controlled_by(3, 4)`` swaps qubits 0 and 1 when qubits 3 and 4 are in the |11> state.

.. automodule:: qibo.base.gates
   :members:
   :member-order: bysource

_______________________

.. _Hamiltonians:

Hamiltonians
------------

We provide the following hamiltonians:

.. automodule:: qibo.hamiltonians
   :members:
   :member-order: bysource

_______________________

.. _Measurements:

Measurements
------------

QIBO is a wave function simulator in the sense that propagates the state vector
through the circuit applying the corresponding gates. In the default usage the
result of executing a circuit is the full final state vector. However for
specific applications it is useful to have measurement samples from the final
wave function, instead of its full vector form.
:class:`qibo.base.measurements.CircuitResult` provides a basic API for this.

In order to execute measurements the user has to add the measurement gate
:class:`qibo.base.gates.M` to the circuit and then execute providing a number
of shots. This will return a :class:`qibo.base.measurements.CircuitResult`
object that is described bellow.

For more information on measurements we refer to the related examples.

.. autoclass:: qibo.base.measurements.CircuitResult
    :members:
    :member-order: bysource


.. _Callbacks:

Callbacks
------------

Callbacks provide a way to calculate quantities on the state vector as it
propagates through the circuit. Example of such quantity is the entanglement
entropy, which is currently the only callback implemented in
:class:`qibo.tensorflow.callbacks.EntanglementEntropy`.
The user can create custom callbacks by inheriting the
:class:`qibo.tensorflow.callbacks.Callback` class.

.. automodule:: qibo.tensorflow.callbacks
   :members:
   :member-order: bysource


.. _Einsum:

Einsum Backends
---------------

.. automodule:: qibo.tensorflow.einsum
   :members:
   :member-order: bysource
