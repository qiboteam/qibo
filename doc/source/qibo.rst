Components
==========

The Qibo package comes with the following modules:

* Models_
* Gates_
* Hamiltonians_

These modules provide all the required components to ...

_______________________

.. _Models:

Models
------

Qibo provides both :ref:`generalpurpose` and :ref:`applicationspecific`.

The general purpose model is called `Circuit` and holds the list of gates
that are applied to the state vector. All `Circuit` models inherit the
:class:`qibo.base.circuit.BaseCircuit` which implements basic properties of the
circuit, such as the list of gates and the number of qubits.

In order to perform calculations and apply gates to a state vector a backend
has to be used. Our current backend of choice is `Tensorflow <http://tensorflow.org/>`_
and the corresponding `Circuit` model is :class:`qibo.tensorflow.circuit.TensorflowCircuit`.

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

_______________________

.. _Gates:

Gates
-----

All supported gates can be accessed from the `qibo.gates` module and inherit
the base gate object :class:`qibo.base.gates.Gate`. Read bellow for a complete
list of supported gates.

All gates support the ``controlled_by`` method that allows to control
the gate on an arbitrary number of qubits.
   - ``gates.X(0).controlled_by(1, 2)`` is equivalent to ``gates.TOFFOLI(1, 2, 0)``,
   - ``gates.RY(0, np.pi).controlled_by(1, 2, 3)`` applies the Y-rotation to qubit 0 when qubits 1, 2 and 3 are in the |111> state.
   - ``gates.SWAP(0, 1).controlled_by(3, 4)`` swaps qubits 0 and 1 when qubits 3 and 4 are in the |11> state.

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
