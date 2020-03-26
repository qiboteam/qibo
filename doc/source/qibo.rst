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

Qibo provides the following models:

.. automodule:: qibo.models
   :members:
   :member-order: bysource

_______________________

.. _Gates:

Gates
-----

The following gates can be accessed as attributes of the gates module:

   - Basic one qubit gates: ``H``, ``X``, ``Y``, ``Z``, ``Iden``. Take as argument the index of the qubit they act on.
   - Parametrized one qubit rotations: ``RX``, ``RY``, ``RZ``. Take as argument the index of the qubit they act on and the value of the parameter theta.
   - Two qubit gates: ``CNOT``, ``SWAP``, ``CRZ``. Take as argument the indices of the two qubits and the theta parameter for ``CRZ``. For controlled gates, the first qubit given is the control and the second is the target.
   - Three qubit gate: ``TOFFOLI``. The first two qubits are controls and the third qubit is the target.
   - Arbitrary unitary gate: ``Unitary``. It takes as input a matrix (numpy array or Tensorflow tensor) and the target qubit ids. For example ``Unitary(np.array([[0, 1], [1, 0]]), 0)`` is equivalent to ``X(0)``. This gate can act to arbitrary number of qubits. There is no check that the given matrix is unitary.
   - The ``Flatten`` gate can be used to input a specific state vector. It takes as input a list/array of the amplitudes.

All gates support the ``controlled_by`` that allows to control them on an arbitrary number of qubits. For example

   - ``gates.X(0).controlled_by(1, 2)`` is equivalent to ``gates.TOFFOLI(1, 2, 0)``,
   - ``gates.RY(0, np.pi).controlled_by(1, 2, 3)`` applies the Y-rotation to qubit 0 when qubits 1, 2 and 3 are in the |111> state.
   - ``gates.SWAP(0, 1).controlled_by(3, 4)`` swaps qubits 0 and 1 when qubits 3 and 4 are in the |11> state.

``controlled_by`` cannot be used on gates that are already controlled.

All gate implementations are based on the ``Gate`` class.

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