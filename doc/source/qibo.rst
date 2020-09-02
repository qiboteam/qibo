Components
==========

The Qibo package comes with the following modules:

* Models_
* Gates_
* Hamiltonians_
* Callbacks_

_______________________

.. _Models:

Models
------

Qibo provides models for both the circuit based and the adiabatic quantum
computation paradigms. Circuit based models include :ref:`generalpurpose` which
allow defining arbitrary circuits and :ref:`applicationspecific` such as the
Quantum Fourier Transform (:class:`qibo.models.QFT`) and the
Variational Quantum Eigensolver (:class:`qibo.models.VQE`).
Adiabatic quantum computation is simulated using the :ref:`timeevolution`
of state vectors.

The general purpose model is called ``Circuit`` and holds the list of gates
that are applied to the state vector or density matrix. All ``Circuit`` models
inherit the :class:`qibo.base.circuit.BaseCircuit` which implements basic
properties of the circuit, such as the list of gates and the number of qubits.

In order to perform calculations and apply gates to a state vector a backend
has to be used. Our current backend of choice is `Tensorflow <http://tensorflow.org/>`_
and the corresponding ``Circuit`` model is :class:`qibo.tensorflow.circuit.TensorflowCircuit`.

.. _generalpurpose:

General circuit models
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.base.circuit.BaseCircuit
    :members:
    :member-order: bysource
.. autoclass:: qibo.tensorflow.circuit.TensorflowCircuit
    :members:
    :member-order: bysource
.. autoclass:: qibo.tensorflow.distcircuit.TensorflowDistributedCircuit
    :members:
    :member-order: bysource
.. autoclass:: qibo.tensorflow.distutils.DistributedState
    :members:
    :member-order: bysource


.. _applicationspecific:

Application specific models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: qibo.models
    :members:
    :member-order: bysource
    :exclude-members: K, Norm


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


.. _circuitfusion:

Circuit fusion
^^^^^^^^^^^^^^

The gates contained in a circuit can be fused up to two-qubits using the
:meth:`qibo.base.circuit.BaseCircuit.fuse` method. This returns a new circuit
that contains :class:`qibo.base.gates.Unitary` gates that are less in number
than the gates in the original circuit but have equivalent action.
For some circuits (such as variational), if the number of qubits is large it is
more efficient to execute the fused instead of the original circuit.

The fusion algorithm starts by creating a :class:`qibo.base.fusion.FusionGroup`.
The first available gates in the circuit's gate queue are added in the group
until the two qubits of the group are identified. Any subsequent one-qubit gate
applied in one of these qubits or two-qubit gates applied to these two qubits
are added in the group. Gates that affect more than two qubits or target
different qubits are left for the next round of fusion. Once all compatible gates
are added in the group the fusion round finishes and a new ``FusionGroup`` is
created for the next round. The algorithm terminates once all gates are assigned
to a group.

A ``FusionGroup`` can either start with any one- or two-qubit gate
except ``CNOT``, ``CZ``, ``SWAP`` and ``CZPow`` because it is more efficient
to apply such gates on their own rather than fusing them with others. These gates
are fused only when "sandwiched" between one-qubit gates. For example

.. code-block::  python

    c.add([gates.H(0), gates.H(1)])
    c.add(gates.CZ(0, 1))
    c.add([gates.X(0), gates.Y(1)])

will be fused to a single ``Unitary(0, 1)`` gate, while

.. code-block::  python

    c.add([gates.H(0), gates.H(1)])
    c.add(gates.CZ(0, 1))

will remain as it is.

Once groups are identified, all gates belonging to a ``FusionGroup`` are fused
by multiplying their respective unitary matrices. This way each group results
to a new :class:`qibo.base.gates.Unitary` gate that is equivalent to applying
all the gates in the group.

.. autoclass:: qibo.base.fusion.FusionGroup
    :members:
    :member-order: bysource



.. _timeevolution:

Time evolution
^^^^^^^^^^^^^^

.. autoclass:: qibo.evolution.StateEvolution
    :members:
    :member-order: bysource
.. autoclass:: qibo.evolution.AdiabaticEvolution
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

The main abstract Hamiltonian object of Qibo is:

.. autoclass:: qibo.base.hamiltonians.Hamiltonian
    :members:
    :member-order: bysource


Qibo provides an additional object that represents Hamiltonians without using
their full matrix representation and can be used for time evolution using the
Trotter decomposition. The Hamiltonians represented by this object are sums of
commuting terms, following the description of Sec. 4.1 of
`arXiv:1901.05824 <https://arxiv.org/abs/1901.05824>`_.

.. autoclass:: qibo.base.hamiltonians.TrotterHamiltonian
    :members:
    :member-order: bysource


In addition to these abstract models, Qibo provides the following pre-coded
Hamiltonians:

.. automodule:: qibo.hamiltonians
   :members:
   :member-order: bysource


Note that all pre-coded Hamiltonians can be created as either
:class:`qibo.base.hamiltonians.Hamiltonian` or
:class:`qibo.base.hamiltonians.TrotterHamiltonian` using the ``trotter`` flag.


_______________________

.. _Measurements:

Measurements
------------

Qibo is a wave function simulator in the sense that propagates the state vector
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
:class:`qibo.tensorflow.callbacks.Callback` class. The point each callback is
calculated inside the circuit is defined by adding a :class:`qibo.base.gates.CallbackGate`.
This can be added similarly to a standard gate and does not affect the state vector.

.. automodule:: qibo.tensorflow.callbacks
   :members:
   :member-order: bysource


.. _Solvers:

Solvers
-------

Solvers are used to numerically calculate the time evolution of state vectors.
They perform steps in time by integrating the time-dependent Schrodinger
equation.

.. automodule:: qibo.solvers
   :members:
   :member-order: bysource


.. _Optimizers:

Optimizers
----------

Optimizers are used automatically by the ``minimize`` methods of
:class:`qibo.models.VQE` and :class:`qibo.evolution.AdiabaticEvolution` models.
The user does not have to use any of the optimizer methods included in the
current section, however the required options of each optimization method
can be passed when calling the ``minimize`` method of the respective Qibo
variational model.

.. automodule:: qibo.optimizers
   :members:
   :member-order: bysource

.. _Backends:

Backends
--------

Qibo currently uses two different backends for applying gates to vectors.
The default backend uses custom Tensorflow operators defined under
``tensorflow/custom_operators`` to apply gates to state vectors. These
operators are much faster than implementations based on Tensorflow.
Currently custom operators do not support the following:

* Density matrices, channels and noise.
* Automatic differentiation for backpropagation of variational circuits.

It is possible to use these features in Qibo by using a backend that uses
Tensorflow primitives. There are two such backends available: the ``"defaulteinsum"``
backend based on ``tf.einsum`` and the ``"matmuleinsum"`` backend based on ``tf.matmul``.
The user can switch backends using

.. code-block::  python

    import qibo
    qibo.set_backend("matmuleinsum")

before creating any circuits or gates. The default backend is ``"custom"`` and
uses the custom Tensorflow operators.
