.. _Components:

Components
==========

The Qibo package comes with the following modules:

* Models_
* Gates_
* Hamiltonians_
* Callbacks_
* Optimizers_
* Parallel_
* Backends_

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
inherit the :class:`qibo.abstractions.circuit.AbstractCircuit` which implements basic
properties of the circuit, such as the list of gates and the number of qubits.

In order to perform calculations and apply gates to a state vector a backend
has to be used. The main ``Circuit`` used for simulation is defined in
:class:`qibo.core.circuit.Circuit`. This uses an abstract backend object ``K``
to perform calculation which can be one of the backends defined in ``qibo/backends``.

.. _generalpurpose:

Circuit models
^^^^^^^^^^^^^^

Abstract circuit
""""""""""""""""

.. autoclass:: qibo.abstractions.circuit.AbstractCircuit
    :members:
    :member-order: bysource

Circuit
"""""""

.. autoclass:: qibo.core.circuit.Circuit
    :members:
    :member-order: bysource


Circuit addition
""""""""""""""""

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

Circuit fusion
""""""""""""""

The gates contained in a circuit can be fused up to two-qubits using the
:meth:`qibo.abstractions.circuit.AbstractCircuit.fuse` method. This returns a new circuit
that contains :class:`qibo.abstractions.gates.Unitary` gates that are less in number
than the gates in the original circuit but have equivalent action.
For some circuits (such as variational), if the number of qubits is large it is
more efficient to execute the fused instead of the original circuit.

The fusion algorithm starts by creating a :class:`qibo.abstractions.fusion.FusionGroup`.
The first available gates in the circuit's gate queue are added in the group
until the two qubits of the group are identified. Any subsequent one-qubit gate
applied in one of these qubits or two-qubit gates applied to these two qubits
are added in the group. Gates that affect more than two qubits or target
different qubits are left for the next round of fusion. Once all compatible gates
are added in the group the fusion round finishes and a new ``FusionGroup`` is
created for the next round. The algorithm terminates once all gates are assigned
to a group.

A ``FusionGroup`` can either start with any one- or two-qubit gate
except ``CNOT``, ``CZ``, ``SWAP`` and ``CU1`` because it is more efficient
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
to a new :class:`qibo.abstractions.gates.Unitary` gate that is equivalent to applying
all the gates in the group.

.. autoclass:: qibo.core.fusion.FusionGroup
    :members:
    :member-order: bysource

Density matrix circuit
""""""""""""""""""""""

.. autoclass:: qibo.core.circuit.DensityMatrixCircuit
    :members:
    :member-order: bysource

Distributed circuit
"""""""""""""""""""

.. autoclass:: qibo.core.distcircuit.DistributedCircuit
    :members:
    :member-order: bysource

Quantum Fourier Transform (QFT)
"""""""""""""""""""""""""""""""

.. autoclass:: qibo.models.circuit.QFT
    :members:
    :member-order: bysource

Variational Quantum Eigensolver (VQE)
"""""""""""""""""""""""""""""""""""""

.. autoclass:: qibo.models.variational.VQE
    :members:
    :member-order: bysource

Quantum Approximate Optimization Algorithm (QAOA)
"""""""""""""""""""""""""""""""""""""""""""""""""

.. autoclass:: qibo.models.variational.QAOA
    :members:
    :member-order: bysource

Feedback-based Algorithm for Quantum Optimization (FALQON)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. autoclass:: qibo.models.variational.FALQON
    :members:
    :member-order: bysource

Grover's Algorithm
""""""""""""""""""

.. autoclass:: qibo.models.grover.Grover
    :members:
    :member-order: bysource


.. _timeevolution:

Time evolution
^^^^^^^^^^^^^^

State evolution
"""""""""""""""

.. autoclass:: qibo.models.evolution.StateEvolution
    :members:
    :member-order: bysource

Adiabatic evolution
"""""""""""""""""""

.. autoclass:: qibo.models.evolution.AdiabaticEvolution
    :members:
    :member-order: bysource

_______________________

.. _Gates:

Gates
-----

All supported gates can be accessed from the ``qibo.gates`` module and inherit
the base gate object :class:`qibo.abstractions.gates.Gate`. Read below for a complete
list of supported gates.

All gates support the ``controlled_by`` method that allows to control
the gate on an arbitrary number of qubits. For example

* ``gates.X(0).controlled_by(1, 2)`` is equivalent to ``gates.TOFFOLI(1, 2, 0)``,
* ``gates.RY(0, np.pi).controlled_by(1, 2, 3)`` applies the Y-rotation to qubit 0 when qubits 1, 2 and 3 are in the |111> state.
* ``gates.SWAP(0, 1).controlled_by(3, 4)`` swaps qubits 0 and 1 when qubits 3 and 4 are in the |11> state.

Gate models
^^^^^^^^^^^

Abstract gates
""""""""""""""

.. autoclass:: qibo.abstractions.gates.Gate
    :members:
    :member-order: bysource

Abstract backend gates
""""""""""""""""""""""

.. autoclass:: qibo.abstractions.abstract_gates.BaseBackendGate
    :members:
    :member-order: bysource

Single qubit gates
^^^^^^^^^^^^^^^^^^

Hadamard (H)
""""""""""""

.. autoclass:: qibo.abstractions.gates.H
   :members:
   :member-order: bysource

Pauli X (X)
"""""""""""

.. autoclass:: qibo.abstractions.gates.X
   :members:
   :member-order: bysource

Pauli Y (Y)
"""""""""""

.. autoclass:: qibo.abstractions.gates.Y
    :members:
    :member-order: bysource

Pauli Z (Z)
"""""""""""

.. autoclass:: qibo.abstractions.gates.Z
    :members:
    :member-order: bysource

Identity (I)
""""""""""""

.. autoclass:: qibo.abstractions.gates.I
    :members:
    :member-order: bysource

Measurement (M)
"""""""""""""""

.. autoclass:: qibo.abstractions.gates.M
    :members:
    :member-order: bysource

Rotation X-axis (RX)
""""""""""""""""""""

.. autoclass:: qibo.abstractions.gates.RX
    :members:
    :member-order: bysource

Rotation Y-axis (RY)
""""""""""""""""""""

.. autoclass:: qibo.abstractions.gates.RY
    :members:
    :member-order: bysource

Rotation Z-axis (RZ)
""""""""""""""""""""

.. autoclass:: qibo.abstractions.gates.RZ
    :members:
    :member-order: bysource

First general unitary (U1)
""""""""""""""""""""""""""

.. autoclass:: qibo.abstractions.gates.U1
    :members:
    :member-order: bysource

Second general unitary (U2)
"""""""""""""""""""""""""""

.. autoclass:: qibo.abstractions.gates.U2
    :members:
    :member-order: bysource

Third general unitary (U3)
""""""""""""""""""""""""""

.. autoclass:: qibo.abstractions.gates.U3
    :members:
    :member-order: bysource

Two qubit gates
^^^^^^^^^^^^^^^

Controlled-NOT (CNOT)
"""""""""""""""""""""

.. autoclass:: qibo.abstractions.gates.CNOT
    :members:
    :member-order: bysource

Controlled-phase (CZ)
"""""""""""""""""""""

.. autoclass:: qibo.abstractions.gates.CZ
    :members:
    :member-order: bysource

Controlled-rotation X-axis (CRX)
""""""""""""""""""""""""""""""""

.. autoclass:: qibo.abstractions.gates.CRX
    :members:
    :member-order: bysource

Controlled-rotation Y-axis (CRY)
""""""""""""""""""""""""""""""""

.. autoclass:: qibo.abstractions.gates.CRY
    :members:
    :member-order: bysource

Controlled-rotation Z-axis (CRZ)
""""""""""""""""""""""""""""""""

.. autoclass:: qibo.abstractions.gates.CRZ
    :members:
    :member-order: bysource

Controlled first general unitary (CU1)
""""""""""""""""""""""""""""""""""""""

.. autoclass:: qibo.abstractions.gates.CU1
    :members:
    :member-order: bysource

Controlled second general unitary (CU2)
"""""""""""""""""""""""""""""""""""""""

.. autoclass:: qibo.abstractions.gates.CU2
    :members:
    :member-order: bysource

Controlled third general unitary (CU3)
""""""""""""""""""""""""""""""""""""""

.. autoclass:: qibo.abstractions.gates.CU3
    :members:
    :member-order: bysource

Controlled third general unitary (CU3)
""""""""""""""""""""""""""""""""""""""

.. autoclass:: qibo.abstractions.gates.CU3
    :members:
    :member-order: bysource

Swap (SWAP)
"""""""""""

.. autoclass:: qibo.abstractions.gates.SWAP
    :members:
    :member-order: bysource

fSim
""""

.. autoclass:: qibo.abstractions.gates.fSim
    :members:
    :member-order: bysource

fSim with general rotation
""""""""""""""""""""""""""

.. autoclass:: qibo.abstractions.gates.GeneralizedfSim
    :members:
    :member-order: bysource


Special gates
^^^^^^^^^^^^^

Toffoli
"""""""

.. autoclass:: qibo.abstractions.gates.TOFFOLI
    :members:
    :member-order: bysource

Arbitrary unitary
"""""""""""""""""

.. autoclass:: qibo.abstractions.gates.Unitary
    :members:
    :member-order: bysource

Variational layer
"""""""""""""""""

.. autoclass:: qibo.abstractions.gates.VariationalLayer
    :members:
    :member-order: bysource

Flatten
"""""""

.. autoclass:: qibo.abstractions.gates.Flatten
    :members:
    :member-order: bysource

Callback gate
"""""""""""""

.. autoclass:: qibo.abstractions.gates.CallbackGate
    :members:
    :member-order: bysource

_______________________

.. _Channels:

Channels
--------

Channels are implemented in Qibo as additional gates and can be accessed from
the ``qibo.gates`` module. Channels can be used on density matrices to perform
noisy simulations. Channels that inherit :class:`qibo.abstractions.gates.UnitaryChannel`
can also be applied to state vectors using sampling and repeated execution.
For more information on the use of channels to simulate noise we refer to
:ref:`How to perform noisy simulation? <noisy-example>`
The following channels are currently implemented:

Partial trace
^^^^^^^^^^^^^

.. autoclass:: qibo.abstractions.gates.PartialTrace
    :members:
    :member-order: bysource

Kraus channel
^^^^^^^^^^^^^

.. autoclass:: qibo.abstractions.gates.KrausChannel
    :members:
    :member-order: bysource

Unitary channel
^^^^^^^^^^^^^^^

.. autoclass:: qibo.abstractions.gates.UnitaryChannel
    :members:
    :member-order: bysource

Pauli noise channel
^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.abstractions.gates.PauliNoiseChannel
    :members:
    :member-order: bysource

Reset channel
^^^^^^^^^^^^^

.. autoclass:: qibo.abstractions.gates.ResetChannel
    :members:
    :member-order: bysource

Thermal relaxation channel
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.abstractions.gates.ThermalRelaxationChannel
    :members:
    :member-order: bysource

_______________________


.. _Hamiltonians:

Hamiltonians
------------

The main abstract Hamiltonian object of Qibo is:

.. autoclass:: qibo.abstractions.hamiltonians.AbstractHamiltonian
    :members:
    :member-order: bysource


Matrix Hamiltonian
^^^^^^^^^^^^^^^^^^

The first implementation of Hamiltonians uses the full matrix representation
of the Hamiltonian operator in the computational basis. This matrix has size
``(2 ** nqubits, 2 ** nqubits)`` and therefore its construction is feasible
only when number of qubits is small.

.. autoclass:: qibo.abstractions.hamiltonians.MatrixHamiltonian
    :members:
    :member-order: bysource

.. autoclass:: qibo.core.hamiltonians.Hamiltonian
    :members:
    :member-order: bysource


Symbolic Hamiltonian
^^^^^^^^^^^^^^^^^^^^

Qibo allows the user to define Hamiltonians using ``sympy`` symbols. In this
case the full Hamiltonian matrix is not constructed unless this is required.
This makes the implementation more efficient for larger qubit numbers.
For more information on constructing Hamiltonians using symbols we refer to the
:ref:`How to define custom Hamiltonians using symbols? <symbolicham-example>` example.

.. autoclass:: qibo.abstractions.hamiltonians.SymbolicHamiltonian
    :members:
    :member-order: bysource

.. autoclass:: qibo.core.hamiltonians.SymbolicHamiltonian
    :members:
    :member-order: bysource


When a :class:`qibo.core.hamiltonians.SymbolicHamiltonian` is used for time
evolution then Qibo will automatically perform this evolution using the Trotter
of the evolution operator. This is done by automatically splitting the Hamiltonian
to sums of commuting terms, following the description of Sec. 4.1 of
`arXiv:1901.05824 <https://arxiv.org/abs/1901.05824>`_.
For more information on time evolution we refer to the
:ref:`How to simulate time evolution? <timeevol-example>` example.

In addition to the abstract Hamiltonian models, Qibo provides the following
pre-coded Hamiltonians:


Heisenberg XXZ
^^^^^^^^^^^^^^

.. autoclass:: qibo.hamiltonians.XXZ
    :members:
    :member-order: bysource

Non-interacting Pauli-X
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.hamiltonians.X
    :members:
    :member-order: bysource

Non-interacting Pauli-Y
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.hamiltonians.Y
    :members:
    :member-order: bysource

Non-interacting Pauli-Z
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.hamiltonians.Z
    :members:
    :member-order: bysource

Transverse field Ising model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.hamiltonians.TFIM
    :members:
    :member-order: bysource

Max Cut
^^^^^^^

.. autoclass:: qibo.hamiltonians.MaxCut
    :members:
    :member-order: bysource


.. note::
    All pre-coded Hamiltonians can be created as
    :class:`qibo.core.hamiltonians.Hamiltonian` using ``dense=True``
    or :class:`qibo.core.hamiltonians.SymbolicHamiltonian`
    using the ``dense=False``. In the first case the Hamiltonian is created
    using its full matrix representation of size ``(2 ** n, 2 ** n)``
    where ``n`` is the number of qubits that the Hamiltonian acts on. This
    matrix is used to calculate expectation values by direct matrix multiplication
    to the state and for time evolution by exact exponentiation.
    In contrast, when ``dense=False`` the Hamiltonian contains a more compact
    representation as a sum of local terms. This compact representation can be
    used to calculate expectation values via a sum of the local term expectations
    and time evolution via the Trotter decomposition of the evolution operator.
    This is useful for systems that contain many qubits for which constructing
    the full matrix is intractable.

_______________________


.. _Symbols:

Symbols
-------

Qibo provides a basic set of symbols which inherit the ``sympy.Symbol`` object
and can be used to construct :class:`qibo.abstractions.hamiltonians.SymbolicHamiltonian`
objects as described in the previous section.

.. autoclass:: qibo.symbols.Symbol
    :members:
    :member-order: bysource

.. autoclass:: qibo.symbols.X
    :members:
    :member-order: bysource

.. autoclass:: qibo.symbols.Y
    :members:
    :member-order: bysource

.. autoclass:: qibo.symbols.Z
    :members:
    :member-order: bysource

_______________________


.. _States:

States
------

Qibo circuits return :class:`qibo.abstractions.states.AbstractState` objects
when executed. By default, Qibo works as a wave function simulator in the sense
that propagates the state vector through the circuit applying the
corresponding gates. In this default usage the result of a circuit execution
is the full final state vector which can be accessed via the ``tensor``
property of states.
However for specific applications it is useful to have measurement samples
from the final wave function, instead of its full vector form.
To that end, :class:`qibo.abstractions.states.AbstractState` provides the
:meth:`qibo.abstractions.states.AbstractState.samples` and
:meth:`qibo.abstractions.states.AbstractState.frequencies` methods.

In order to perform measurements the user has to add the measurement gate
:class:`qibo.core.gates.M` to the circuit and then execute providing a number
of shots. If this is done, the :class:`qibo.abstractions.states.AbstractState`
returned by the circuit will contain the measurement samples.

For more information on measurements we refer to the
:ref:`How to perform measurements? <measurement-examples>` example.

Abstract state
^^^^^^^^^^^^^^

.. autoclass:: qibo.abstractions.states.AbstractState
    :members:
    :member-order: bysource

Distributed state
^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.core.states.DistributedState
    :members:
    :member-order: bysource


.. _Callbacks:

Callbacks
---------

Callbacks provide a way to calculate quantities on the state vector as it
propagates through the circuit. Example of such quantity is the entanglement
entropy, which is currently the only callback implemented in
:class:`qibo.abstractions.callbacks.EntanglementEntropy`.
The user can create custom callbacks by inheriting the
:class:`qibo.abstractions.callbacks.Callback` class. The point each callback is
calculated inside the circuit is defined by adding a :class:`qibo.abstractions.gates.CallbackGate`.
This can be added similarly to a standard gate and does not affect the state vector.

.. autoclass:: qibo.abstractions.callbacks.Callback
   :members:
   :member-order: bysource

Entanglement entropy
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.abstractions.callbacks.EntanglementEntropy
   :members:
   :member-order: bysource

Norm
^^^^

.. autoclass:: qibo.abstractions.callbacks.Norm
   :members:
   :member-order: bysource

Overlap
^^^^^^^

.. autoclass:: qibo.abstractions.callbacks.Overlap
    :members:
    :member-order: bysource

Energy
^^^^^^

.. autoclass:: qibo.abstractions.callbacks.Energy
    :members:
    :member-order: bysource

Gap
^^^

.. autoclass:: qibo.abstractions.callbacks.Gap
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
   :exclude-members: ParallelBFGS

.. _Parallel:

Parallelism
-----------

We provide CPU multi-processing methods for circuit evaluation for multiple
input states and multiple parameters for fixed input state.

When using the methods below the ``processes`` option controls the number of
processes used by the parallel algorithms through the ``multiprocessing``
library. By default ``processes=None``, in this case the total number of logical
cores are used. Make sure to select the appropriate number of processes for your
computer specification, taking in consideration memory and physical cores. In
order to obtain optimal results you can control the number of threads used by
each process with the ``qibo.set_threads`` method. For example, for small-medium
size circuits you may benefit from single thread per process, thus set
``qibo.set_threads(1)`` before running the optimization.

.. automodule:: qibo.parallel
   :members:
   :member-order: bysource
   :exclude-members: ParallelResources

.. _Backends:

Backends
--------

The main calculation engine is defined in the abstract backend object
:class:`qibo.backends.abstract.AbstractBackend`. This object defines the methods
required by all Qibo models to perform simulation.

Qibo currently provides two different calculation backends, one based on
numpy and one based on Tensorflow. It is possible to define new backends by
inheriting :class:`qibo.backends.abstract.AbstractBackend` and implementing
its abstract methods.

Both backends are supplemented by custom operators defined under which can be
used to efficiently apply gates to state vectors or density matrices.
These custom operators are shipped as the separate libraries qibojit and qibotf.
We refer to :ref:`Packages <packages>` section for a complete list of the
available computation backends and instructions on how to install each of
these libraries on top of qibo.

Custom operators are much faster than implementations based on numpy or Tensorflow
primitives (such as ``einsum``) but do not support some features, such as
automatic differentiation for backpropagation of variational circuits which is
only supported by the native ``tensorflow`` backend.

The user can switch backends using

.. code-block::  python

    import qibo
    qibo.set_backend("qibotf")
    qibo.set_backend("numpy")

before creating any circuits or gates. The default backend is the first available
from ``qibojit``, ``qibotf``, ``tensorflow``, ``numpy``.

For developers, we provide a configuration file in
``qibo/backends/profiles.yml`` containing the technical specifications for all
backends supported by the Qibo team. If you are planning to introduce a new
backend module for simulation or hardware, you can simply edit this profile file
and include the reference to your new module. Alternatively, you can set a
custom profile file by storing the file path in the ``QIBO_PROFILE`` environment
variable before executing the code.

.. autoclass:: qibo.backends.abstract.AbstractBackend
    :members:
    :member-order: bysource
