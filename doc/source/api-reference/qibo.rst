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

In order to perform calculations and apply gates to a state vector a backend
has to be used. The backends are defined in ``qibo/backends``.
Circuit and gate objects are backend independent and can be executed with
any of the available backends.

Qibo uses big-endian byte order, which means that the most significant qubit
is the one with index 0, while the least significant qubit is the one with
the highest index.

.. _generalpurpose:

Circuit models
^^^^^^^^^^^^^^

Circuit
"""""""

.. autoclass:: qibo.models.circuit.Circuit
    :members:
    :member-order: bysource


Circuit addition
""""""""""""""""

:class:`qibo.models.circuit.Circuit` objects support addition. For example

.. testsetup::

    import qibo
    from qibo import models
    from qibo import gates

.. testcode::

    c1 = models.QFT(4)

    c2 = models.Circuit(4)
    c2.add(gates.RZ(0, 0.1234))
    c2.add(gates.RZ(1, 0.1234))
    c2.add(gates.RZ(2, 0.1234))
    c2.add(gates.RZ(3, 0.1234))

    c = c1 + c2

will create a circuit that performs the Quantum Fourier Transform on four qubits
followed by Rotation-Z gates.


.. _circuit-fusion:

Circuit fusion
""""""""""""""

The gates contained in a circuit can be fused up to two-qubits using the
:meth:`qibo.models.circuit.Circuit.fuse` method. This returns a new circuit
for which the total number of gates is less than the gates in the original
circuit as groups of gates have been fused to a single
:class:`qibo.gates.special.FusedGate` gate. Simulating the new circuit
is equivalent to simulating the original one but in most cases more efficient
since less gates need to be applied to the state vector.

The fusion algorithm works as follows: First all gates in the circuit are
transformed to unmarked :class:`qibo.gates.special.FusedGate`. The gates
are then processed in the order they were added in the circuit. For each gate
we identify the neighbors forth and back in time and attempt to fuse them to
the gate. Two gates can be fused if their total number of target qubits is
smaller than the fusion maximum qubits (specified by the user) and there are
no other gates between acting on the same target qubits. Gates that are fused
to others are marked. The new circuit queue contains the gates that remain
unmarked after the above operations finish.

Gates are processed in the original order given by user. There are no
additional simplifications performed such as commuting gates acting on the same
qubit or canceling gates even when such simplifications are mathematically possible.
The user can specify the maximum number of qubits in a fused gate using
the ``max_qubits`` flag in :meth:`qibo.models.circuit.Circuit.fuse`.

For example the following:

.. testcode::

    from qibo import models, gates

    c = models.Circuit(2)
    c.add([gates.H(0), gates.H(1)])
    c.add(gates.CZ(0, 1))
    c.add([gates.X(0), gates.Y(1)])
    fused_c = c.fuse()

will create a new circuit with a single :class:`qibo.gates.special.FusedGate`
acting on ``(0, 1)``, while the following:

.. testcode::

    from qibo import models, gates

    c = models.Circuit(3)
    c.add([gates.H(0), gates.H(1), gates.H(2)])
    c.add(gates.CZ(0, 1))
    c.add([gates.X(0), gates.Y(1), gates.Z(2)])
    c.add(gates.CNOT(1, 2))
    c.add([gates.H(0), gates.H(1), gates.H(2)])
    fused_c = c.fuse()

will give a circuit with two fused gates, the first of which will act on
``(0, 1)`` corresponding to

.. code-block::  python

    [H(0), H(1), CZ(0, 1), X(0), H(0)]

and the second will act to ``(1, 2)`` corresponding to

.. code-block::  python

    [Y(1), Z(2), CNOT(1, 2), H(1), H(2)]

.. _applicationspecific:

Quantum Fourier Transform (QFT)
"""""""""""""""""""""""""""""""

.. autoclass:: qibo.models.qft.QFT
    :members:
    :member-order: bysource

Variational Quantum Eigensolver (VQE)
"""""""""""""""""""""""""""""""""""""

.. autoclass:: qibo.models.variational.VQE
    :members:
    :member-order: bysource

Adiabatically Assisted Variational Quantum Eigensolver (AAVQE)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. autoclass:: qibo.models.variational.AAVQE
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


Style-based Quantum Generative Adversarial Network (style-qGAN)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. autoclass:: qibo.models.qgan.StyleQGAN
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

All supported gates can be accessed from the ``qibo.gates`` module.
Read below for a complete list of supported gates.

All gates support the ``controlled_by`` method that allows to control
the gate on an arbitrary number of qubits. For example

* ``gates.X(0).controlled_by(1, 2)`` is equivalent to ``gates.TOFFOLI(1, 2, 0)``,
* ``gates.RY(0, np.pi).controlled_by(1, 2, 3)`` applies the Y-rotation to qubit 0 when qubits 1, 2 and 3 are in the ``|111>`` state.
* ``gates.SWAP(0, 1).controlled_by(3, 4)`` swaps qubits 0 and 1 when qubits 3 and 4 are in the ``|11>`` state.

Abstract gate
^^^^^^^^^^^^^

.. autoclass:: qibo.gates.abstract.Gate
    :members:
    :member-order: bysource

Single qubit gates
^^^^^^^^^^^^^^^^^^

Hadamard (H)
""""""""""""

.. autoclass:: qibo.gates.H
   :members:
   :member-order: bysource

Pauli X (X)
"""""""""""

.. autoclass:: qibo.gates.X
   :members:
   :member-order: bysource

Pauli Y (Y)
"""""""""""

.. autoclass:: qibo.gates.Y
    :members:
    :member-order: bysource

Pauli Z (Z)
"""""""""""

.. autoclass:: qibo.gates.Z
    :members:
    :member-order: bysource

S gate (S)
"""""""""""

.. autoclass:: qibo.gates.S
    :members:
    :member-order: bysource

T gate (T)
"""""""""""

.. autoclass:: qibo.gates.T
    :members:
    :member-order: bysource

Identity (I)
""""""""""""

.. autoclass:: qibo.gates.I
    :members:
    :member-order: bysource

Measurement (M)
"""""""""""""""

.. autoclass:: qibo.gates.M
    :members:
    :member-order: bysource

Rotation X-axis (RX)
""""""""""""""""""""

.. autoclass:: qibo.gates.RX
    :members:
    :member-order: bysource

Rotation Y-axis (RY)
""""""""""""""""""""

.. autoclass:: qibo.gates.RY
    :members:
    :member-order: bysource

Rotation Z-axis (RZ)
""""""""""""""""""""

.. autoclass:: qibo.gates.RZ
    :members:
    :member-order: bysource

First general unitary (U1)
""""""""""""""""""""""""""

.. autoclass:: qibo.gates.U1
    :members:
    :member-order: bysource

Second general unitary (U2)
"""""""""""""""""""""""""""

.. autoclass:: qibo.gates.U2
    :members:
    :member-order: bysource

Third general unitary (U3)
""""""""""""""""""""""""""

.. autoclass:: qibo.gates.U3
    :members:
    :member-order: bysource

Two qubit gates
^^^^^^^^^^^^^^^

Controlled-NOT (CNOT)
"""""""""""""""""""""

.. autoclass:: qibo.gates.CNOT
    :members:
    :member-order: bysource

Controlled-phase (CZ)
"""""""""""""""""""""

.. autoclass:: qibo.gates.CZ
    :members:
    :member-order: bysource

Controlled-rotation X-axis (CRX)
""""""""""""""""""""""""""""""""

.. autoclass:: qibo.gates.CRX
    :members:
    :member-order: bysource

Controlled-rotation Y-axis (CRY)
""""""""""""""""""""""""""""""""

.. autoclass:: qibo.gates.CRY
    :members:
    :member-order: bysource

Controlled-rotation Z-axis (CRZ)
""""""""""""""""""""""""""""""""

.. autoclass:: qibo.gates.CRZ
    :members:
    :member-order: bysource

Controlled first general unitary (CU1)
""""""""""""""""""""""""""""""""""""""

.. autoclass:: qibo.gates.CU1
    :members:
    :member-order: bysource

Controlled second general unitary (CU2)
"""""""""""""""""""""""""""""""""""""""

.. autoclass:: qibo.gates.CU2
    :members:
    :member-order: bysource

Controlled third general unitary (CU3)
""""""""""""""""""""""""""""""""""""""

.. autoclass:: qibo.gates.CU3
    :members:
    :member-order: bysource

Swap (SWAP)
"""""""""""

.. autoclass:: qibo.gates.SWAP
    :members:
    :member-order: bysource

f-Swap (FSWAP)
""""""""""""""

.. autoclass:: qibo.gates.FSWAP
    :members:
    :member-order: bysource

fSim
""""

.. autoclass:: qibo.gates.fSim
    :members:
    :member-order: bysource

fSim with general rotation
""""""""""""""""""""""""""

.. autoclass:: qibo.gates.GeneralizedfSim
    :members:
    :member-order: bysource


Special gates
^^^^^^^^^^^^^

Toffoli
"""""""

.. autoclass:: qibo.gates.TOFFOLI
    :members:
    :member-order: bysource

Arbitrary unitary
"""""""""""""""""

.. autoclass:: qibo.gates.Unitary
    :members:
    :member-order: bysource

Variational layer
"""""""""""""""""

.. autoclass:: qibo.gates.VariationalLayer
    :members:
    :member-order: bysource

Callback gate
"""""""""""""

.. autoclass:: qibo.gates.CallbackGate
    :members:
    :member-order: bysource

Fusion gate
"""""""""""

.. autoclass:: qibo.gates.FusedGate
    :members:
    :member-order: bysource

_______________________

.. _Channels:

Channels
--------

Channels are implemented in Qibo as additional gates and can be accessed from
the ``qibo.gates`` module. Channels can be used on density matrices to perform
noisy simulations. Channels that inherit :class:`qibo.gates.UnitaryChannel`
can also be applied to state vectors using sampling and repeated execution.
For more information on the use of channels to simulate noise we refer to
:ref:`How to perform noisy simulation? <noisy-example>`
The following channels are currently implemented:

Kraus channel
^^^^^^^^^^^^^

.. autoclass:: qibo.gates.KrausChannel
    :members:
    :member-order: bysource

Unitary channel
^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.UnitaryChannel
    :members:
    :member-order: bysource

Pauli noise channel
^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.PauliNoiseChannel
    :members:
    :member-order: bysource

Reset channel
^^^^^^^^^^^^^

.. autoclass:: qibo.gates.ResetChannel
    :members:
    :member-order: bysource

Thermal relaxation channel
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.ThermalRelaxationChannel
    :members:
    :member-order: bysource

_______________________

Noise
-----

In Qibo it is possible to create a custom noise model using the
class :class:`qibo.noise.NoiseModel`. This enables the user to create
circuits where the noise is gate and qubit dependent.

For more information on the use of :class:`qibo.noise.NoiseModel` see
:ref:`How to perform noisy simulation? <noisemodel-example>`

.. autoclass:: qibo.noise.NoiseModel
    :members:
    :member-order: bysource

Quantum errors
^^^^^^^^^^^^^^

The quantum errors available to build a noise model are the following:

.. autoclass:: qibo.noise.PauliError
    :members:
    :member-order: bysource

.. autoclass:: qibo.noise.ThermalRelaxationError
    :members:
    :member-order: bysource

.. autoclass:: qibo.noise.ResetError
    :members:
    :member-order: bysource


.. _Hamiltonians:

Hamiltonians
------------

The main abstract Hamiltonian object of Qibo is:

.. autoclass:: qibo.hamiltonians.abstract.AbstractHamiltonian
    :members:
    :member-order: bysource


Matrix Hamiltonian
^^^^^^^^^^^^^^^^^^

The first implementation of Hamiltonians uses the full matrix representation
of the Hamiltonian operator in the computational basis. This matrix has size
``(2 ** nqubits, 2 ** nqubits)`` and therefore its construction is feasible
only when number of qubits is small.

Alternatively, the user can construct this Hamiltonian using a sparse matrices.
Sparse matrices from the
`scipy.sparse <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_
module are supported by the numpy and qibojit backends while the
`tf.sparse <https://www.tensorflow.org/api_docs/python/tf/sparse>_` can be
used for tensorflow. Scipy sparse matrices support algebraic
operations (addition, subtraction, scalar multiplication), linear algebra
operations (eigenvalues, eigenvectors, matrix exponentiation) and
multiplication to dense or other sparse matrices. All these properties are
inherited by :class:`qibo.hamiltonians.Hamiltonian` objects created
using sparse matrices. Tensorflow sparse matrices support only multiplication
to dense matrices. Both backends support calculating Hamiltonian expectation
values using a sparse Hamiltonian matrix.

.. autoclass:: qibo.hamiltonians.Hamiltonian
    :members:
    :member-order: bysource
    :noindex:


Symbolic Hamiltonian
^^^^^^^^^^^^^^^^^^^^

Qibo allows the user to define Hamiltonians using ``sympy`` symbols. In this
case the full Hamiltonian matrix is not constructed unless this is required.
This makes the implementation more efficient for larger qubit numbers.
For more information on constructing Hamiltonians using symbols we refer to the
:ref:`How to define custom Hamiltonians using symbols? <symbolicham-example>` example.

.. autoclass:: qibo.hamiltonians.SymbolicHamiltonian
    :members:
    :member-order: bysource
    :noindex:


When a :class:`qibo.hamiltonians.SymbolicHamiltonian` is used for time
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
    :class:`qibo.hamiltonians.Hamiltonian` using ``dense=True``
    or :class:`qibo.hamiltonians.SymbolicHamiltonian`
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
and can be used to construct :class:`qibo.hamiltonians.SymbolicHamiltonian`
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

Qibo circuits return :class:`qibo.states.CircuitResult` objects
when executed. By default, Qibo works as a wave function simulator in the sense
that propagates the state vector through the circuit applying the
corresponding gates. In this default usage the result of a circuit execution
is the full final state vector which can be accessed via :meth:`qibo.states.CircuitResult.state`.
However, for specific applications it is useful to have measurement samples
from the final wave function, instead of its full vector form.
To that end, :class:`qibo.states.CircuitResult` provides the
:meth:`qibo.states.CircuitResult.samples` and
:meth:`qibo.states.CircuitResult.frequencies` methods.

The state vector (or density matrix) is saved in memory as a tensor supported
by the currently active backend (see :ref:`Backends <Backends>` for more information).
A copy of the state can be created using :meth:`qibo.states.CircuitResult.copy`.
The new state will point to the same tensor in memory as the original one unless
the ``deep=True`` option was used during the ``copy`` call.
Note that the qibojit backend performs in-place updates
state is used as input to a circuit or time evolution. This will modify the
state's tensor and the tensor of all shallow copies and the current state vector
values will be lost. If you intend to keep the current state values,
we recommend creating a deep copy before using it as input to a qibo model.

In order to perform measurements the user has to add the measurement gate
:class:`qibo.gates.M` to the circuit and then execute providing a number
of shots. If this is done, the :class:`qibo.states.CircuitResult`
returned by the circuit will contain the measurement samples.

For more information on measurements we refer to the
:ref:`How to perform measurements? <measurement-examples>` example.

Circuit result
^^^^^^^^^^^^^^

.. autoclass:: qibo.states.CircuitResult
    :members:
    :member-order: bysource


.. _Callbacks:

Callbacks
---------

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
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.callbacks.EntanglementEntropy
   :members:
   :member-order: bysource

Norm
^^^^

.. autoclass:: qibo.callbacks.Norm
   :members:
   :member-order: bysource

Overlap
^^^^^^^

.. autoclass:: qibo.callbacks.Overlap
    :members:
    :member-order: bysource

Energy
^^^^^^

.. autoclass:: qibo.callbacks.Energy
    :members:
    :member-order: bysource

Gap
^^^

.. autoclass:: qibo.callbacks.Gap
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
:class:`qibo.backends.abstract.Backend`. This object defines the methods
required by all Qibo models to perform simulation.

Qibo currently provides two different calculation backends, one based on
numpy and one based on Tensorflow. It is possible to define new backends by
inheriting :class:`qibo.backends.abstract.Backend` and implementing
its abstract methods.

Both backends are supplemented by custom operators defined under which can be
used to efficiently apply gates to state vectors or density matrices.
These custom operators are shipped as the separate library qibojit.
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
    qibo.set_backend("qibojit")
    qibo.set_backend("numpy")

before creating any circuits or gates. The default backend is the first available
from ``qibojit``, ``tensorflow``, ``numpy``.

Some backends support different platforms. For example, the qibojit backend
provides two platforms (``cupy`` and ``cuquantum``) when used on GPU.
The active platform can be switched using

.. code-block::  python

    import qibo
    qibo.set_backend("qibojit", platform="cuquantum")
    qibo.set_backend("qibojit", platform="cupy")

For developers, we provide a configuration file in
``qibo/backends/profiles.yml`` containing the technical specifications for all
backends supported by the Qibo team. If you are planning to introduce a new
backend module for simulation or hardware, you can simply edit this profile file
and include the reference to your new module. Alternatively, you can set a
custom profile file by storing the file path in the ``QIBO_PROFILE`` environment
variable before executing the code.

.. autoclass:: qibo.backends.abstract.Backend
    :members:
    :member-order: bysource
