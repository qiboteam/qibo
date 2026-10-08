.. _Gates:

Gates
=====

All supported gates can be accessed from the ``qibo.gates`` module.
Read below for a complete list of supported gates.

All gates support the ``controlled_by`` method that allows to control
the gate on an arbitrary number of qubits. For example

* ``gates.X(0).controlled_by(1, 2)`` is equivalent to ``gates.TOFFOLI(1, 2, 0)``,
* ``gates.RY(0, np.pi).controlled_by(1, 2, 3)`` applies the Y-rotation to qubit 0 when qubits 1, 2 and 3 are in the ``|111>`` state.
* ``gates.SWAP(0, 1).controlled_by(3, 4)`` swaps qubits 0 and 1 when qubits 3 and 4 are in the ``|11>`` state.

Abstract gate
-------------

.. autoclass:: qibo.gates.abstract.Gate
    :members:
    :member-order: bysource
    :noindex:

Single qubit gates
------------------

Hadamard (H)
^^^^^^^^^^^^

.. autoclass:: qibo.gates.H
   :members:
   :member-order: bysource

Pauli X (X)
^^^^^^^^^^^

.. autoclass:: qibo.gates.X
   :members:
   :member-order: bysource

Pauli Y (Y)
^^^^^^^^^^^

.. autoclass:: qibo.gates.Y
    :members:
    :member-order: bysource

Pauli Z (Z)
^^^^^^^^^^^

.. autoclass:: qibo.gates.Z
    :members:
    :member-order: bysource

Square-root of Pauli X (SX)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.SX
    :members:
    :member-order: bysource

S gate (S)
^^^^^^^^^^^

.. autoclass:: qibo.gates.S
    :members:
    :member-order: bysource

T gate (T)
^^^^^^^^^^^

.. autoclass:: qibo.gates.T
    :members:
    :member-order: bysource

Identity (I)
^^^^^^^^^^^^

.. autoclass:: qibo.gates.I
    :members:
    :member-order: bysource

Align (A)
^^^^^^^^^

.. autoclass:: qibo.gates.Align
    :members:
    :member-order: bysource

Measurement (M)
^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.M
    :members:
    :member-order: bysource

Rotation X-axis (RX)
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.RX
    :members:
    :member-order: bysource

Rotation Y-axis (RY)
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.RY
    :members:
    :member-order: bysource

Rotation Z-axis (RZ)
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.RZ
    :members:
    :member-order: bysource

First general unitary (U1)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.U1
    :members:
    :member-order: bysource

Second general unitary (U2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.U2
    :members:
    :member-order: bysource

Third general unitary (U3)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.U3
    :members:
    :member-order: bysource

Two qubit gates
---------------

Controlled-NOT (CNOT)
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.CNOT
    :members:
    :member-order: bysource

Controlled-Y (CY)
^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.CY
    :members:
    :member-order: bysource

Controlled-phase (CZ)
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.CZ
    :members:
    :member-order: bysource

Controlled-Hadamard (CH)
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.CH
    :members:
    :member-order: bysource

Controlled-Square Root of X (CSX)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.CSX
    :members:
    :member-order: bysource

Controlled-rotation X-axis (CRX)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.CRX
    :members:
    :member-order: bysource

Controlled-rotation Y-axis (CRY)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.CRY
    :members:
    :member-order: bysource

Controlled-rotation Z-axis (CRZ)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.CRZ
    :members:
    :member-order: bysource

Controlled first general unitary (CU1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.CU1
    :members:
    :member-order: bysource

Controlled second general unitary (CU2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.CU2
    :members:
    :member-order: bysource

Controlled third general unitary (CU3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.CU3
    :members:
    :member-order: bysource

Swap (SWAP)
^^^^^^^^^^^

.. autoclass:: qibo.gates.SWAP
    :members:
    :member-order: bysource

iSwap (iSWAP)
^^^^^^^^^^^^^

.. autoclass:: qibo.gates.iSWAP
    :members:
    :member-order: bysource

Square root of iSwap (SiSWAP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.SiSWAP
    :members:
    :member-order: bysource

f-Swap (FSWAP)
^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.FSWAP
    :members:
    :member-order: bysource

fSim
^^^^

.. autoclass:: qibo.gates.fSim
    :members:
    :member-order: bysource

Sycamore gate
^^^^^^^^^^^^^

.. autoclass:: qibo.gates.SYC
    :members:
    :member-order: bysource

fSim with general rotation
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.GeneralizedfSim
    :members:
    :member-order: bysource

Parametric XX interaction (RXX)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.RXX
    :members:
    :member-order: bysource

Parametric YY interaction (RYY)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.RYY
    :members:
    :member-order: bysource

Parametric ZZ interaction (RZZ)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.RZZ
    :members:
    :member-order: bysource

Parametric ZX interaction (RZX)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.RZX
    :members:
    :member-order: bysource

Parametric XX-YY interaction (RXXYY)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.RXXYY
    :members:
    :member-order: bysource

Givens gate
^^^^^^^^^^^

.. autoclass:: qibo.gates.GIVENS
    :members:
    :member-order: bysource

Reconfigurable Beam Splitter gate (RBS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.RBS
    :members:
    :member-order: bysource

Echo Cross-Resonance gate (ECR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.ECR
    :members:
    :member-order: bysource

Special gates
-------------

Toffoli
^^^^^^^

.. autoclass:: qibo.gates.TOFFOLI
    :members:
    :member-order: bysource

CCZ
^^^

.. autoclass:: qibo.gates.CCZ
    :members:
    :member-order: bysource

Deutsch
^^^^^^^

.. autoclass:: qibo.gates.DEUTSCH
    :members:
    :member-order: bysource


Fan-out
^^^^^^^

.. autoclass:: qibo.gates.FanOut
    :members:
    :member-order: bysource


Generalized Reconfigurable Beam Splitter (RBS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.GeneralizedRBS
    :members:
    :member-order: bysource


Arbitrary unitary
^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.Unitary
    :members:
    :member-order: bysource


Barrier
^^^^^^^

.. autoclass:: qibo.gates.Barrier
    :members:
    :member-order: bysource


Callback gate
^^^^^^^^^^^^^

.. autoclass:: qibo.gates.CallbackGate
    :members:
    :member-order: bysource

Fusion gate
^^^^^^^^^^^

.. autoclass:: qibo.gates.FusedGate
    :members:
    :member-order: bysource

IONQ Native gates
-----------------

GPI
^^^

.. autoclass:: qibo.gates.GPI
    :members:
    :member-order: bysource

GPI2
^^^^

.. autoclass:: qibo.gates.GPI2
    :members:
    :member-order: bysource

Mølmer–Sørensen (MS)
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.MS
    :members:
    :member-order: bysource

Quantinuum native gates
-----------------------

U1q
^^^

.. autoclass:: qibo.gates.U1q
    :members:
    :member-order: bysource

.. note::
    The other Quantinuum single-qubit and two-qubit native gates are
    implemented in Qibo as:

    - Pauli-:math:`Z` rotation: :class:`qibo.gates.RZ`
    - Arbitrary :math:`ZZ` rotation: :class:`qibo.gates.RZZ`
    - Fully-entangling :math:`ZZ`-interaction: :math:`R_{ZZ}(\pi/2)`


IQM native gates
----------------

Phase-:math:`RX`
^^^^^^^^^^^^^^^^

.. autoclass:: qibo.gates.PRX
    :members:
    :member-order: bysource

.. note::
    The other IQM two-qubit native gate is implemented in Qibo as:

    - Controlled-:math:`Z` rotation: :class:`qibo.gates.CZ`
