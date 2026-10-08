.. _Transpiler:

Transpiler
==========

This module provides tools for transpilation of generic quantum circuits
into a given set of native gates.

.. automodule:: qibo.transpiler
   :members:
   :member-order: bysource


Abstract classes
----------------


Optimizer
^^^^^^^^^

.. autoclass:: qibo.transpiler.abstract.Optimizer
    :members:
    :member-order: bysource


Placer
^^^^^^

.. autoclass:: qibo.transpiler.abstract.Placer
    :members:
    :member-order: bysource


Router
^^^^^^

.. autoclass:: qibo.transpiler.abstract.Router
    :members:
    :member-order: bysource


Asserts
-------

Assert circuit equivalence
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: qibo.transpiler.asserts.assert_circuit_equivalence


Assert connectivity
^^^^^^^^^^^^^^^^^^^

.. autofunction:: qibo.transpiler.asserts.assert_connectivity


Assert decomposition
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: qibo.transpiler.asserts.assert_decomposition


Assert placement
^^^^^^^^^^^^^^^^

.. autofunction:: qibo.transpiler.asserts.assert_placement


Assert transpilation
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: qibo.transpiler.asserts.assert_transpiling


Blocks
------

Block
^^^^^

.. autoclass:: qibo.transpiler.blocks.Block
    :members:
    :member-order: bysource


Circuit blocks
^^^^^^^^^^^^^^

.. autoclass:: qibo.transpiler.blocks.CircuitBlocks
    :members:
    :member-order: bysource


Block decomposition
^^^^^^^^^^^^^^^^^^^

.. autofunction:: qibo.transpiler.blocks.block_decomposition


Decompositions
--------------

Gate decompositions
^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.transpiler.decompositions.GateDecompositions
    :members:
    :member-order: bysource


Multi-controlled gate decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`tutorials_multicontrolled` for examples of all the options.

.. autofunction:: qibo.transpiler.multicontrolled_decompositions.multi_controlled_decomposition


Optimizer
---------

Inverse cancellation
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.transpiler.optimizer.InverseCancellation
    :members:
    :member-order: bysource


Optimize 1-qubit gates decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.transpiler.optimizer.Optimize1qGatesDecomposition
    :members:
    :member-order: bysource


Parametrized gate fusion
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.transpiler.optimizer.ParametrizedGateFusion
    :members:
    :member-order: bysource


Preprocessing
^^^^^^^^^^^^^

.. autoclass:: qibo.transpiler.optimizer.Preprocessing
    :members:
    :member-order: bysource


Rearrange
^^^^^^^^^

.. autoclass:: qibo.transpiler.optimizer.Rearrange
    :members:
    :member-order: bysource


Remove diagonal gates before measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.transpiler.optimizer.RemoveDiagonalGatesBeforeMeasurement
    :members:
    :member-order: bysource


Remove final reset
^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.transpiler.optimizer.RemoveFinalReset
    :members:
    :member-order: bysource


Remove identity equivalent
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.transpiler.optimizer.RemoveIdentityEquivalent
    :members:
    :member-order: bysource


Remove reset in zero state
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.transpiler.optimizer.RemoveResetInZeroState
    :members:
    :member-order: bysource


Reset after measure simplification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.transpiler.optimizer.ResetAfterMeasureSimplification
    :members:
    :member-order: bysource




T-gate rules
^^^^^^^^^^^^

.. autoclass:: qibo.transpiler.optimizer.TGateRules
    :members:
    :member-order: bysource


Pipeline
--------

Passes
^^^^^^

.. autoclass:: qibo.transpiler.pipeline.Passes
    :members:
    :member-order: bysource


Restrict qubit connectivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: qibo.transpiler.pipeline.restrict_connectivity_qubits


Placer
------

Random
^^^^^^

.. autoclass:: qibo.transpiler.placer.Random
    :members:
    :member-order: bysource


Reverse traversal
^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.transpiler.placer.ReverseTraversal
    :members:
    :member-order: bysource


Subgraph
^^^^^^^^

.. autoclass:: qibo.transpiler.placer.Subgraph
    :members:
    :member-order: bysource


Star connectivity
^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.transpiler.placer.StarConnectivityPlacer
    :members:
    :member-order: bysource


Router
------

Circuit map
^^^^^^^^^^^

.. autoclass:: qibo.transpiler.router.CircuitMap
    :members:
    :member-order: bysource


Sabre
^^^^^

.. autoclass:: qibo.transpiler.router.Sabre
    :members:
    :member-order: bysource


Shorterst paths
^^^^^^^^^^^^^^^

.. autoclass:: qibo.transpiler.router.ShortestPaths
    :members:
    :member-order: bysource


Start connectivity
^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.transpiler.router.StarConnectivityRouter
    :members:
    :member-order: bysource


Unroller
--------

Native gates
^^^^^^^^^^^^

.. autoclass:: qibo.transpiler.unroller.NativeGates
    :members:
    :member-order: bysource


Unroller
^^^^^^^^

.. autoclass:: qibo.transpiler.unroller.Unroller
    :members:
    :member-order: bysource


Translate gate
^^^^^^^^^^^^^^

.. autofunction:: qibo.transpiler.unroller.translate_gate
