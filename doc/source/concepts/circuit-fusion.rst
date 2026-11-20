.. _circuit-fusion:

Circuit fusion
==============

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

    from qibo import Circuit, gates

    circuit = Circuit(2)
    circuit.add([gates.H(0), gates.H(1)])
    circuit.add(gates.CZ(0, 1))
    circuit.add([gates.X(0), gates.Y(1)])
    fused_circuit = circuit.fuse()

will create a new circuit with a single :class:`qibo.gates.special.FusedGate`
acting on ``(0, 1)``, while the following:

.. testcode::

    from qibo import Circuit, gates

    circuit = Circuit(3)
    circuit.add([gates.H(0), gates.H(1), gates.H(2)])
    circuit.add(gates.CZ(0, 1))
    circuit.add([gates.X(0), gates.Y(1), gates.Z(2)])
    circuit.add(gates.CNOT(1, 2))
    circuit.add([gates.H(0), gates.H(1), gates.H(2)])
    fused_circuit = circuit.fuse()

will give a circuit with two fused gates, the first of which will act on
``(0, 1)`` corresponding to

.. code-block::  python

    [H(0), H(1), CZ(0, 1), X(0), H(0)]

and the second will act to ``(1, 2)`` corresponding to

.. code-block::  python

    [Y(1), Z(2), CNOT(1, 2), H(1), H(2)]

Parametrized custom gates
-------------------------

Use :class:`qibo.gates.ParametrizedFusedGate` to build a reusable gate from
several operations while retaining their trainable parameters. The block is
applied as a single matrix, and its parameters can be updated with
:meth:`qibo.models.circuit.Circuit.set_parameters` without rebuilding it.

For example, the following block exposes two rotation angles and keeps a
third rotation fixed:

.. testcode::

    from qibo import Circuit, gates

    block = gates.ParametrizedFusedGate(0, 1)
    block.append(gates.RX(0, theta=0.1))
    block.append(gates.CNOT(0, 1))
    block.append(gates.RY(1, theta=-0.2))
    fixed_rotation = gates.RZ(1, theta=0.3, trainable=False)
    block.append(fixed_rotation)

    circuit = Circuit(2)
    circuit.add(block)
    assert circuit.get_parameters() == [(0.1, -0.2)]

    # Update with a flat vector, as in a variational optimization loop.
    circuit.set_parameters([0.4, -0.5])
    assert block.parameters == (0.4, -0.5)
    assert fixed_rotation.parameters == (0.3,)

    # A dictionary can update a specific block in a larger circuit.
    circuit.set_parameters({block: (0.6, 0.7)})
    assert circuit.get_parameters("flatlist") == [0.6, 0.7]

Parameters follow the order of the contained gates, including gates added
with ``prepend``. Gates such as :class:`qibo.gates.U3` expose each of their
parameters in that order. Array-valued parameters, such as the matrix of a
:class:`qibo.gates.Unitary`, retain their shape in ``block.parameters`` and
``circuit.get_parameters()``. The ``"flatlist"`` format and flat parameter
updates use the individual array elements instead.

Only trainable child gates contribute to the block's parameter interface.
Setting ``trainable=False`` on the block excludes the entire block from
circuit parameter updates. As with other parametrized gates, its parameters
can still be assigned directly. Parametrized blocks may contain other
parametrized blocks; a non-trainable nested block contributes its operation
but does not expose its parameters to the enclosing block.

Finish building the block before adding it to a circuit or another
parametrized block. Once added, ``append`` and ``prepend`` raise
``RuntimeError``, including on nested blocks, so the containing block's
and circuit's qubits and parameter count remain consistent.
Parameter values can still change. Child gates are retained by reference;
changing a child gate's parameters directly is also reflected in the block.
Keep child trainability unchanged after constructing the block.

Parametrized blocks support qubit remapping with ``on_qubits``, inversion
with ``dagger``, and independent copies with ``circuit.copy(deep=True)``.
The block is also preserved when the surrounding circuit is fused.
Analytical block gradients and OpenQASM export are not implemented.
Serialization relies on the serialization support of the contained gates.
