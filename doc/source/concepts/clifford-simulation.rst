.. _Clifford:

Clifford Simulation
===================

A special backend in qibo supports the simulation of Clifford circuits.
This :class:`qibo.backends.clifford.CliffordBackend` backend implements the phase-space formalism
introduced in `https://arxiv.org/abs/quant-ph/0406196 <aaronson_>`_ to efficiently simulate gate
application and measurements sampling in the stabilizers state representation.
The execution of a circuit through this backend creates a
:class:`qibo.quantum_info.clifford.Clifford` object that gives access to the final measured
samples through the :meth:`qibo.quantum_info.clifford.Clifford.samples` method,
similarly to :class:`qibo.result.CircuitResult`.
The probabilities and frequencies are computed starting from the samples by the
:meth:`qibo.quantum_info.clifford.Clifford.frequencies` and
:meth:`qibo.quantum_info.clifford.Clifford.probabilities` methods.

.. _aaronson: https://arxiv.org/abs/quant-ph/0406196

It is also possible to recover the standard state representation with the
:meth:`qibo.quantum_info.clifford.Clifford.state` method.
Note, however, that this process is inefficient as it involves the construction of all the
stabilizers starting from the generators encoded inside the symplectic matrix.

As for the other backends, the Clifford backend can be set with

.. testcode::  python

    import qibo
    qibo.set_backend("clifford", platform="numpy")

by specifying the engine used for calculation, if not provided the current backend is used

.. testcode::  python

    import qibo

    # setting numpy as the global backend
    qibo.set_backend("numpy")
    # the clifford backend will use the numpy backend as engine
    backend = qibo.backends.CliffordBackend()

Alternatively, a Clifford circuit can also be executed starting from the :class:`qibo.quantum_info.clifford.Clifford` object

.. code-block::  python

    from qibo.quantum_info import Clifford, random_clifford

    nqubits = 2
    circuit = random_clifford(nqubits)
    result = Clifford.from_circuit(circuit)
