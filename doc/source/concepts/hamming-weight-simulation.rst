.. _Hamming-weight:

Simulation of Hamming-weight-preserving circuits
================================================

This specialised backend in ``qibo`` supports the fast simulation of
Hamming-weight-preserving circuits.
Given a quantum state that only contains non-zero amplitudes associated with computational basis
states of Hamming weight :math:`k`, the :class:`qibo.backends.hamming_weight.HammingWeightBackend` works with
a compressed representation of quantum states that is restricted to the subspace
with fixed Hamming weight :math:`k`.
The dimension of this subspace is :math:`d = \binom{n}{k}`,
where :math:`n` is the total number of qubits.

As for the other backends, the Hamming-weight backend can be set
by specifying the engine used for calculation, *i.e.*

.. testcode::  python

    from qibo import set_backend

    set_backend("hamming_weight", platform="numpy")

If not provided, the current global backend is used:

.. testcode::  python

    from qibo import set_backend
    from qibo.backends import HammingWeightBackend

    # setting numpy as the global backend
    set_backend("numpy")
    # the Hamming-weight backend will use the numpy backend as engine
    backend = HammingWeightBackend()

A Hamming-weight-preserving circuit with Hamming weight :math:`k` can be executed as the following example:

.. testcode::  python

    from qibo import Circuit, gates, set_backend
    from qibo.backends import HammingWeightBackend

    backend = HammingWeightBackend()

    nqubits = 4
    circuit = Circuit(nqubits)
    circuit.add(gates.SWAP(0, 1))

    result = backend.execute_circuit(circuit, weight=2)

    # the above execution is equivalent to the following
    # that uses the global backend and the ``weight`` kwarg
    #
    # set_backend("hamming_weight", platform="numpy")
    # result = circuit(weight=2)

The results of execution of a circuit through this backend creates a
:class:`qibo.quantum_info.hamming_weight.HammingWeightResult`
object that gives access to the final state through the
:meth:`qibo.quantum_info.hamming_weight.HammingWeightResult.state` method.
The probabilities are computed by the
:meth:`qibo.quantum_info.hamming_weight.HammingWeightResult.probabilities` method.
If there are no measurements in the circuit or the number of shots is :math:`0`,
they are calculated from the statevector.
Otherwise, they are calculated from the samples.
The final measured samples and frequencies are accessible, respectively, through the
:meth:`qibo.quantum_info.hamming_weight.HammingWeightResult.samples` and the
:meth:`qibo.quantum_info.hamming_weight.HammingWeightResult.frequencies` methods.

It is also possible to recover the standard state representation with the
:meth:`qibo.quantum_info.hamming_weight.HammingWeightResult.full_state` method.
Note, however, that this process is inefficient as it involves the construction of
a statevector with size exponential in the number of qubits.
