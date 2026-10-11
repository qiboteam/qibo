.. _state-tomography-tutorial:

State tomography tutorial
=========================

Quantum state tomography reconstructs the density matrix of a state from measurements.
In this tutorial we use :class:`qibo.tomography.state_tomography.StateTomography`, which
implements the vanilla protocol: Pauli measurements and linear inversion. We go through:

* the idea, and the circuits that are executed;
* the reconstruction of a state, and its properties;
* how the error depends on the number of shots, and how the cost depends on the number of qubits;
* how to run the protocol on other backends, and its limitations.

All the outputs below were obtained with the seeds shown, on the ``numpy`` backend.
Estimates are random, so different seeds, or versions of the libraries, give different digits.


The idea
--------

The density matrix of :math:`n` qubits is determined by the expectation values of the
:math:`4^{n}` Pauli strings :math:`P`, since

.. math::
    \rho = \frac{1}{2^{n}} \sum_{P} \text{tr}(P \, \rho) \, P \, .

All the Pauli strings are measured with only :math:`3^{n}` *settings*: in each one, every qubit
is measured in the basis :math:`X`, :math:`Y`, or :math:`Z`, which is done by rotating the qubit
and measuring it in the computational basis. The outcomes of one setting give the expectation
values of all the Pauli strings that have, on each qubit, either the Pauli of the setting or the
identity. A Pauli string with :math:`w` non-identity factors is thus measured by
:math:`3^{n - w}` settings, and all the estimates are averaged. Replacing the expectation values
in the expansion above by these estimates gives the estimated state :math:`\hat{\rho}`.


The circuits
------------

``circuits`` appends the rotations and the measurements of every setting to the circuit that
prepares the state. As an example, we take the Bell state of two qubits:

.. code-block:: python

    from qibo import Circuit, gates
    from qibo.tomography import StateTomography

    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))

    tomography = StateTomography()
    circuits = tomography.circuits(circuit)
    print(len(circuits))

.. code-block:: text

    9

The settings are in the order of ``itertools.product("XYZ", repeat=n)``, so the first circuit
measures :math:`X \otimes X`, and the sixth one measures :math:`Y \otimes Z`. The basis
:math:`X` is measured after a Hadamard gate, and :math:`Y` after an :math:`S^{\dagger}` gate
followed by a Hadamard gate:

.. code-block:: python

    circuits[0].draw()

.. code-block:: text

    0: ─H─o─H─M─
    1: ───X─H─M─

.. code-block:: python

    circuits[5].draw()

.. code-block:: text

    0: ─H─o─SDG─H─M─
    1: ───X───────M─


The estimated state
-------------------

Calling the object runs the whole pipeline: it generates the circuits, executes them with
``nshots`` shots each, and reconstructs the density matrix:

.. code-block:: python

    import numpy as np

    rho = tomography(circuit, nshots=1000, seed=1)
    print(np.round(rho.real, 3))

.. code-block:: text

    [[ 0.518  0.006 -0.016  0.5  ]
     [ 0.006 -0.001  0.     0.012]
     [-0.016  0.     0.001  0.01 ]
     [ 0.5    0.012  0.01   0.482]]

which is close to the density matrix of the Bell state, with entries :math:`1/2` in the corners
(only the real part is shown, and the largest imaginary part is :math:`0.017`). The estimate has
unit trace, is Hermitian, and it is unbiased, but it is **not** positive semidefinite in general:

.. code-block:: python

    print(f"{np.trace(rho).real:.3f}", np.allclose(rho, rho.conj().T))
    print(f"{np.linalg.eigvalsh(rho).min():+.3f}")

    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    print(f"{(bell @ rho @ bell).real:.3f}")

.. code-block:: text

    1.000 True
    -0.021
    1.000

The state is pure, so its smallest eigenvalue is zero, and the statistical errors make it
negative. The fidelity with the Bell state is still very close to one. If a physical state is
needed, the estimate must be projected onto the positive semidefinite matrices afterwards, which
is not done by this protocol.


Shots and qubits
----------------

The statistical error of the estimate decreases as :math:`1 / \sqrt{N}` with the number
of shots :math:`N` of each setting. For the Bell state, the largest error of an entry and the
Frobenius norm of the error are:

.. code-block:: python

    for nshots in (100, 1000, 10000, 100000):
        estimate = tomography(circuit, nshots=nshots, seed=1)
        error = estimate - np.outer(bell, bell)
        print(f"{nshots:>6}  {np.abs(error).max():.3f}  {np.linalg.norm(error):.3f}")

.. code-block:: text

       100  0.053  0.133
      1000  0.018  0.051
     10000  0.005  0.011
    100000  0.002  0.004

The number of circuits grows exponentially with the number of qubits: for
:math:`n = 1, 2, 3, 4, 5, 6, 8` there are :math:`3, 9, 27, 81, 243, 729, 6561` circuits, and the
reconstruction also needs memory and time that grow as :math:`4^{n}` or faster. This is why
the protocol is meant for few qubits. To estimate many observables of a larger state, use
:ref:`classical shadows <classical-shadows-tutorial>`.


Other backends and limitations
------------------------------

The circuits are run with :meth:`qibo.tomography.abstract.Tomography.execute`, with the backend
given in the argument ``backend`` of the call, so a hardware backend, or a backend that simulates
noise, is used without changes to the code. The outcomes are read from the frequencies of the
results, hence only sampled outcomes are needed.

* The state preparation and measurement errors of the device are not corrected: they are part of
  the reconstructed state.
* The estimate is not positive semidefinite in general, as shown above.
* The cost, :math:`3^{n}` circuits, makes the protocol impractical beyond a few qubits.

The protocol is an implementation of the abstract class
:class:`qibo.tomography.abstract.Tomography`, so it can be subclassed to change the settings, or the
way the circuits are executed.
