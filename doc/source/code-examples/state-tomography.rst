.. _state-tomography-tutorial:

State tomography tutorial
=========================

Quantum state tomography reconstructs the density matrix of a state from measurements.
In this tutorial we use :class:`qibo.tomography.state_tomography.StateTomography`, which
implements the protocol with Pauli measurements, and four estimators of the state. We go through:

* the idea, and the circuits that are executed;
* the reconstruction of a state by linear inversion, and its properties;
* the physical estimators: projected linear inversion, least squares, and maximum likelihood;
* how the error depends on the number of shots, and how the cost depends on the number of qubits;
* how to mitigate readout errors;
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

    import math
    from qibo.backends import NumpyBackend

    backend = NumpyBackend()

    rho = tomography(circuit, nshots=1000, seed=1, backend=backend)
    print(backend.round(backend.real(rho), 3))

.. code-block:: text

    [[ 0.518  0.006 -0.016  0.5  ]
     [ 0.006 -0.001  0.     0.012]
     [-0.016  0.     0.001  0.01 ]
     [ 0.5    0.012  0.01   0.482]]

which is close to the density matrix of the Bell state, with entries :math:`1/2` in the corners
(only the real part is shown, and the largest imaginary part is :math:`0.017`). The estimate has
unit trace, is Hermitian, and it is unbiased, but it is **not** positive semidefinite in general:

.. code-block:: python

    print(f"{backend.real(backend.trace(rho)):.3f}", backend.allclose(rho, backend.dagger(rho)))
    print(f"{backend.eigvalsh(rho).min():+.3f}")

    bell = backend.cast([1, 0, 0, 1]) / math.sqrt(2)
    print(f"{(bell @ rho @ bell).real:.3f}")

.. code-block:: text

    1.000 True
    -0.021
    1.000

The state is pure, so its smallest eigenvalue is zero, and the statistical errors make it
negative. The fidelity with the Bell state is still very close to one, but if a physical state is
needed, one of the estimators of the next section must be used.


Physical estimates
------------------

The argument ``method`` selects how the outcomes are turned into a state. Besides
``"linear-inversion"``, which is the default, there are three estimators whose result is always
positive semidefinite, with unit trace:

* ``"physical-linear-inversion"`` projects the linear inversion estimate onto the closest density
  matrix in the Frobenius norm, i.e. it sets the negative eigenvalues to zero and shifts the
  others to keep the trace [1]. It is the cheapest, and it needs no iterations.
* ``"least-squares"`` finds the physical state whose predicted probabilities are the closest, in
  the sum of squares, to the observed frequencies of all outcomes of all settings. It is solved
  by an accelerated projected gradient method.
* ``"mle"`` finds the physical state that maximizes the likelihood of the observed outcomes, with
  the diluted iterative algorithm [2, 3].

The last two are iterative, and stop after ``iterations`` iterations, or when the largest
change of an element of the estimate is smaller than ``tolerance``. We compare them with the
data of 100 shots per setting, which is small enough to make the linear inversion estimate
unphysical:

.. code-block:: python

    for method in (
        "linear-inversion",
        "physical-linear-inversion",
        "least-squares",
        "mle",
    ):
        rho = tomography(circuit, nshots=100, seed=1, method=method, backend=backend)
        eigenvalue = backend.round(backend.eigvalsh(rho).min(), 3) + 0.0
        fidelity = (bell @ rho @ bell).real
        error = backend.matrix_norm(rho - backend.outer(bell, bell))
        print(f"{method:<26} {eigenvalue:+.3f}  {fidelity:.3f}  {error:.3f}")

.. code-block:: text

    linear-inversion           -0.064  1.000  0.133
    physical-linear-inversion  +0.000  0.968  0.105
    least-squares              +0.000  0.956  0.112
    mle                        +0.000  0.998  0.068

The columns are the smallest eigenvalue, the fidelity with the Bell state, and the Frobenius norm
of the error. The three physical estimates have no negative eigenvalues, and for this seed
they are all closer to the Bell state than the linear inversion estimate. This is not guaranteed
in general, except for ``"physical-linear-inversion"``: projecting onto a convex set that
contains the state never moves the estimate away from it. The maximum likelihood estimate is
biased towards states with low rank, which suits pure states, but it converges slowly when the
state is on the boundary of the physical states, as pure states are, so a small ``tolerance``
may need many ``iterations``.


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


Readout errors
--------------

Errors in the readout flip the outcomes of the measurements, and they are part of the
reconstructed state unless they are corrected. If the readout is characterized by a response
matrix, whose element :math:`(i, j)` is the probability of reading the outcome :math:`i` when the
computational basis state :math:`j` is prepared, it can be given in the argument
``response_matrix``. The linear estimators multiply the probabilities of each setting by its
inverse, and ``"least-squares"`` and ``"mle"`` include the response matrix in the model of the
probabilities, so that the frequencies are used as they were observed.

As an example, we simulate a readout that flips :math:`0` into :math:`1` with probability
:math:`0.07`, and :math:`1` into :math:`0` with probability :math:`0.12`, by subclassing the
protocol to execute the circuits with a noise model. The response matrix is estimated with
:func:`qibo.models.error_mitigation.get_response_matrix`, which, on a device, would be done by
preparing and measuring all the computational basis states:

.. code-block:: python

    from qibo.backends import get_backend
    from qibo.models.error_mitigation import get_response_matrix
    from qibo.noise import NoiseModel, ReadoutError

    backend = get_backend()
    backend.set_seed(7)

    readout = backend.cast([[0.93, 0.07], [0.12, 0.88]], dtype=backend.float64)
    noise_model = NoiseModel()
    noise_model.add(ReadoutError(backend.kron(readout, readout)), gates.M)


    class NoisyStateTomography(StateTomography):
        def execute(self, circuits, nshots=1000, backend=None):
            results = []
            for tomography_circuit in circuits:
                noisy = noise_model.apply(tomography_circuit)
                noisy.density_matrix = True
                results.append(backend.execute_circuit(noisy, nshots=nshots))
            return results

    response_matrix = get_response_matrix(
        2, noise_model=noise_model, nshots=100000, backend=backend
    )
    print(backend.round(response_matrix, 2))

.. code-block:: text

    [[0.86 0.11 0.11 0.01]
     [0.07 0.82 0.01 0.1 ]
     [0.07 0.01 0.82 0.1 ]
     [0.   0.06 0.06 0.78]]

The fidelity with the Bell state, from 20000 shots per setting, without and with the correction is:

.. code-block:: python

    noisy_tomography = NoisyStateTomography()
    for method in ("linear-inversion", "mle"):
        for matrix in (None, response_matrix):
            rho = noisy_tomography(
                circuit,
                nshots=20000,
                seed=1,
                method=method,
                response_matrix=matrix,
                backend=backend,
            )
            label = "raw" if matrix is None else "mitigated"
            print(f"{method:<18} {label:<10} {(bell @ rho @ bell).real:.3f}")

.. code-block:: text

    linear-inversion   raw        0.740
    linear-inversion   mitigated  0.995
    mle                raw        0.740
    mle                mitigated  0.992

The correction is as good as the response matrix: it does not mitigate errors that depend on the
setting, and the errors of the state preparation, which are not those of the readout, are not
corrected.


Other backends and limitations
------------------------------

The circuits are run with :meth:`qibo.tomography.abstract.Tomography.execute`, with the backend
given in the argument ``backend`` of the call, so a hardware backend, or a backend that simulates
noise, is used without changes to the code. The outcomes are read from the frequencies of the
results, hence only sampled outcomes are needed.

* The errors of the state preparation are not corrected, and neither are those of the readout,
  unless a ``response_matrix`` is given: they are part of the reconstructed state.
* The estimate of ``"linear-inversion"`` is not positive semidefinite in general, as shown above.
* The estimators are not designed for a large number of qubits: the cost, :math:`3^{n}` circuits,
  and the memory of the iterative ones, make the protocol impractical beyond a few qubits.

The protocol is an implementation of the abstract class
:class:`qibo.tomography.abstract.Tomography`, so it can be subclassed to change the settings, or the
way the circuits are executed.


References
----------

1. J. A. Smolin, J. M. Gambetta and G. Smith,
   *Efficient method for computing the maximum-likelihood quantum state from measurements with
   additive Gaussian noise* (2012),
   `Physical Review Letters 108, 070502 <https://doi.org/10.1103/PhysRevLett.108.070502>`_.

2. Z. Hradil, *Quantum-state estimation* (1997),
   `Physical Review A 55, R1561 <https://doi.org/10.1103/PhysRevA.55.R1561>`_.

3. A. I. Lvovsky, *Iterative maximum-likelihood reconstruction in quantum homodyne
   tomography* (2004),
   `Journal of Optics B 6, S556 <https://doi.org/10.1088/1464-4266/6/6/014>`_.
