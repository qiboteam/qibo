.. _classical-shadows-tutorial:

Classical shadows tutorial
==========================

Classical shadows estimate the expectation values of many observables from a single set of
randomized measurements. In this tutorial we go through the functionalities of
:class:`qibo.tomography.classical_shadows.ClassicalShadow`:

* the three ensembles of random measurements ``"global-clifford"``, ``"local-clifford"``
  and ``"ultra-shallow"``;
* observables given as Pauli strings, and as Hamiltonians or matrices;
* what happens inside the pipeline: generating circuits, executing them, and the frame operator;
* the median-of-means estimator, and the options that control memory and reproducibility;
* ultra-shallow shadows, whose frame operator is sampled, and their robustness to circuit noise;
* how to write your own protocol with the abstract class :class:`qibo.tomography.abstract.Tomography`.

All the outputs below were obtained with the seeds shown, on the ``numpy`` backend.
Estimates are random, so different seeds, or versions of the libraries, give different digits.


The idea
--------

Let :math:`\rho` be the density matrix (the matrix that describes the state) of an
:math:`n`-qubit state that a circuit prepares. One *sample* of the protocol is:

1. apply to the state a random unitary :math:`U`, drawn from a fixed set of unitaries;
2. measure all qubits in the computational basis, which returns a bitstring :math:`b`;
3. store the *snapshot* :math:`\sigma = U^{\dagger} \ketbra{b}{b} U`, where
   :math:`\ketbra{b}{b}` is the projector onto the bitstring :math:`b`, and
   :math:`U^{\dagger}` is the conjugate transpose of :math:`U`.

Averaged over the random choices, snapshots do not give back :math:`\rho`, but the output of the
*frame operator* :math:`\mathcal{M}` (the measurement channel) applied to it,

.. math::
    \mathbb{E}[\sigma] = \mathcal{M}(\rho) \, .

The set of unitaries is chosen such that :math:`\mathcal{M}` can be inverted. Then,
for any observable :math:`O`, the number :math:`\text{tr}(O \, \mathcal{M}^{-1}(\bar{\sigma}))`, where
:math:`\bar{\sigma}` is the average snapshot over all samples, is an unbiased estimate of
:math:`\text{tr}(O \rho)`, the expectation value of :math:`O`. The same samples serve for
as many observables as you wish [1, 2].

A *Pauli string* is a tensor product of single-qubit Pauli matrices
:math:`I, X, Y, Z`, one per qubit, for instance ``"XIZ"`` (the first character acts on qubit 0).
Its *weight* is the number of factors different from :math:`I`, and its *support* tells which
qubits those are. All the ensembles below treat Pauli strings alike up to their support, so the
frame operator is diagonal in the Pauli basis: :math:`\mathcal{M}(P) = f_{P} P` for a Pauli string
:math:`P`, and :math:`f_{P}` is its eigenvalue.

.. list-table::
    :header-rows: 1
    :widths: 20 40 40

    * - ``method``
      - Random unitary :math:`U`
      - Eigenvalue :math:`f_{P}` of :math:`P \neq I^{\otimes n}`
    * - ``"global-clifford"``
      - uniformly from the Clifford group on :math:`n` qubits (the unitaries that map
        Pauli strings to Pauli strings)
      - :math:`1 / (2^{n} + 1)`
    * - ``"local-clifford"``
      - one random single-qubit Clifford on each qubit
      - :math:`3^{-w}`, with :math:`w` the weight of :math:`P`
    * - ``"ultra-shallow"``
      - ``depth + 1`` layers of random single-qubit Cliffords, with a layer of CNOT
        (controlled-NOT) gates before each layer but the first
      - depends on the support of :math:`P`, and is sampled [3]

The identity has eigenvalue :math:`1` in all cases.


Quick start
-----------

We estimate the fidelity of a circuit that prepares a Bell state
:math:`(\ket{00} + \ket{11}) / \sqrt{2}` with respect to the ideal Bell state, which is the expectation
value of the observable :math:`(II + ZZ + XX - YY) / 4`:

.. code-block:: python

    from qibo import Circuit, gates
    from qibo.tomography import ClassicalShadow

    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))

    # fidelity with the Bell state: (II + ZZ + XX - YY) / 4
    observable = {"II": 0.25, "ZZ": 0.25, "XX": 0.25, "YY": -0.25}

    shadow = ClassicalShadow()
    fidelity = shadow(circuit, observable, nsamples=1000, method="local-clifford", seed=1)
    print(f"{float(fidelity):.3f}")

.. code-block:: text

    0.990

The circuit must not contain measurements, since the protocol adds its own. ``nsamples`` is the
number of random unitaries, i.e. of circuits that are executed. The result is a number
(an array of the backend, which ``float`` converts) and the exact value is :math:`1`.


Choosing the ensemble
---------------------

The argument ``method`` selects the ensemble. Here the three of them, with the same observable
and number of samples:

.. code-block:: python

    for method in ("local-clifford", "global-clifford", "ultra-shallow"):
        estimate = shadow(
            circuit,
            observable,
            nsamples=1000,
            method=method,
            depth=1,
            ncalibration=3000,
            seed=1,
        )
        print(f"{method:16s} {float(estimate):.3f}")

.. code-block:: text

    local-clifford   0.990
    global-clifford  0.919
    ultra-shallow    1.010

``depth`` and ``ncalibration`` only matter for ``"ultra-shallow"``, and are discussed
:ref:`below <classical-shadows-ultra-shallow>`. For the other methods they are ignored.

As a rule of thumb [1]:

* ``"local-clifford"`` needs circuits with a single layer of one-qubit gates, and its variance for a
  Pauli string grows as :math:`3^{w}` with the weight :math:`w`. It suits observables made of
  low-weight Pauli strings.
* ``"global-clifford"`` suits observables with a large weight but low rank, such as fidelities, because
  its variance grows like :math:`2^{n}` instead. However, its circuits are deep, which on hardware
  accumulates noise.
* ``"ultra-shallow"`` lies in between: a few entangling layers (``depth``) are enough to improve over
  local Cliffords, while the circuits remain shallow [3].


Observables and bases
---------------------

By default, the frame operator is applied in the Pauli basis (``basis="pauli"``), where it is diagonal.
The observable is then written as a sum :math:`O = \sum_{P} \alpha_{P} P` of Pauli strings with real
coefficients :math:`\alpha_{P}`, given as a list of tuples ``(string, coefficient)`` or as a dictionary
``{string: coefficient}``. The estimate is

.. math::
    \hat{o} = \sum_{P} \alpha_{P} \, f_{P}^{-1} \, \text{tr}(P \bar{\sigma}) \, ,

so only the diagonal of the frame, and the Pauli coefficients :math:`\text{tr}(P \bar{\sigma})` of the
average snapshot, are needed. Those are computed for all :math:`4^{n}` strings at once with a fast
Hadamard transform. Strings that appear more than once in a list are added up:

.. code-block:: python

    terms = [("II", 0.25), ("ZZ", 0.125), ("XX", 0.25), ("YY", -0.25), ("ZZ", 0.125)]
    print(f"{float(shadow(circuit, terms, 1000, 'local-clifford', seed=1)):.3f}")

.. code-block:: text

    0.990

which is the same observable, and the same samples, as before.

With ``basis="computational"``, the observable is instead a Hamiltonian
(:class:`qibo.hamiltonians.Hamiltonian` or :class:`qibo.hamiltonians.SymbolicHamiltonian`)
or a matrix, and the inverse frame is applied to the average snapshot as a matrix. The expectation
value is then computed with the ``expectation_from_state`` method of the Hamiltonian, or with the
backend for a matrix:

.. code-block:: python

    import numpy as np
    from qibo import hamiltonians

    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    hamiltonian = hamiltonians.Hamiltonian(2, np.outer(bell, bell))  # projector on the Bell state

    state = circuit().state()
    print("exact        ", f"{float(hamiltonian.expectation_from_state(state)):.3f}")

    for observable_ in (hamiltonian, hamiltonian.matrix):
        estimate = shadow(
            circuit, observable_, 1000, "local-clifford", basis="computational", seed=1
        )
        print("computational", f"{float(estimate):.3f}")

.. code-block:: text

    exact         1.000
    computational 0.990
    computational 0.990

Both bases give the same estimate for the same samples. Use the Pauli basis whenever the observable
has a Pauli decomposition with few terms: the computational basis builds a :math:`4^{n} \times 4^{n}` matrix.


Looking inside the pipeline
---------------------------

Calling the object runs three steps, each available as a method.

**1. Circuits.** ``circuits`` appends random unitaries and a measurement of all qubits to a copy
of the circuit that prepares the state:

.. code-block:: python

    circuits = shadow.circuits(circuit, nsamples=3, method="local-clifford", seed=1)
    print(len(circuits))
    circuits[0].draw()

.. code-block:: text

    3
    0: ─H─o─H─S─H─S─M─
    1: ───X─H─S─S─H─M─

The state preparation (``H`` and ``CNOT``) is untouched. What follows are random single-qubit
Cliffords, written as sequences of ``H`` and ``S`` gates (the Hadamard and the phase gate), and the
measurement ``M``. For ``"local-clifford"`` and ``"ultra-shallow"``, the 24 single-qubit Cliffords, up to a
global phase, are tabulated, so sampling them is cheap. Circuits are reproducible given ``seed``.

**2. Execution.** ``execute`` runs a list of circuits on a backend, and returns the results in order:

.. code-block:: python

    results = shadow.execute(circuits, nshots=1)
    print([result.samples(binary=True).tolist() for result in results])

.. code-block:: text

    [[[1, 1]], [[0, 1]], [[1, 1]]]

Each random circuit is run for ``nshots`` shots (the pipeline uses one shot by default). The pipeline
passes its ``backend`` to this method, so you can execute on any simulator or on hardware.

**3. Frame operator.** ``frame_operator`` returns the frame, or its inverse with ``inverse=True``.
In the Pauli basis it returns only the diagonal, as a real vector with the :math:`4^{n}` Pauli
strings in the order ``II, IX, IY, IZ, XI, ...``, i.e. the base-4 digits of the index spell the string
(:math:`I=0, X=1, Y=2, Z=3`):

.. code-block:: python

    print(np.round(shadow.frame_operator(2, "local-clifford"), 3))
    print(np.round(shadow.frame_operator(2, "global-clifford"), 3))
    print(np.round(shadow.frame_operator(2, "local-clifford", inverse=True), 1))

.. code-block:: text

    [1.    0.333 0.333 0.333 0.333 0.111 0.111 0.111 0.333 0.111 0.111 0.111
     0.333 0.111 0.111 0.111]
    [1.  0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2]
    [1. 3. 3. 3. 3. 9. 9. 9. 3. 9. 9. 9. 3. 9. 9. 9.]

For example, ``XX`` is the entry :math:`1 \cdot 4 + 1 = 5`, with eigenvalue :math:`1/9` for local
Cliffords, and :math:`1/5` for global ones. With ``basis="computational"``, the full matrix that acts on
the row-vectorized density matrix is returned instead:

.. code-block:: python

    print(shadow.frame_operator(2, "local-clifford", basis="computational").shape)

.. code-block:: text

    (16, 16)


Statistics, memory, and reproducibility
---------------------------------------

**Shots.** ``nshots`` is the number of shots per random unitary. In general, many unitaries with
few shots are better than few unitaries with many shots.

**Median of means.** Averages are sensitive to rare, very large snapshot estimates. With ``nbatches``,
the ``nsamples`` unitaries are split into that many batches of (almost) equal size, one estimate is
computed for each batch, and the median of those is returned. With the default ``nbatches=1`` this is the
usual empirical average:

.. code-block:: python

    for nbatches in (1, 5, 25):
        estimate = shadow(
            circuit, observable, 1000, "local-clifford", nbatches=nbatches, seed=1
        )
        print(f"nbatches={nbatches:2d}  {float(estimate):.3f}")

.. code-block:: text

    nbatches= 1  0.990
    nbatches= 5  1.026
    nbatches=25  0.981

The median is slightly biased, and many batches have few samples each, so use it
when single estimates have heavy tails, not as a default. ``nbatches`` must be between 1 and ``nsamples``.

**Memory.** Circuits are generated and executed in chunks of at most ``chunksize`` circuits
(1000 by default), which are discarded once processed. Each batch is also turned into its estimate as
soon as it is complete. The memory used by the circuits is thus set by ``chunksize`` and not by ``nsamples``:

.. code-block:: python

    print(f"{float(shadow(circuit, observable, 1000, 'local-clifford', nshots=5, chunksize=100, seed=1)):.3f}")

.. code-block:: text

    1.008

A smaller ``chunksize`` lowers the memory, but makes the run slightly slower when chunks are very small.
In our tests with four qubits and 2000 samples, going from a single chunk to chunks of 200 circuits reduced
the peak memory (of Python allocations) from 33 MB to 3.5 MB at the same run time.

**Reproducibility.** The same ``seed`` gives the same estimate. One seed per chunk is drawn from it,
so also changing ``chunksize`` changes the random unitaries:

.. code-block:: python

    a = shadow(circuit, observable, 200, "local-clifford", seed=7)
    b = shadow(circuit, observable, 200, "local-clifford", seed=7)
    c = shadow(circuit, observable, 200, "local-clifford", seed=8)
    print(float(a) == float(b), float(a) == float(c))

.. code-block:: text

    True False


.. _classical-shadows-ultra-shallow:

Ultra-shallow shadows
---------------------

``method="ultra-shallow"`` implements the ensemble of Ref. [3]. A measurement circuit has
``depth + 1`` layers of random single-qubit Cliffords. A layer of CNOT gates comes
before each layer but the first. It is the ``"shifted"`` architecture of
:func:`qibo.models.encodings.entangling_layer`: CNOT gates on the pairs of qubits
``(0, 1), (2, 3), ...``, followed by those on ``(1, 2), (3, 4), ...``.
Four qubits and ``depth=2`` look like this:

.. code-block:: python

    shadow.circuits(Circuit(4), 1, "ultra-shallow", depth=2, seed=3)[0].draw()

.. code-block:: text

    0: ─S─H─S─S─S─o─────S─H───────o───────────M─
    1: ─S─────────X───o─S─H─S─S─S─X───o─S─────M─
    2: ─S─H─────────o─X─S─S─H─S─S───o─X─H─S─S─M─
    3: ─S─S─────────X───H─S─S─S─────X───S─S─S─M─

With ``depth=0`` there are no entangling layers, and the ensemble is ``"local-clifford"``. Larger depths
give more entangled measurement bases, at a circuit depth set by ``depth`` and not by the number of qubits.

**The frame has to be sampled.** For ``depth > 0`` there is no closed formula for :math:`f_{P}`. But
the ensemble is invariant under single-qubit Cliffords, so :math:`f_{P}` only depends on the support of
:math:`P`, written as a bitstring :math:`k \in \{0, 1\}^{n}` with a one on each qubit where :math:`P` is not
the identity, and it can be estimated by running the same random circuits
on the zero state :math:`\ket{0}^{\otimes n}`. If :math:`g` is the random unitary, :math:`z` the outcome, and
:math:`Z^{k}` the product of Pauli :math:`Z` matrices on the qubits of the support :math:`k`,

.. math::
    f_{k} = \mathbb{E}_{g} \sum_{z} \bra{0} g^{\dagger} \ketbra{z}{z} g \ket{0} \,
    \bra{z} g Z^{k} g^{\dagger} \ket{z} \, ,

which the pipeline does with ``ncalibration`` circuits (``nsamples`` if ``None``), whose seed is
``seed + 1`` so as to be independent of the data. You can also call it directly:

.. code-block:: python

    ghz = Circuit(3)
    ghz.add(gates.H(0))
    ghz.add(gates.CNOT(0, 1))
    ghz.add(gates.CNOT(1, 2))

    frame = shadow.frame_operator(3, "ultra-shallow", depth=1, nsamples=4000, seed=1)
    local = shadow.frame_operator(3, "local-clifford")

    print("support  ultra-shallow  local-clifford")
    for support in range(1, 8):
        bits = format(support, "03b")
        index = int(bits.replace("1", "3"), 4)  # string with Z on the support
        print(f"  {bits}      {frame[index]:.3f}          {local[index]:.3f}")

.. code-block:: text

    support  ultra-shallow  local-clifford
      001      0.178          0.333
      010      0.089          0.333
      011      0.112          0.111
      100      0.144          0.333
      101      0.066          0.111
      110      0.166          0.111
      111      0.110          0.037

Note that, unlike for local Cliffords, the eigenvalues of high-weight strings are not much smaller than
those of low-weight ones, which lowers the variance for high-weight observables.
Each eigenvalue is the average of numbers between :math:`-1` and :math:`1`, so its error with :math:`N`
calibration circuits is at most about :math:`1/\sqrt{N}`, which is large relative to eigenvalues of
a few percent. Use enough calibration circuits (``ncalibration`` larger than ``nsamples`` is often
needed), as the estimate divides by these numbers. If a sampled eigenvalue is not positive, the inverse
frame cannot be built, and a ``ValueError`` asks for more circuits.

Here is the fidelity of the 3-qubit GHZ (Greenberger–Horne–Zeilinger) state :math:`(\ket{000} + \ket{111})/\sqrt{2}`,
which is the mean of the 8 operators of its stabilizer group (the Pauli strings that leave it unchanged):

.. code-block:: python

    ghz_fidelity = {"III": 1, "ZZI": 1, "IZZ": 1, "ZIZ": 1, "XXX": 1, "XYY": -1, "YXY": -1, "YYX": -1}
    ghz_fidelity = {string: coefficient / 8 for string, coefficient in ghz_fidelity.items()}

    for method, kwargs in (
        ("local-clifford", {}),
        ("ultra-shallow", {"depth": 1, "ncalibration": 4000}),
    ):
        estimate = shadow(ghz, ghz_fidelity, 1500, method, chunksize=500, seed=2, **kwargs)
        print(f"{method:16s} {float(estimate):.3f}")

.. code-block:: text

    local-clifford   1.049
    ultra-shallow    0.977

The exact fidelity is :math:`1`. Both methods are consistent with it here; their difference in variance only shows
up when repeating the experiment over many seeds.


Circuit noise
-------------

Since the frame of ``"ultra-shallow"`` is sampled from the same circuits as the data, the calibration can be
done on the *noisy* circuits. The sampled frame then describes the noisy measurement, and the estimator remains
unbiased, which is the robustness of Ref. [3]. This needs the noise to affect the calibration and measurement
circuits alike, and not the state preparation, whose noise cannot be told apart from a different state.

To try it, we execute the circuits on a backend that adds noise. The following backend applies a
:class:`qibo.noise.NoiseModel` to the random circuits, but not to the gates of the state preparation, and can
switch the noise off for the calibration circuits, which have no state preparation:

.. code-block:: python

    from qibo.backends import NumpyBackend
    from qibo.noise import DepolarizingError, NoiseModel


    class NoisyBackend(NumpyBackend):
        """Noise on the measurement circuits, but not on the state preparation."""

        def __init__(self, noise_model, preparation, noisy_calibration=True):
            super().__init__()
            self.noise_model = noise_model
            self.ideal = {id(gate) for gate in preparation.queue}
            self.noisy_calibration = noisy_calibration

        def execute_circuit(self, circuit, initial_state=None, nshots=1000):
            if not circuit.measurements:
                return super().execute_circuit(circuit, initial_state, nshots)
            queue = circuit.queue
            nideal = 0  # the leading gates that are the state preparation
            while nideal < len(queue) and id(queue[nideal]) in self.ideal:
                nideal += 1
            if nideal == 0 and not self.noisy_calibration:
                return super().execute_circuit(circuit, initial_state, nshots)
            random_part = Circuit(circuit.nqubits)
            random_part.add(queue[nideal:-1])
            noisy = Circuit(circuit.nqubits)
            noisy.add(queue[:nideal])
            noisy.add(self.noise_model.apply(random_part).queue)
            noisy.add(queue[-1])
            return super().execute_circuit(noisy, initial_state, nshots)


    noise = NoiseModel()
    noise.add(DepolarizingError(0.02), gates.H)
    noise.add(DepolarizingError(0.02), gates.S)
    noise.add(DepolarizingError(0.15), gates.CNOT)

We estimate the GHZ fidelity with noise in the measurement circuits, calibrating the frame either on noisy
circuits or on noiseless ones:

.. code-block:: python

    kwargs = dict(
        nsamples=1500, method="ultra-shallow", depth=1, ncalibration=4000, chunksize=500, seed=2
    )
    for label, noisy_calibration in (
        ("noisy calibration", True),
        ("noiseless calibration", False),
    ):
        backend = NoisyBackend(noise, ghz, noisy_calibration)
        estimate = shadow(ghz, ghz_fidelity, backend=backend, **kwargs)
        print(f"{label:22s} {float(estimate):.3f}")

.. code-block:: text

    noisy calibration      1.024
    noiseless calibration  0.575

Ignoring the noise in the calibration underestimates the fidelity, even though the state preparation is
perfect. The sampled frames show why: noise shrinks the eigenvalues, so dividing the data by the noiseless ones
does not undo the damping.

.. code-block:: python

    ideal_frame = shadow.frame_operator(3, "ultra-shallow", depth=1, nsamples=4000, seed=3)
    noisy_frame = shadow.frame_operator(
        3, "ultra-shallow", depth=1, nsamples=4000, seed=3, backend=NoisyBackend(noise, Circuit(3))
    )

    print("support  noiseless  noisy")
    for support in range(1, 8):
        bits = format(support, "03b")
        index = int(bits.replace("1", "3"), 4)
        print(f"  {bits}      {ideal_frame[index]:.3f}     {noisy_frame[index]:.3f}")

.. code-block:: text

    support  noiseless  noisy
      001      0.182     0.134
      010      0.085     0.048
      011      0.109     0.059
      100      0.138     0.097
      101      0.066     0.038
      110      0.161     0.102
      111      0.099     0.051

The price is a larger variance, as the estimates are divided by smaller numbers. In a test with this noise model
and a weighted sum of Pauli strings of the GHZ state, the spread of the estimates over seeds
was roughly two to three times larger than without noise.

On a real device, pass the backend of the device: the noise is then real, and the calibration is
run on the same device with no changes to the code.


Writing your own protocol
-------------------------

:class:`qibo.tomography.classical_shadows.ClassicalShadow` derives from the abstract class
:class:`qibo.tomography.abstract.Tomography`. A protocol implements ``circuits``, which builds the circuits to
execute, and ``__call__``, which runs the whole pipeline, and inherits ``execute``, which runs circuits on a
backend, and ``_check_circuit``, which rejects circuits that already contain measurements. For instance, a
protocol that only measures in the computational basis:

.. code-block:: python

    from qibo.tomography import Tomography


    class ComputationalBasis(Tomography):
        """Measures all qubits in the computational basis."""

        def __call__(self, circuit, nshots=1000, backend=None):
            (result,) = self.execute(self.circuits(circuit), nshots, backend)
            return result.frequencies()

        def circuits(self, circuit):
            self._check_circuit(circuit)
            measured = circuit.copy()
            measured.add(gates.M(*range(circuit.nqubits)))
            return [measured]


    print(ComputationalBasis()(circuit, nshots=1000))

.. code-block:: text

    Counter({'00': 509, '11': 491})


Summary of the options
----------------------

.. list-table::
    :header-rows: 1
    :widths: 20 20 60

    * - Argument
      - Default
      - Meaning
    * - ``nsamples``
      - (required)
      - number of random unitaries, i.e. of circuits executed
    * - ``method``
      - ``"global-clifford"``
      - ensemble: ``"global-clifford"``, ``"local-clifford"`` or ``"ultra-shallow"``
    * - ``basis``
      - ``"pauli"``
      - basis where the frame is applied. The observable is a list or dictionary of Pauli strings
        and coefficients for ``"pauli"``, and a Hamiltonian or a matrix for ``"computational"``
    * - ``nshots``
      - ``1``
      - shots per random unitary
    * - ``nbatches``
      - ``1``
      - number of batches of the median of means; ``1`` is the empirical average
    * - ``chunksize``
      - ``1000``
      - maximum number of circuits generated and executed at once, which bounds their memory
    * - ``depth``
      - ``1``
      - number of entangling layers of ``"ultra-shallow"``; ``0`` is local Cliffords
    * - ``ncalibration``
      - ``None``
      - number of circuits that sample the frame of ``"ultra-shallow"``; ``None`` means ``nsamples``
    * - ``seed``
      - ``None``
      - seed of the random unitaries; the calibration uses ``seed + 1``
    * - ``backend``
      - ``None``
      - backend that executes the circuits; ``None`` is the current backend


Limitations
-----------

* The average snapshot is a dense :math:`2^{n} \times 2^{n}` matrix, so memory grows as :math:`4^{n}`,
  and the cost per sample as well. The implementation targets systems of a few qubits, and
  the computational basis, which builds a :math:`4^{n} \times 4^{n}` matrix, even fewer.
* The circuits of ``"global-clifford"`` are deep and are generated with
  :func:`qibo.quantum_info.random_clifford`.
* Single-qubit layers of ``"ultra-shallow"`` are Cliffords, not Haar-random unitaries, so all
  measurement circuits are Clifford circuits.
* The noise robustness requires the calibration and the measurement circuits to be noisy in the same
  way, as discussed in Ref. [3]. The noisy backend above is an illustration, not part of Qibo.


References
----------

1. H.-Y. Huang, R. Kueng and J. Preskill,
   *Predicting many properties of a quantum system from very few measurements* (2020),
   `Nature Physics 16, 1050 <https://doi.org/10.1038/s41567-020-0932-7>`_.

2. L. Innocenti *et al.*, *Shadow tomography on general measurement frames* (2023),
   `PRX Quantum 4, 040328 <https://doi.org/10.1103/PRXQuantum.4.040328>`_.

3. R. M. S. Farias, R. D. Peddinti, I. Roth and L. Aolita,
   *Robust ultra-shallow shadows*,
   `Quantum Science and Technology 10, 025044 <https://doi.org/10.1088/2058-9565/adc14f>`_.
