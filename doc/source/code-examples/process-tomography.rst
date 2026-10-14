.. _process-tomography-tutorial:

Process tomography tutorial
===========================

Quantum process tomography characterizes what a circuit does to the state of its qubits.
In this tutorial we use :class:`qibo.tomography.process_tomography.ProcessTomography`, which
implements the vanilla protocol: input states, :ref:`state tomography <state-tomography-tutorial>`
of the outputs, and linear inversion. We go through:

* the idea, and the circuits that are executed;
* the estimate of a gate, and its properties;
* a noisy process, and the other representations;
* how the error depends on the number of shots, and how the cost depends on the number of qubits.

All the outputs below were obtained with the seeds shown, on the ``numpy`` backend.
Estimates are random, so different seeds, or versions of the libraries, give different digits.


The idea
--------

A process :math:`\mathcal{E}` is described by its *Pauli transfer matrix*,

.. math::
    R_{ij} = \frac{1}{2^{n}} \text{tr}\left(P_{i} \, \mathcal{E}(P_{j})\right) \, ,

where :math:`P_{i}` are the :math:`4^{n}` Pauli strings of :math:`n` qubits. It is a real matrix
that maps the Pauli coefficients :math:`s_{j} = \text{tr}(P_{j} \, \rho)` of any state to those of
its output: :math:`o_{i} = \sum_{j} R_{ij} \, s_{j}`. Thus, :math:`R` is known once the outputs
of :math:`4^{n}` inputs with linearly independent coefficients are known. We take the states in
which each qubit is in :math:`|0\rangle`, :math:`|1\rangle`, :math:`|+\rangle`, or
:math:`|{+i}\rangle`, and we reconstruct each output with the state tomography of Pauli
measurements, which needs :math:`3^{n}` settings. Collecting the coefficients of the inputs in the
rows of :math:`S`, and those of the outputs in the rows of :math:`O`, we have :math:`O = S R^{T}`,
so the estimate is :math:`\hat{R} = (S^{-1} O)^{T}`.


The circuits
------------

``circuits`` prepends each input state to the circuit, and appends the rotations and the
measurements of each setting. As an example, we take the Hadamard gate:

.. code-block:: python

    from qibo import Circuit, gates, set_backend, get_backend
    from qibo.tomography import ProcessTomography

    set_backend("numpy")
    backend = get_backend()

    circuit = Circuit(1)
    circuit.add(gates.H(0))

    tomography = ProcessTomography()
    circuits = tomography.circuits(circuit)
    print(len(circuits))

.. code-block:: text

    12

There are :math:`4 \times 3` circuits: the input states change the slowest, in the order
:math:`|0\rangle, |1\rangle, |+\rangle, |{+i}\rangle`, and for each of them there are the
:math:`3^{n}` settings of the state tomography, in their order. The first circuit has the input
:math:`|0\rangle`, i.e. no gate, and measures the output in :math:`X`, with a Hadamard gate
before the measurement. The last one has the input :math:`|{+i}\rangle`, which is prepared with
the gates ``H`` and ``S``, and measures :math:`Z`:

.. code-block:: python

    circuits[0].draw()

.. code-block:: text

    0: ─H─H─M─

.. code-block:: python

    circuits[11].draw()

.. code-block:: text

    0: ─H─S─H─M─


The estimated process
---------------------

Calling the object runs the whole pipeline. We estimate the Pauli transfer matrix of the Hadamard
gate, in the basis :math:`I, X, Y, Z`:

.. code-block:: python

    estimate = tomography(circuit, nshots=1000, seed=1, backend=backend)
    print(backend.round(estimate, 3))

.. code-block:: text

    [[ 1.     0.     0.     0.   ]
     [ 0.     0.04  -0.014  1.   ]
     [-0.018  0.034 -0.982 -0.04 ]
     [-0.016  1.016 -0.004  0.062]]

The Hadamard gate exchanges :math:`X` and :math:`Z`, and flips the sign of :math:`Y`, so the ideal
matrix has the entries :math:`R_{XZ} = R_{ZX} = 1` and :math:`R_{YY} = -1`, which are close to the
estimate. The first row, :math:`(1, 0, 0, 0)`, is exact for any number of shots, because the
process preserves the trace. The process fidelity with the ideal gate is
:math:`\text{tr}(R^{T}_{\text{ideal}} \hat{R}) / 4^{n}`:

.. code-block:: python

    ideal = backend.cast(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0]], dtype=backend.float64
    )
    print(f"{backend.trace(ideal.T @ estimate) / 4:.3f}")

.. code-block:: text

    1.000

The estimate is **not** completely positive in general. The ideal Choi matrix of a unitary has
a single non-zero eigenvalue, and the statistical errors make the others non-zero, and some of
them negative:

.. code-block:: python

    choi = tomography(circuit, nshots=1000, representation="choi", seed=1, backend=backend)
    print(backend.round(backend.eigvalsh((choi + backend.dagger(choi)) / 2), 3))

.. code-block:: text

    [-0.057 -0.004  0.061  2.   ]

Because of this, the process fidelity of an estimate can even be larger than :math:`1`.
If a physical process is needed, the estimate has to be projected onto the completely
positive and trace preserving maps afterwards, which is not done by this protocol.


A noisy process
---------------

The circuit can have noise channels, and it can be a circuit of density matrices. As an example,
we take an amplitude damping channel of strength :math:`\gamma = 0.3`, as a Kraus channel:

.. code-block:: python

    import math

    gamma = 0.3
    damping = [
        backend.cast([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=backend.float64),
        backend.cast([[0, math.sqrt(gamma)], [0, 0]], dtype=backend.float64),
    ]

    noisy_circuit = Circuit(1, density_matrix=True)
    noisy_circuit.add(gates.KrausChannel(0, damping))

    estimate = tomography(noisy_circuit, nshots=20000, seed=1, backend=backend)
    print(backend.round(estimate, 3))

.. code-block:: text

    [[ 1.     0.     0.     0.   ]
     [-0.007  0.848  0.021 -0.005]
     [ 0.009 -0.02   0.835  0.001]
     [ 0.297 -0.001 -0.     0.703]]

The ideal matrix is diagonal, with :math:`\sqrt{1 - \gamma} = 0.837` for :math:`X` and :math:`Y`,
which the channel shrinks, and :math:`1 - \gamma = 0.7` for :math:`Z`. The entry
:math:`R_{ZI} = \gamma = 0.3` is the part of the process that is not unital, i.e. that does not
map the identity to itself.

The matrix is the Pauli-Liouville representation of :mod:`qibo.quantum_info`, in the normalized
Pauli basis. The argument ``representation`` returns the process in the Liouville, with
``"liouville"``, or Choi, with ``"choi"``, representation, with the row vectorization, as shown
above. They are the ones of :func:`qibo.quantum_info.kraus_to_liouville`, and
:func:`qibo.quantum_info.kraus_to_choi`, for the same process.


Shots and qubits
----------------

The statistical error decreases as :math:`1 / \sqrt{N}` with the number of shots :math:`N` of each
circuit. For the Hadamard gate, the largest error of an entry and the Frobenius norm of the error
are:

.. code-block:: python

    for nshots in (100, 1000, 10000, 100000):
        estimate = tomography(circuit, nshots=nshots, seed=1, backend=backend)
        error = estimate - ideal
        print(f"{nshots:>6}  {backend.abs(error).max():.3f}  {backend.matrix_norm(error):.3f}")

.. code-block:: text

       100  0.110  0.184
      1000  0.062  0.098
     10000  0.016  0.033
    100000  0.003  0.006

The number of circuits grows as :math:`12^{n}`: for :math:`n = 1, 2, 3, 4, 5` there are
:math:`12, 144, 1728, 20736, 248832` circuits. The error also grows with :math:`n`, because the
inversion of the inputs amplifies the statistical errors of the outputs, and because a Pauli string
of many qubits is measured by fewer settings. For the CNOT gate on two qubits, with :math:`144`
circuits, the largest error of an entry is:

.. code-block:: python

    cnot = Circuit(2)
    cnot.add(gates.CNOT(0, 1))

    print(len(tomography.circuits(cnot)))

.. code-block:: text

    144

.. code-block:: text

    nshots  largest error
       100  0.450
      1000  0.133
     10000  0.040

so about :math:`10^{4}` shots per circuit are needed for an error of a few percent. This is why the
protocol is meant for few qubits.


Limitations
-----------

* The estimate is not completely positive in general, as shown above.
* The state preparation and measurement errors of the device are not corrected: they are part of
  the reconstructed process.
* The cost, :math:`12^{n}` circuits, and the error make the protocol impractical beyond a few
  qubits.

The circuits are run with :meth:`qibo.tomography.abstract.Tomography.execute`, with the backend
given in the argument ``backend`` of the call, so a hardware backend, or a backend that simulates
noise, is used without changes to the code. The protocol is an implementation of the abstract class
:class:`qibo.tomography.abstract.Tomography`.
