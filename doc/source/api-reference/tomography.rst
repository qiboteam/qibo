Tomography
==========

Classes and functions used to classically simulate tomography protocols.


Abstract class
--------------

Tomographic protocols that build, execute, and post-process circuits derive from the
following abstract class.

.. autoclass:: qibo.tomography.abstract.Tomography
    :members:
    :member-order: bysource
    :special-members: __call__


.. _StateTomography:

State Tomography
----------------

State tomography reconstructs the density matrix :math:`\rho` of :math:`n` qubits from Pauli
measurements [1, 2]. In the Pauli basis,

.. math::
    \rho = \frac{1}{2^{n}} \sum_{P} \text{tr}(P \, \rho) \, P \, ,

where the sum runs over the :math:`4^{n}` Pauli strings :math:`P`. Each qubit is measured in the
basis :math:`X`, :math:`Y`, or :math:`Z`, which gives :math:`3^{n}` settings, each one executed
for ``nshots`` shots. The Walsh-Hadamard transform of the outcome probabilities of a setting
gives the expectation values of the :math:`2^{n}` Pauli strings that it measures, i.e. those
that have the measured Pauli, or the identity, on each qubit. A Pauli string with :math:`w`
non-identity factors is measured by :math:`3^{n - w}` settings, and the estimates of all of them
are averaged. The coefficients of all Pauli strings are then mapped to :math:`\hat{\rho}` with a
fast Hadamard transform.

The outcomes can be turned into a state with four estimators, selected with the argument
``method``:

* ``"linear-inversion"``, the default, inserts the estimated expectation values in the expansion
  above. The estimate is Hermitian, with unit trace, and unbiased, but it is not positive
  semidefinite in general, because of the statistical errors.
* ``"physical-linear-inversion"`` is the density matrix closest to the linear inversion
  estimate in the Frobenius norm, obtained by projecting its eigenvalues onto the probability
  simplex [3].
* ``"least-squares"`` is the physical state that minimizes the sum of the squared differences
  between the observed and the predicted probabilities, found with an accelerated projected
  gradient method [3, 6].
* ``"mle"`` is the maximum likelihood estimate, found with the diluted iterative algorithm of
  Refs. [4, 5]. It is physical, and biased towards states with low rank.

The readout errors of the device can be mitigated in all of them by giving the response matrix
of the readout in the argument ``response_matrix``, e.g. the one returned by
:func:`qibo.models.error_mitigation.get_response_matrix` [7]. The number of circuits grows as
:math:`3^{n}`, so the protocol is meant for few qubits. A step-by-step guide is given in the
:ref:`state tomography tutorial <state-tomography-tutorial>`.

References:
    1. M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*,
    Cambridge University Press (2010), Sec. 8.4.2.

    2. D. F. V. James, P. G. Kwiat, W. J. Munro and A. G. White,
    *Measurement of qubits* (2001),
    `Physical Review A 64, 052312 <https://doi.org/10.1103/PhysRevA.64.052312>`_.

    3. J. A. Smolin, J. M. Gambetta and G. Smith,
    *Efficient method for computing the maximum-likelihood quantum state from measurements with
    additive Gaussian noise* (2012),
    `Physical Review Letters 108, 070502 <https://doi.org/10.1103/PhysRevLett.108.070502>`_.

    4. Z. Hradil, *Quantum-state estimation* (1997),
    `Physical Review A 55, R1561 <https://doi.org/10.1103/PhysRevA.55.R1561>`_.

    5. A. I. Lvovsky, *Iterative maximum-likelihood reconstruction in quantum homodyne
    tomography* (2004),
    `Journal of Optics B 6, S556 <https://doi.org/10.1088/1464-4266/6/6/014>`_.

    6. A. Beck and M. Teboulle, *A fast iterative shrinkage-thresholding algorithm for linear
    inverse problems* (2009),
    `SIAM Journal on Imaging Sciences 2, 183 <https://doi.org/10.1137/080716542>`_.

    7. S. Bravyi, S. Sheldon, A. Kandala, D. C. Mckay and J. M. Gambetta,
    *Mitigating measurement errors in multiqubit experiments* (2021),
    `Physical Review A 103, 042605 <https://doi.org/10.1103/PhysRevA.103.042605>`_.


.. autoclass:: qibo.tomography.state_tomography.StateTomography
    :members:
    :member-order: bysource
    :special-members: __call__


.. _ProcessTomography:

Process Tomography
------------------

Vanilla process tomography characterizes the process :math:`\mathcal{E}` implemented by a circuit
of :math:`n` qubits by linear inversion [1, 2]. The process is described by its Pauli transfer
matrix

.. math::
    R_{ij} = \frac{1}{2^{n}} \text{tr}\left(P_{i} \, \mathcal{E}(P_{j})\right) \, ,

which maps the Pauli coefficients :math:`s_{j} = \text{tr}(P_{j} \, \rho)` of a state to those
of its output, :math:`o_{i} = \sum_{j} R_{ij} \, s_{j}`. The process is probed with the
:math:`4^{n}` product states in which each qubit is in :math:`|0\rangle`, :math:`|1\rangle`,
:math:`|+\rangle`, or :math:`|{+i}\rangle`, which are linearly independent. The output of each
of them is reconstructed with :ref:`state tomography <StateTomography>`, which needs
:math:`3^{n}` settings, so there are :math:`12^{n}` circuits. If :math:`S` and :math:`O` have
the coefficients of the inputs and of the outputs in their rows, then :math:`O = S R^{T}`, and
the estimate is :math:`\hat{R} = (S^{-1} O)^{T}`. It can also be returned in the Liouville, or
Choi, representation of :mod:`qibo.quantum_info`.

The estimate is unbiased, and it preserves the trace exactly, but it is not completely positive
in general, because of the statistical errors. The number of circuits grows as :math:`12^{n}`, so
the protocol is meant for few qubits, and the errors in the preparation of the inputs and in the
measurements are part of the reconstructed process. A step-by-step guide is given in the
:ref:`process tomography tutorial <process-tomography-tutorial>`.

References:
    1. I. L. Chuang and M. A. Nielsen, *Prescription for experimental determination of the
    dynamics of a quantum black box* (1997),
    `Journal of Modern Optics 44, 2455 <https://doi.org/10.1080/09500349708231894>`_.

    2. M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*,
    Cambridge University Press (2010), Sec. 8.4.2.


.. autoclass:: qibo.tomography.process_tomography.ProcessTomography
    :members:
    :member-order: bysource
    :special-members: __call__


Gate Set Tomography
-------------------

Gate Set Tomography (GST) is a powerful technique employed in quantum information processing
to characterize the behavior of quantum gates on quantum hardware [1, 2, 3].
The primary objective of GST is to provide a robust framework for obtaining a representation
of quantum gates within a predefined gate set when subjected to noise inherent to the
quantum hardware.

By characterizing the impact of noise on quantum gates, GST enables the identification and
quantification of errors, laying the groundwork for subsequent error mitigation strategies.
The insights gained from GST are instrumental, for instance, in setting up the necessary
parameters for Probabilistic Error Cancellation (PEC).

In practice, given a set of operators (or gates), :math:`\mathcal{O}=\{O_0, O_1, \dots, O_n\}`,
a set of initial states :math:`\{\rho_k\}`, and a set of measurement bases :math:`\{M_j\}`,
one performs GST on the :math:`l`-th operator by choosing an initial state :math:`\rho_k`,
applying the gate :math:`O_l \in \mathcal{O}`, measuring in the :math:`M_j` basis in order to
obtain the following matrix:

.. math::
   \{\tilde{O}_l\}_{jk} = \text{tr}(M_j\,O_l\,\rho_k) \, ,

which provides an estimated representation of the operator :math:`O_l` in the specific system.

This implementation makes use, in particular, of
:math:`\rho_k \in \{ \ketbra{0}{0}, \ketbra{1}{1}, \ketbra{+}{+}, \ketbra{y+}{y+} \}^{\otimes n}`
and :math:`M_j \in \{ I, X, Y, Z\}^{\otimes n}` [4], with :math:`n\in\{1,2\}` being the number of
qubits. However, :math:`\{\tilde{O}_l\}_{jk}` is not yet given in the Pauli-Liouville
representation (also known as *Pauli Transfer Matrix*). To obtain the Pauli-Liouville
representation, one needs the two matrices, described below. The matrix :math:`\tilde{g}` has its
elements :math:`\tilde{g}_{jk}` defined as

.. math::
   \tilde{g}_{jk} = \text{tr}(M_j\,\rho_k) \, ,

which is obtained by measuring the initial states :math:`\{\rho_k\}` in each basis element
:math:`\{M_j\}` without any gates' application. The *gauge matrix* :math:`T` is given by

.. math::
    T = \begin{pmatrix}
        1 & 1 & 1 & 1 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 1 \\
        1 & -1 & 0 & 0 \\
    \end{pmatrix} \, .

This is the matrix, in a common gauge, implementing a change of basis.
Therefore, the Pauli-Liouville representation can be recovered as

.. math::
    O_l^{PL} = T\,g^{-1}\,\tilde{O_l}\,T^{-1} \, .

References:
    1. R. Blume-Kohout *et al*.
    *Robust, self-consistent, closed-form tomography of quantum logic gates on a trapped ion qubit*
    (2013), `arXiv:1310.4492 <https://arxiv.org/abs/1310.4492>`_.

    2. D. Greenbaum, *Introduction to quantum gate set tomography* (2015),
    `arXiv:1509.02921 <https://arxiv.org/abs/1509.02921>`_.

    3. E. Nielsen *et al.*, *Gate set tomography* (2021),
    `Quantum 5, 557 <https://doi.org/10.22331/q-2021-10-05-557>`_.

    4. S. Endo, S. C. Benjamin, and Y. Li,
    *Practical quantum error mitigation for near-future applications* (2018),
    `Physical Review X 8.3: 031027 <https://doi.org/10.1103/PhysRevX.8.031027>`_.


.. autofunction:: qibo.tomography.gate_set_tomography.GST


.. _GateSetTomography:

Gate Set Tomography of a circuit
--------------------------------

The class :class:`qibo.tomography.gate_set_tomography.GateSetTomography`, built on
:class:`qibo.tomography.abstract.Tomography`, performs the Gate Set Tomography above
on a circuit that implements the operation to be characterized.

In practice, given a circuit implementing an operation :math:`O` on :math:`n \in \{1, 2\}` qubits,
a set of initial states :math:`\{\rho_{k}\}`, and a set of measurement bases :math:`\{M_{j}\}`,
one performs GST on :math:`O` by choosing an initial state :math:`\rho_{k}`, applying the
operation :math:`O`, and measuring in the :math:`M_{j}` basis in order to obtain the following
matrix [1]:

.. math::
   \tilde{O}_{jk} = \text{tr}(M_{j}\,O\,\rho_{k}) \, ,

which provides an estimated representation of the operation :math:`O` in the specific system.

This implementation makes use, in particular, of
:math:`\rho_{k} \in \{ \ketbra{0}{0}, \ketbra{1}{1}, \ketbra{+}{+}, \ketbra{y+}{y+} \}^{\otimes n}`
and :math:`M_{j} \in \{ I, X, Y, Z\}^{\otimes n}` [4]. The states are prepared, and the
measurement bases are implemented, by gates of the circuit itself, which makes it the linear
inversion variant of GST (LGST) of Ref. [1] (Sec. 3).
However, :math:`\tilde{O}_{jk}` is not yet given in the Pauli-Liouville
representation (also known as *Pauli Transfer Matrix*). To obtain the Pauli-Liouville
representation, one needs the two matrices, described below. The matrix :math:`\tilde{g}`, the
*Gram matrix* of the fiducial states and measurements, has its
elements :math:`\tilde{g}_{jk}` defined as

.. math::
   \tilde{g}_{jk} = \text{tr}(M_{j}\,\rho_{k}) \, ,

which is obtained by measuring the initial states :math:`\{\rho_{k}\}` in each basis element
:math:`\{M_{j}\}` without any gates' application, i.e. by running GST on an empty circuit.
Hence, it accounts for the errors of state preparation and measurement. The *gauge matrix*
:math:`T` is given by

.. math::
    T = \begin{pmatrix}
        1 & 1 & 1 & 1 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 1 \\
        1 & -1 & 0 & 0 \\
    \end{pmatrix} \, .

This is the matrix, in a common gauge, implementing a change of basis. Its columns are the ideal
fiducial states in the Pauli basis, which is the best *a priori* choice of gauge [1].
Therefore, the Pauli-Liouville representation can be recovered as

.. math::
    O^{\text{PL}} = T\,\tilde{g}^{-1}\,\tilde{O}\,T^{-1} \, .

For :math:`n = 2` qubits, the tensor product :math:`T \otimes T` is used.
The Gram matrix must be well-conditioned for its inverse not to amplify the statistical error [1].

References:
    1. E. Nielsen *et al.*, *Gate set tomography* (2021),
    `Quantum 5, 557 <https://doi.org/10.22331/q-2021-10-05-557>`_,
    `arXiv:2009.07301 <https://arxiv.org/abs/2009.07301>`_.

    2. R. Blume-Kohout *et al*.
    *Robust, self-consistent, closed-form tomography of quantum logic gates on a trapped ion qubit*
    (2013), `arXiv:1310.4492 <https://arxiv.org/abs/1310.4492>`_.

    3. D. Greenbaum, *Introduction to quantum gate set tomography* (2015),
    `arXiv:1509.02921 <https://arxiv.org/abs/1509.02921>`_.

    4. S. Endo, S. C. Benjamin, and Y. Li,
    *Practical quantum error mitigation for near-future applications* (2018),
    `Physical Review X 8.3: 031027 <https://doi.org/10.1103/PhysRevX.8.031027>`_.


.. autoclass:: qibo.tomography.gate_set_tomography.GateSetTomography
    :members:
    :member-order: bysource
    :special-members: __call__


.. _ClassicalShadows:

Classical Shadows
-----------------

Classical shadows estimate the expectation values of many observables from the outcomes of
randomized measurements [1]. Each sample applies a random unitary :math:`U` to the state
:math:`\rho`, and measures all qubits in the computational basis, returning :math:`b`.
The resulting *snapshot* is

.. math::
    \sigma = U^{\dagger} \ketbra{b}{b} U \, ,

and the average snapshot over many samples is mapped through the inverse of the
*frame operator* :math:`\mathcal{M}`, i.e. the measurement channel of the ensemble of :math:`U`,

.. math::
    \mathcal{M}(\rho) = \mathbb{E}_{U} \sum_{b} \bra{b} U \rho U^{\dagger} \ket{b} \,
    U^{\dagger} \ketbra{b}{b} U \, ,

giving :math:`\hat{\rho} = \mathcal{M}^{-1}(\bar{\sigma})`. Then, :math:`\text{tr}(O \hat{\rho})`
is an unbiased estimate of :math:`\text{tr}(O \rho)` for any observable :math:`O` [1, 2].

A step-by-step guide to all the options is given in the
:ref:`classical shadows tutorial <classical-shadows-tutorial>`.

The ensemble of :math:`U` is chosen through the argument ``method``, which defines the frame
operator. All the ensembles are locally invariant, so :math:`\mathcal{M}` is diagonal in the
Pauli basis, with :math:`\mathcal{M}(P) = f_{P} P` for each Pauli string :math:`P`:

* ``"global-clifford"``: :math:`U` is sampled from the Clifford group :math:`\text{Cl}(2^{n})`
  of :math:`n` qubits, and :math:`f_{P} = 1 / (2^{n} + 1)` for any :math:`P \neq I^{\otimes n}`.
* ``"local-clifford"``: :math:`U` is sampled from :math:`\text{Cl}(2)^{\otimes n}`,
  and :math:`f_{P} = 3^{-w}` for a Pauli string of weight :math:`w`.
* ``"ultra-shallow"``: :math:`U` has ``depth + 1`` layers of random single-qubit Cliffords,
  with a layer of CNOT gates (the ``"shifted"`` architecture of
  :func:`qibo.models.encodings.entangling_layer`) before each layer but the first [3].
  For ``depth=0`` it is ``"local-clifford"``. For larger depths, :math:`f_{P}` depends
  only on the support :math:`k \in \{0, 1\}^{n}` of :math:`P`, and has no analytical form.
  It is sampled instead as

  .. math::
      f_{k} = \mathbb{E}_{g} \sum_{z} \bra{0} g^{\dagger} \ketbra{z}{z} g \ket{0} \,
      \bra{z} g Z^{k} g^{\dagger} \ket{z} \, ,

  with :math:`Z^{k} = \bigotimes_{l} Z^{k_{l}}`, that is, from the same random circuits
  :math:`g` applied to the zero state, whose number is given by ``ncalibration``.
  Hence, if the circuits are noisy, the sampled frame includes the noise [3].

The frame operator is applied in the Pauli basis (default) or in the computational basis,
selected by the argument ``basis``. In the Pauli basis, the observable must be given as the
Pauli strings and coefficients :math:`O = \sum_{P} \alpha_{P} P`, and the estimate is

.. math::
    \hat{o} = \sum_{P} \alpha_{P} \, f_{P}^{-1} \, \text{tr}(P \bar{\sigma}) \, ,

where the coefficients :math:`\text{tr}(P \bar{\sigma})` of all Pauli strings are obtained
with a fast Hadamard transform, and only the diagonal of the frame is needed.
In the computational basis, the observable is a Hamiltonian or a matrix, and
:math:`\mathcal{M}^{-1}` is a matrix.

If ``robust=True``, the frame of ``"local-clifford"`` is also sampled from ``ncalibration``
circuits applied to the zero state, in the same way as for ``"ultra-shallow"``, instead of taking
the analytical eigenvalues :math:`3^{-w}`. The sampled frame includes the noise of the measurement
circuits, so the estimator remains unbiased under noise that is the same in the calibration and the
measurement circuits, which is the robust shadow estimation of Ref. [4]. The frame of
``"global-clifford"`` has no robust version.

The method :meth:`~qibo.tomography.classical_shadows.ClassicalShadow.density_matrix` returns the
estimated state :math:`\hat{\rho}` itself, instead of the expectation value of an observable. It is
Hermitian, with unit trace and unbiased, but not positive semidefinite in general.

The samples can be split into batches to use the median-of-means estimator [1],
which is the empirical average for a single batch. The circuits are generated and executed in
chunks, which bounds the memory they use, and each batch is processed as soon as it is complete.

References:
    1. H.-Y. Huang, R. Kueng and J. Preskill,
    *Predicting many properties of a quantum system from very few measurements* (2020),
    `Nature Physics 16, 1050 <https://doi.org/10.1038/s41567-020-0932-7>`_.

    2. L. Innocenti *et al.*, *Shadow tomography on general measurement frames* (2023),
    `PRX Quantum 4, 040328 <https://doi.org/10.1103/PRXQuantum.4.040328>`_.

    3. R. M. S. Farias, R. D. Peddinti, I. Roth and L. Aolita,
    *Robust ultra-shallow shadows*,
    `Quantum Science and Technology 10, 025044 <https://doi.org/10.1088/2058-9565/adc14f>`_.

    4. S. Chen, W. Yu, P. Zeng and S. T. Flammia, *Robust shadow estimation* (2021),
    `PRX Quantum 2, 030348 <https://doi.org/10.1103/PRXQuantum.2.030348>`_.


.. autoclass:: qibo.tomography.classical_shadows.ClassicalShadow
    :members:
    :member-order: bysource
    :special-members: __call__
