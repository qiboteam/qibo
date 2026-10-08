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


.. autoclass:: qibo.tomography.classical_shadows.ClassicalShadow
    :members:
    :member-order: bysource
    :special-members: __call__
