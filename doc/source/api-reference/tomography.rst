Tomography
==========

Functions used to classically simulate tomography protocols.


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
:math:`\rho_k \in \{ \ketbra{0}{0}, \ketbra{1}{1}, \ketbra{+}{+}, \ketbra{y+}{y+} \}^{\otimes n}` and
:math:`M_j \in \{ I, X, Y, Z\}^{\otimes n}` [4], with :math:`n\in\{1,2\}`
being the number of qubits. However, :math:`\{\tilde{O}_l\}_{jk}` is not yet given in
the Pauli-Liouville representation (also known as *Pauli Transfer Matrix*).
To obtain the Pauli-Liouville representation, one needs the two matrices, described below.
The matrix :math:`\tilde{g}` has its elements :math:`\tilde{g}_{jk}` defined as

.. math::
   \tilde{g}_{jk} = \text{tr}(M_j\,\rho_k) \, ,

which is obtained by measuring the initial states :math:`\{\rho_k\}` in each basis element :math:`\{M_j\}`
without any gates' application.
The *gauge matrix* :math:`T` is given by

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
