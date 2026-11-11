.. _Algorithms:

Algorithms
==========

Qibo provides circuit constructions of quantum algorithms. They are implemented in
``qibo/models/algorithms``, and all of them are available from ``qibo.models``, *e.g.*
``from qibo.models import QFT``.

Fourier transform and phase estimation
--------------------------------------

Quantum Fourier Transform (QFT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.models.QFT
    :members:
    :member-order: bysource


Quantum Phase Estimation (QPE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: qibo.models.QPE


Iterative Quantum Phase Estimation (IQPE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: qibo.models.IQPE


Amplitude amplification and search
----------------------------------

Grover's Algorithm
^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.models.Grover
    :members:
    :member-order: bysource


Quantum Amplitude Amplification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Quantum Amplitude Amplification (QAA) generalizes Grover's algorithm to an arbitrary circuit
:math:`\mathcal{A}` that prepares the state :math:`|\Psi\rangle = \mathcal{A}|0\rangle`
from the all-zero state :math:`|0\rangle`. Given a Boolean function :math:`\chi`, which marks the
"good" computational basis states, let :math:`a` be the probability of measuring a good state in
:math:`|\Psi\rangle`. Repeating the circuit and measuring it takes :math:`O(1 / a)` runs on
average to find a good state, whereas amplification needs only :math:`O(1 / \sqrt{a})` applications
of :math:`\mathcal{A}` and its inverse :math:`\mathcal{A}^{-1}`. This is done by applying the
operator

.. math::
    \mathbf{Q} = -\mathcal{A} \, \mathbf{S}_{0} \, \mathcal{A}^{-1} \, \mathbf{S}_{\chi},

where :math:`\mathbf{S}_{\chi}` changes the sign of the amplitude of the good states and
:math:`\mathbf{S}_{0}` changes the sign of the amplitude of the state :math:`|0\rangle`. After
:math:`m` applications, the probability of measuring a good state is
:math:`\sin^{2}((2m + 1)\theta_{a})`, where :math:`a = \sin^{2}(\theta_{a})`. Depending on
what is known about :math:`a`, the following can be used:

* :func:`qibo.models.amplification_operator` builds the circuit of :math:`\mathbf{Q}`;
* :func:`qibo.models.QAA` builds the circuit :math:`\mathbf{Q}^{m}\mathcal{A}`, either with a
  chosen number :math:`m` of iterations, or with the number that is computed from a known
  :math:`a`, which can also find a good state with certainty;
* :func:`qibo.models.FPQAA` builds a fixed-point version, which only needs a lower bound of
  :math:`a`, and whose failure probability stays below a chosen bound if more iterations are
  applied, while keeping the quadratic speedup;
* :func:`qibo.models.OAA` builds an oblivious version, which amplifies the amplitude with
  which a circuit implements a unitary on an input state that is unknown, so that the reflection
  about that state, which is not available, is not needed;
* :func:`qibo.models.QSearch` finds a good state when :math:`a` is unknown, by running circuits
  with an exponentially increasing number of iterations.

.. autofunction:: qibo.models.amplification_operator

.. autofunction:: qibo.models.FPQAA

.. autofunction:: qibo.models.OAA

.. autofunction:: qibo.models.QAA

.. autofunction:: qibo.models.QSearch


Quantum Rejection Sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: qibo.models.QRS


Quantum Minimum Finding
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: qibo.models.DH


Amplitude estimation
--------------------

Quantum Amplitude Estimation (QAE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: qibo.models.QAE


Quantum Counting
^^^^^^^^^^^^^^^^

.. autofunction:: qibo.models.Counting


Maximum-likelihood Quantum Amplitude Estimation (MLQAE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: qibo.models.MLQAE


Iterative Quantum Amplitude Estimation (IQAE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.models.IQAE
    :members:
    :member-order: bysource

.. autoclass:: qibo.models.IterativeAmplitudeEstimationResult
    :members:
    :member-order: bysource


Quantum Monte Carlo
^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.models.QMC
    :members:
    :member-order: bysource


Oracle algorithms
-----------------

Deutsch-Jozsa
^^^^^^^^^^^^^^

.. autofunction:: qibo.models.DeutschJozsa


Bernstein-Vazirani
^^^^^^^^^^^^^^^^^^

.. autofunction:: qibo.models.BernsteinVazirani


Quantum arithmetic
------------------

Quantum Adders
^^^^^^^^^^^^^^

.. autofunction:: qibo.models.Adder

.. autofunction:: qibo.models.Cuccaro

.. autofunction:: qibo.models.DKRS

.. autofunction:: qibo.models.Draper

.. autofunction:: qibo.models.Gidney

.. autofunction:: qibo.models.RV

.. autofunction:: qibo.models.TTK

.. autofunction:: qibo.models.VBE


Factoring
---------

Shor's Algorithm
^^^^^^^^^^^^^^^^

.. autoclass:: qibo.models.Shor
    :members:
    :member-order: bysource


Block encodings and signal processing
-------------------------------------

Block Encoding
^^^^^^^^^^^^^^

.. autofunction:: qibo.models.block_encoding_circuit


Linear Combination of Unitaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: qibo.models.lcu_circuit

.. autofunction:: qibo.models.lcu_prepare

.. autofunction:: qibo.models.lcu_select


Quantum Signal Processing
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: qibo.models.qsp_phases

.. autofunction:: qibo.models.qsp_circuit


Quantum Singular Value Transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: qibo.models.qsvt_phases

.. autofunction:: qibo.models.qsvt_circuit


Hamiltonian diagonalization
---------------------------

Double-Bracket Iterations
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.models.DoubleBracketIteration
    :members:
    :member-order: bysource
