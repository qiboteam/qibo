.. _Hamiltonians:

Hamiltonians
============

The main abstract Hamiltonian object of Qibo is:

.. autoclass:: qibo.hamiltonians.abstract.AbstractHamiltonian
    :members:
    :member-order: bysource


Matrix Hamiltonian
------------------

The first implementation of Hamiltonians uses the full matrix representation
of the Hamiltonian operator in the computational basis.
For :math:`n` qubits, this matrix has size :math:`2^{n} \times 2^{n}`.
Therefore, its construction is feasible only when :math:`n` is small.

Alternatively, the user can construct this Hamiltonian using a sparse matrices.
Sparse matrices from the
`scipy.sparse <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_
module are supported by the ``numpy`` and ``qibojit`` backends while the
`tensorflow.sparse <https://www.tensorflow.org/api_docs/python/tf/sparse>`_ can be
used for ``tensorflow``. Scipy sparse matrices support algebraic
operations (addition, subtraction, scalar multiplication), linear algebra
operations (eigenvalues, eigenvectors, matrix exponentiation) and
multiplication to dense or other sparse matrices. All these properties are
inherited by :class:`qibo.hamiltonians.Hamiltonian` objects created
using sparse matrices. Tensorflow sparse matrices support only multiplication
to dense matrices. Both backends support calculating Hamiltonian expectation
values using a sparse Hamiltonian matrix.

.. autoclass:: qibo.hamiltonians.Hamiltonian
    :members:
    :member-order: bysource


Symbolic Hamiltonian
--------------------

Qibo allows the user to define Hamiltonians using ``sympy`` symbols. In this
case the full Hamiltonian matrix is not constructed unless this is required.
This makes the implementation more efficient for larger qubit numbers.
For more information on constructing Hamiltonians using symbols we refer to the
:ref:`How to define custom Hamiltonians using symbols? <symbolicham-example>` example.

.. autoclass:: qibo.hamiltonians.SymbolicHamiltonian
    :members:
    :member-order: bysource


When a :class:`qibo.hamiltonians.SymbolicHamiltonian` is used for time
evolution then Qibo will automatically perform this evolution using the Trotter
of the evolution operator. This is done by automatically splitting the Hamiltonian
to sums of commuting terms, following the description of Sec. 4.1 of
`arXiv:1901.05824 <https://arxiv.org/abs/1901.05824>`_.
For more information on time evolution we refer to the
:ref:`How to simulate time evolution? <timeevol-example>` example.

In addition to the abstract Hamiltonian models, Qibo provides the following
pre-coded Hamiltonians:


Non-interacting Pauli-X
-----------------------

.. autoclass:: qibo.hamiltonians.X
    :members:
    :member-order: bysource

Non-interacting Pauli-Y
-----------------------

.. autoclass:: qibo.hamiltonians.Y
    :members:
    :member-order: bysource

Non-interacting Pauli-Z
-----------------------

.. autoclass:: qibo.hamiltonians.Z
    :members:
    :member-order: bysource

Ising model
-----------

.. autoclass:: qibo.hamiltonians.Ising
    :members:
    :member-order: bysource

Transverse-field Ising model
----------------------------

.. autoclass:: qibo.hamiltonians.TFIM
    :members:
    :member-order: bysource

Max Cut
-------

.. autoclass:: qibo.hamiltonians.MaxCut
    :members:
    :member-order: bysource

LABS
----

.. autoclass:: qibo.hamiltonians.LABS
    :members:
    :member-order: bysource


Fermi-Hubbard model
-------------------

.. autoclass:: qibo.hamiltonians.FermiHubbard
    :members:
    :member-order: bysource


Heisenberg model
----------------

.. autoclass:: qibo.hamiltonians.Heisenberg
    :members:
    :member-order: bysource


Heisenberg XXX
--------------

.. autoclass:: qibo.hamiltonians.XXX
    :members:
    :member-order: bysource


Heisenberg XXZ
--------------

.. autoclass:: qibo.hamiltonians.XXZ
    :members:
    :member-order: bysource


Folded XXZ model
----------------

.. autoclass:: qibo.hamiltonians.FoldedXXZ
    :members:
    :member-order: bysource


Graph Partitioning Problem
--------------------------

.. autoclass:: qibo.hamiltonians.GPP
    :members:
    :member-order: bysource


.. note::
    All pre-coded Hamiltonians can be created as
    :class:`qibo.hamiltonians.Hamiltonian` using ``dense=True``
    or :class:`qibo.hamiltonians.SymbolicHamiltonian`
    using the ``dense=False``. In the first case the Hamiltonian is created
    using its full matrix representation of size ``(2 ** n, 2 ** n)``
    where ``n`` is the number of qubits that the Hamiltonian acts on. This
    matrix is used to calculate expectation values by direct matrix multiplication
    to the state and for time evolution by exact exponentiation.
    In contrast, when ``dense=False`` the Hamiltonian contains a more compact
    representation as a sum of local terms. This compact representation can be
    used to calculate expectation values via a sum of the local term expectations
    and time evolution via the Trotter decomposition of the evolution operator.
    This is useful for systems that contain many qubits for which constructing
    the full matrix is intractable.
