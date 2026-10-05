.. _Configuration:

Configuration
=============

The :func:`set_max_qubits` and :func:`set_max_qubits_dm` functions control the
maximum number of qubits allowed for state vector and density matrix
allocations, respectively. These limits help prevent uncontrolled memory
consumption; memory scales as ``2^n * 16`` bytes (complex128) for state vectors
and ``4^n * 16`` bytes (complex128) for density matrices. The corresponding
constant values can be accessed via :func:`get_max_qubits` and
:func:`get_max_qubits_dm` functions. Both functions can also be configured via
environment variables ``QIBO_MAX_QUBITS`` and ``QIBO_MAX_QUBITS_DM``
respectively.

.. code-block::  python

    import qibo

    # Set maximum to 20 qubits for state vectors
    qibo.set_max_qubits(20)

    # Get current maximum
    max_qubits = qibo.get_max_qubits()

    # Set maximum to 15 qubits for density matrices
    qibo.set_max_qubits_dm(15)

    # Get current maximum for density matrices
    max_qubits_dm = qibo.get_max_qubits_dm()
