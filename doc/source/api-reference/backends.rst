.. _Backends:

Backends
========

:class:`qibo.backends.abstract.Backend` is the main calculation engine to execute circuits.
Qibo provides backends for quantum simulation on classical hardware, as well as quantum hardware
management and control. For a complete list of available backends, refer to the
:ref:`Packages <packages>` section. To create new backends, inherit from
:class:`qibo.backends.abstract.Backend` and implement its new methods.
This abstract class defines the required methods for circuit execution.


The user can set the backend using the :func:`qibo.set_backend` function.

.. code-block::  python

    import qibo
    qibo.set_backend("qibojit")

    # Switch to the numpy backend
    qibo.set_backend("numpy")


If no backend is specified, the default backend is used.
The default backend is selected in the following order: ``qibojit``, ``numpy``, and ``qiboml``.
The list of default backend candidates can be changed using the ``QIBO_BACKEND`` environment variable.

Some backends support different platforms. For example, the ``qibojit`` backend
provides two platforms (``cupy`` and ``cuquantum``) when used on GPU.
The platform can be specified using the ``platform`` parameter in the :func:`qibo.set_backend` function.

.. code-block::  python

    import qibo
    qibo.set_backend("qibojit", platform="cuquantum")

    # Switch to the cupy platform
    qibo.set_backend("qibojit", platform="cupy")

.. autoclass:: qibo.backends.abstract.Backend
    :members:
    :member-order: bysource

Clifford Simulation
-------------------

Qibo provides a :class:`qibo.backends.clifford.CliffordBackend` that
efficiently simulates Clifford circuits using the phase-space (stabilizer)
formalism. See :doc:`/concepts/clifford-simulation` for a detailed
description and worked examples.

.. autoclass:: qibo.backends.clifford.CliffordBackend
    :members:
    :member-order: bysource

Simulation of Hamming-weight-preserving circuits
-------------------------------------------------

Qibo provides a :class:`qibo.backends.hamming_weight.HammingWeightBackend`
that fast-simulates circuits preserving the Hamming weight of the state, using
a compressed representation restricted to the fixed-weight subspace. See
:doc:`/concepts/hamming-weight-simulation` for a detailed description and
worked examples.

.. autoclass:: qibo.backends.hamming_weight.HammingWeightBackend
    :members:
    :member-order: bysource

.. autoclass:: qibo.quantum_info.hamming_weight.HammingWeightResult
    :members:
    :member-order: bysource


Cloud Backends
--------------

Additional backends that support the remote execution of quantum circuits through
cloud service providers, such as IBM and QRC-TII, are provided by the optional qibo plugin
`qibo-cloud-backends <https://github.com/qiboteam/qibo-cloud-backends>`_.
For more information please refer to the
`official documentation <https://qibo.science/qibo-cloud-backends/stable/>`_.
