Code overview
=============

The Qibo framework in this repository implements a common system to deal with
classical hardware and future quantum hardware.

Abstractions
------------

The code abstraction is located in ``qibo/abstractions`` and the core simulation
engine is located in ``qibo/core``. This simulation engine uses an abstract
backend object ``K`` to perform calculation the structure of which is defined in
``qibo/backends/abstract.py``.

* :class:`qibo.abstractions.circuit.AbstractCircuit`: base class with attributes and abstract methods for circuit execution.
* :class:`qibo.abstractions.states.AbstractState`: data structure with tensor representation and state properties.
* :class:`qibo.abstractions.gates.Gate`: implementation of standard gates and respective class attributes, virtual methods for gate matrix construction and operation.
* :class:`qibo.backends.abstract.AbstractBackend`: virtual methods for the most common linear algebra operations.
* :class:`qibo.abstractions.hamiltonians.AbstractHamiltonian`: virtual methods for quantum operators math operations.

Including a new backend
-----------------------

New backends can be implemented by inheriting the
:class:`qibo.backends.abstract.AbstractBackend` and implementing its abstract
methods. In particular the developer should:

* Perform an ``AbstractBackend`` inheritance.
* Register the new backend in ``src/qibo/backends/profiles.yml``, or point to a new profile file with the environment flag ``QIBO_PROFILE``.
* Load your backend with ``qibo.set_backend("your_backend_name")`` or use the environment flag ``QIBO_BACKEND="your_backend_name"``.

Here you have an example for the structure of the  ``profile.yml`` file:

.. code-block:: yaml

    backends:
      # simulation backends - numpy is available by default
      - name: qibojit
        driver: qibojit.custom_operators.JITCustomBackend
        minimum_version: 0.0.4rc0

      - name: qibotf
        driver: qibotf.custom_operators.TensorflowCustomBackend
        minimum_version: 0.0.4rc0

      - name: tensorflow
        driver: qibo.backends.tensorflow.TensorflowBackend
        minimum_version: 2.2.0

      # hardware backends
      - name: qibolab
        driver: qibolab.backend.QibolabBackend
        minimum_version: 0.0.1.dev0
        is_hardware: True

    # default active backend after importing all modules
    default: qibojit

When including a new backend, you should include its:

* **name:** The name of the new backend.
* **driver:** The class which performs the inheritance from ``AbstractBackend``, including the python module containing it.
* **minimum_version:** (optional) Minimum backend version for it to be compatible with qibo.
* **is_hardware:** (optional) Boolean that specifies if the backend targets a hardware device. If ``False`` the backend is used for classical simulation.

Finally, the default backend which is loaded when importing Qibo can be updated
by changing the value of the ``default:`` key.


Examples and tutorials
----------------------

The ``examples`` folder contains benchmark code for applications/tutorials
described in :ref:`Applications <applications>` while ``examples/benchmarks``
contains some code for benchmarking only.
