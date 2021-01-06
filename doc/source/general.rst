Code Overview
=============

The Qibo framework in this repository implements a common system to deal with classical hardware and future quantum hardware.

The code abstraction is located in ``qibo/abstractions`` and the core simulation engine is located in ``qibo/core``.
This simulation engine uses an abstract backend object ``K`` to perform calculation the structure of which is defined in ``qibo/backends/base.py``.

Currently two calculation backends are implemented, one based in numpy and one based in `Tensorflow <http://tensorflow.org/>`_.
The Tensorflow backend is supplemented by custom operators for efficient application of quantum gates to state vectors and density matrices.
Other backends can be implemented by inheriting the ``BaseBackend`` and implementing its abstract methods.

Regression tests, which are run by the continous integration workflow are stored in ``qibo/tests``. These tests contain several examples about how to use qibo.

The ``examples/benchmarks`` folder contains benchmark code that has been implemented so far for specific applications.
