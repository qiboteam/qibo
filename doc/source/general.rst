Code Overview
=============

The QIBO framework in this repository implements a common system to deal with classical hardware and future quantum hardware.

The code abstraction is located in ``qibo/base``. The default simulation engine is implemented using tensorflow and is located in ``qibo/tensorflow``.

Other backends can be implemented by following the tensorflow example and adding a switcher in ``qibo/config.py``.

Regression tests, which are run by the continous integration workflow are stored in ``qibo/tests``. These tests contain several examples about how to use qibo.

The ``qibo/benchmarks`` folder contains benchmark code that has been implemented so far for specific applications.
