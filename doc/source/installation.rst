Installing QIBO
===============

The QIBO package comes with the following modules:

* :ref:`installing-with-pip`
* :ref:`installing-from-source`

_______________________

.. _installing-with-pip:

Installing with pip
-------------------

The installation using ``pip`` is the recommended approach to install QIBO.
We provide precompiled packages for linux x86/64 and macosx 10.11 or greater
for multiple Python versions.

Make sure you have Python 3.6 or greater, then
use ``pip`` to install ``qibo`` with:

.. code-block:: bash

      pip install qibo

The ``pip`` program will download and install all the required
dependencies for QIBO.

.. note::
    The ``pip`` packages for linux are compiled with CUDA support, so if your
    system has a NVIDIA GPU, QIBO will perform calculations on GPU.

.. _installing-from-source:

Installing from source
----------------------

The installation procedure presented in this section is useful in two situations:

- you need to install QIBO in an operating system and environment not supported by the ``pip`` packages (see :ref:`installing-with-pip`).

- you have to develop the code from source.

In order to install QIBO from source, you can simply clone the GitHub repository with

.. code-block::

      git clone https://github.com/Quantum-TII/qibo.git
      cd qibo

then proceed with the installation of requirements with:

.. code-block::

      pip install -r requirements.txt

.. note::
      If your system has a NVIDIA GPU, make sure TensorFlow is installed
      properly and runs on GPU, please refer to the `official
      documentation <https://www.tensorflow.org/install/gpu>`_.

      In that case, you can activate GPU support for QIBO by:

      1. installing the NVCC compiler matching the TensorFlow CUDA version, see the `CUDA documentation <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_.

      2. exporting the ``CUDA_PATH`` variable with the CUDA installation path containing the cuda compiler.

      For example, TensorFlow 2.2 supports CUDA 10.1. After installing
      TensorFlow proceed with the NVCC 10.1 installation. On linux the
      installation path is ``/usr/local/cuda-10.1/``.

      Before installing QIBO do ``export CUDA_PATH=/usr/local/cuda-10.1``.

      Note that QIBO will not enable GPU support if points 1 and 2 are not
      performed.


Then proceed with the QIBO installation using ``pip``

.. code-block::

      pip install .

or if you prefer to manually execute all installation steps:

.. code-block::

      # builds binaries
      python setup.py build

      # installs the QIBO packages
      python setup.py install # or python setup.py develop

If you prefer to keep changes always synchronized with the code then install using the develop option:

.. code-block::

      pip install -e .
      # or
      python setup.py develop

Optionally, in order to run tests and build documentation locally
you can install extra dependencies with:

.. code-block::

      pip install qibo[tests,docs]
