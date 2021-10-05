Installation instructions
=========================

Operating systems support
-------------------------

In the table below we summarize the status of *pre-compiled binaries
distributed with pypi* for the packages listed above.

+------------------+------+---------+--------+------------+
| Operating System | qibo | qibojit | qibotf | tensorflow |
+==================+======+=========+========+============+
| Linux x86        | Yes  | Yes     | Yes    | Yes        |
+------------------+------+---------+--------+------------+
| MacOS >= 10.15   | Yes  | Yes     | Yes    | Yes        |
+------------------+------+---------+--------+------------+
| Windows          | Yes  | Yes     | No     | Yes        |
+------------------+------+---------+--------+------------+

.. note::
      All packages are supported for Python >= 3.6.


Backend installation
--------------------

.. _installing-qibo:

qibo
^^^^

The ``qibo`` is the base required package which includes the language API and a
lightweight cross-platform simulator based on ``numpy``. In order to accelerate
simulation please consider specialized backends listed in
:ref:`simulation-backends`.

Installing with pip
"""""""""""""""""""

The installation using ``pip`` is the recommended approach to install Qibo.
Make sure you have Python 3.6 or greater, then use ``pip`` to install ``qibo`` with:

.. code-block:: bash

      pip install qibo

The ``pip`` program will download and install all the required
dependencies for Qibo.


Installing from source
""""""""""""""""""""""

The installation procedure presented in this section is useful when you have to develop the code from source.

In order to install Qibo from source, you can simply clone the GitHub repository with

.. code-block::

      git clone https://github.com/qiboteam/qibo.git
      cd qibo
      pip install . # or pip install -e .

_______________________

.. _installing-qibojit:

qibojit
^^^^^^^

The ``qibojit`` package contains a simulator implementation based on
just-in-time (JIT) custom kernels using `numba <https://numba.pydata.org/>`_
and `cupy <https://cupy.dev/>`_.

This backend is used by default, however, if needed, in order to switch to the
``qibojit`` backend please do:

.. code-block:: python

      import qibo
      qibo.set_backend("qibojit")

Installing with pip
"""""""""""""""""""

The installation using ``pip`` is the recommended approach to install
``qibojit``.

In order to install the package use the following command:

.. code-block:: bash

      pip install qibo[qibojit]

.. note::
      The ``pip`` program will download and install all the required
      dependencies except `cupy <https://cupy.dev/>`_ which is required for GPU
      acceleration. Please install `cupy <https://cupy.dev/>`_ by following the
      instructions from the `official website
      <https://docs.cupy.dev/en/stable/install.html>`_ for your GPU hardware.


Installing from source
""""""""""""""""""""""

The installation procedure presented in this section is useful if you have to
develop the code from source.

In order to install the package perform the following steps:

.. code-block::

      git clone https://github.com/qiboteam/qibojit.git
      cd qibojit

then proceed with the installation of requirements with:

.. code-block::

      pip install -r requirements.txt

Then proceed with the ``qibojit`` installation using ``pip``

.. code-block::

      pip install .

or if you prefer to manually execute all installation steps:

.. code-block::

      # builds binaries
      python setup.py deve

_______________________

.. _installing-qibotf:

qibotf
^^^^^^

The ``qibotf`` package contains a custom simulator implementation based on
TensorFlow and custom operators in CUDA/C++.

If needed, in order to switch to the ``qibotf`` backend please do:

.. code-block:: python

      import qibo
      qibo.set_backend("qibotf")

Installing with pip
"""""""""""""""""""

The installation using ``pip`` is the recommended approach to install
``qibotf``. We provide precompiled packages for linux x86/64 and macosx 10.15 or
greater for Python 3.6, 3.7, 3.8 and 3.9.

In order to install the package use the following command:

.. code-block:: bash

      pip install qibo[qibotf]

The ``pip`` program will download and install all the required
dependencies.

.. note::
      The ``pip`` packages for linux are compiled with CUDA support, so if your
      system has a NVIDIA GPU, Qibo will perform calculations on GPU. Note that
      ``qibotf`` uses TensorFlow for GPU management, if your system has a NVIDIA
      GPU, make sure TensorFlow runs on GPU, please refer to the `official
      documentation <https://www.tensorflow.org/install/gpu>`_.


Installing from source
""""""""""""""""""""""

The installation procedure presented in this section is useful if the
pre-compiled binary packages for your operating system is not available or if
you have to develop the code from source.

In order to install the package perform the following steps:

.. code-block::

      git clone https://github.com/qiboteam/qibotf.git
      cd qibotf

then proceed with the installation of requirements with:

.. code-block::

      pip install -r requirements.txt

Make sure your system has a GNU ``g++ >= 4`` compiler. If you are working on
macosx make sure the command ``c++`` is ``clang >= 11`` and install the libomp
library with ``brew install libomp`` command.

Optionally, you can use the ``CXX`` environment variable to set then compiler
path. Similarly, the ``PYTHON`` environment variable sets the python interpreter
path.

.. note::
      If your system has a NVIDIA GPU, make sure TensorFlow is installed
      properly and runs on GPU, please refer to the `official
      documentation <https://www.tensorflow.org/install/gpu>`_.

      In that case, you can activate GPU support for Qibo by:

      1. installing the NVCC compiler matching the TensorFlow CUDA version, see the `CUDA documentation <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_.

      2. exporting the ``CUDA_PATH`` variable with the CUDA installation path containing the cuda compiler.

      3. make sure the NVCC compiler is available from ``CUDA_PATH/bin/nvcc``, otherwise the compilation may fail. You can locate it with ``whereis nvcc`` and eventually link/copy to your ``CUDA_PATH/bin`` folder.

      For example, TensorFlow 2.5.0 supports CUDA 11.2. After installing
      TensorFlow proceed with the NVCC 11.2 installation. On linux the
      installation path usually is ``/usr/local/cuda-11.2/``.

      Before installing Qibo do ``export CUDA_PATH=/usr/local/cuda-11.2``.

      Note that Qibo will not enable GPU support if points 1 and 2 are not
      performed.


Then proceed with the ``qibotf`` installation using ``pip``

.. code-block::

      pip install .

or if you prefer to manually execute all installation steps:

.. code-block::

      # builds binaries
      python setup.py build

      # installs the Qibo packages
      python setup.py install # or python setup.py develop



_______________________

.. _installing-tensorflow:

tensorflow
^^^^^^^^^^

If the `TensorFlow <https://www.tensorflow.org>`_ package is installed Qibo
will detect and provide to the user the possibility to use ``tensorflow``
backend.

This backend is used by default if ``qibotf`` is not installed, however, if
needed, in order to switch to the ``tensorflow`` backend please do:

.. code-block:: python

      import qibo
      qibo.set_backend("tensorflow")

In order to install the package, we recommend the installation using:

.. code-block:: bash

      pip install qibo[tensorflow]

.. note::
      TensorFlow can be installed following its `documentation
      <https://www.tensorflow.org/install>`_.

_______________________

.. _installing-numpy:

numpy
^^^^^

The ``qibo`` base package is distributed with a lightweight quantum simulator
shipped with the qibo base package. No extra packages are required.

This backend is used by default if ``qibotf`` or ``tensorflow`` are not
installed, however, if needed, in order to switch to the ``numpy`` backend
please do:

.. code-block:: python

      import qibo
      qibo.set_backend("numpy")

_______________________

.. _docker:

Using the code with docker
--------------------------

We provide docker images for tag release of the code using GitHub Packages. The
docker images contain a pre-configured linux environment with the Qibo
framework installed with the specific tag version.

Please refer to the download and authentication instructions from the `Qibo GitHub Packages`_.

In order to start the docker image in interactive mode please use docker
standard syntax, for example:

.. code::

    docker run -it ghcr.io/qiboteam/qibo:<tag_version> bash

This will open a bash shell with the Qibo environment already activated, with
all binaries and scripts from the Qibo framework.

.. _Qibo GitHub Packages: https://github.com/qiboteam/qibo/pkgs/container/qibo
