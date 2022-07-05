Installation instructions
=========================

Operating systems support
-------------------------

In the table below we summarize the status of *pre-compiled binaries
distributed with pypi* for the packages listed above.

+------------------+------+---------+------------+
| Operating System | qibo | qibojit | tensorflow |
+==================+======+=========+============+
| Linux x86        | Yes  | Yes     | Yes        |
+------------------+------+---------+------------+
| MacOS >= 10.15   | Yes  | Yes     | Yes        |
+------------------+------+---------+------------+
| Windows          | Yes  | Yes     | Yes        |
+------------------+------+---------+------------+

.. note::
      All packages are supported for Python >= 3.7.


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
Make sure you have Python 3.7 or greater, then use ``pip`` to install ``qibo`` with:

.. code-block:: bash

      pip install qibo

The ``pip`` program will download and install all the required
dependencies for Qibo.

Installing with conda
"""""""""""""""""""""

We provide conda packages for ``qibo`` through the `conda-forge
<https://anaconda.org/conda-forge>`_ channel.

To install the package with conda run:

.. code-block:: bash

      conda install -c conda-forge qibo


Installing from source
""""""""""""""""""""""

The installation procedure presented in this section is useful when you have to
develop the code from source.

In order to install Qibo from source, you can simply clone the GitHub repository
with

.. code-block::

      git clone https://github.com/qiboteam/qibo.git
      cd qibo
      pip install .

_______________________

.. _installing-qibojit:

qibojit
^^^^^^^

The ``qibojit`` package contains a simulator implementation based on
just-in-time (JIT) custom kernels using `numba <https://numba.pydata.org/>`_
and `cupy <https://cupy.dev/>`_. We also provide another implementation based
on `cuQuantum <https://developer.nvidia.com/cuquantum-sdk>`_ primitives available
when running Qibo on GPU.

This backend is used by default, however, if needed, in order to switch to the
``qibojit`` backend please do:

.. code-block::  python

      import qibo
      qibo.set_backend("qibojit")

Custom cupy kernels will be used by default if a GPU is available and
custom numba kernels if a GPU is not available.
If a GPU is available it is possible to switch to the cuQuantum implementation
using the ``platform`` argument, for example:

.. code-block::  python

      import qibo
      # switch to the cuquantum implementation
      qibo.set_backend("qibojit", platform="cuquantum")
      # switch to custom numba kernels (even if a GPU is available)
      qibo.set_backend("qibojit", platform="numba")


Installing with pip
"""""""""""""""""""

The installation using ``pip`` is the recommended approach to install
``qibojit``.

In order to install the package use the following command:

.. code-block:: bash

      pip install qibojit

.. note::
      The ``pip`` program will download and install all the required
      dependencies except `cupy <https://cupy.dev/>`_ and/or
      `cuQuantum <https://developer.nvidia.com/cuquantum-sdk>`_
      which are required for GPU acceleration.
      The cuQuantum dependency is optional, as it is required only for
      the ``cuquantum`` platform. Please install `cupy <https://cupy.dev/>`_ by following the
      instructions from the `official website
      <https://docs.cupy.dev/en/stable/install.html>`_ for your GPU hardware.
      The installation instructions for `cuQuantum <https://developer.nvidia.com/cuquantum-sdk>`_
      are available in the `official documentation <https://docs.nvidia.com/cuda/cuquantum/python/README.html>`__.

Installing with conda
"""""""""""""""""""""

We provide conda packages for ``qibo`` and ``qibojit`` through the `conda-forge
<https://anaconda.org/conda-forge>`_ channel.

To install both packages with conda run:

.. code-block:: bash

      conda install -c conda-forge qibojit

.. note::
      The ``conda`` program will download and install all the required
      dependencies except `cupy <https://cupy.dev/>`_ and/or
      `cuQuantum <https://developer.nvidia.com/cuquantum-sdk>`_
      which are required for GPU acceleration.
      The cuQuantum dependency is optional, as it is required only for
      the ``cuquantum`` platform. Please install `cupy <https://cupy.dev/>`_ by following the
      instructions from the `official website
      <https://docs.cupy.dev/en/stable/install.html>`_ for your GPU hardware.
      The installation instructions for `cuQuantum <https://developer.nvidia.com/cuquantum-sdk>`_
      are available in the `official documentation <https://docs.nvidia.com/cuda/cuquantum/python/README.html>`__.

Installing from source
""""""""""""""""""""""

The installation procedure presented in this section is useful if you have to
develop the code from source.

In order to install the package perform the following steps:

.. code-block::

      git clone https://github.com/qiboteam/qibojit.git
      cd qibojit

Then proceed with the ``qibojit`` installation using ``pip``

.. code-block::

      pip install .

_______________________

.. _installing-tensorflow:

tensorflow
^^^^^^^^^^

If the `TensorFlow <https://www.tensorflow.org>`_ package is installed Qibo
will detect and provide to the user the possibility to use ``tensorflow``
backend.

This backend is used by default if ``qibojit`` is not installed, however, if
needed, in order to switch to the ``tensorflow`` backend please do:

.. code-block::  python

      import qibo
      qibo.set_backend("tensorflow")

In order to install the package, we recommend the installation using:

.. code-block:: bash

      pip install qibo tensorflow

.. note::
      TensorFlow can be installed following its `documentation
      <https://www.tensorflow.org/install>`_.

_______________________

.. _installing-numpy:

numpy
^^^^^

The ``qibo`` base package is distributed with a lightweight quantum simulator
shipped with the qibo base package. No extra packages are required.

This backend is used by default if ``qibojit`` or ``tensorflow`` are not
installed, however, if needed, in order to switch to the ``numpy`` backend
please do:

.. code-block::  python

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


.. note::
      The docker image contains the basic ``qibo`` package with the lightweight
      ``numpy`` backend for simulation.
