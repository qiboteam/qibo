Installation instructions
=========================

Operating systems support
-------------------------

In the table below we summarize the status of *pre-compiled binaries
distributed with pypi* for the packages listed above.

+------------------+------+---------+-----------+---------+
| Operating System | qibo | qibojit |Tensorflow | Pytorch |
+==================+======+=========+===========+=========+
| Linux x86        | Yes  | Yes     | Yes       | Yes     |
+------------------+------+---------+-----------+---------+
| MacOS >= 10.15   | Yes  | Yes     | Yes       | Yes     |
+------------------+------+---------+-----------+---------+
| Windows          | Yes  | Yes     | Yes       | Yes     |
+------------------+------+---------+-----------+---------+

.. note::
      All packages are supported for Python >= 3.9.


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
Make sure you have Python 3.9 or greater, then use ``pip`` to install ``qibo`` with:

.. code-block:: bash

      pip install qibo

The ``pip`` program will download and install all the required
dependencies for Qibo.


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
      ``qibojit`` is compatible with
      `cuQuantum SDK v22.03 <https://docs.nvidia.com/cuda/cuquantum/cuquantum_sdk_release_notes.html#cuquantum-sdk-v22-03>`__
      and
      `cuQuantum SDK v22.05 <https://docs.nvidia.com/cuda/cuquantum/cuquantum_sdk_release_notes.html#cuquantum-sdk-v22-05>`__.


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


.. _installing-qml-backends:

Backends with automatic differentiation support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you need automatic differentiation support, for tracing gradients of your
quantum algorithm or for building some quantum machine learning routine,
the right backends for you are those provided by the `Qiboml <https://github.com/qiboteam/qiboml>`__
package.

In particular, Qiboml currently support `Pytorch <https://pytorch.org/>`_ and
`Tensorflow <https://www.tensorflow.org>`_ interfaces, integrating the qibo functionalities
into these well-known machine learning frameworks. Quantum layers can be constructed
and added to your Pytorch or Tensorflow models, and trained using any supported
optimization routine.

In order to use these quantum machine learning backends please make sure the
preferred package is installed following `Tensorflow's <https://www.tensorflow.org/install>`_
or `Pytorch's <https://pytorch.org/get-started/locally/>`_ installation instructions.

To switch to Tensorflow or Pytorch backend please do:

.. code-block::  python

      import qibo
      # in case of Tensorflow
      qibo.set_backend(backend="qiboml", platform="tensorflow")
      # in case of Pytorch
      qibo.set_backend(backend="qiboml", platform="pytorch")

In order to start using automatic differentiation tools with Qibo,
we recommend the installation using:

.. code-block:: bash

      pip install qibo qiboml tensorflow

or

.. code-block:: bash

      pip install qibo qiboml torch
