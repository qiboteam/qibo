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
We provide precompiled packages for linux and macos operating systems
for multiple Python versions.

Make sure you have Python 3.6 or greater, then
use ``pip`` to install ``qibo`` with:

.. code-block:: bash

      pip install qibo

The ``pip`` program will download and install all the required
dependencies for QIBO.

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
