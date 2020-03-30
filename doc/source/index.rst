.. title::
      qibo's documentation!

===================================
QIBO: a quantum computing framework
===================================

.. contents::
   :local:
   :depth: 1

Qibo is an open-source high-level API, written in Python and capable of running a quantum algorithms on quantum computers and simulators. It provides a standard interface, allows the implementation and extension of new algorithms.

Some of the key features of QIBO are:

    - Modular implementation on single (multi-threading) CPU and GPU.

General Overview
================

The QIBO framework in this repository implements a common system to deal with classical hardware and future quantum hardware.

The code abstraction is located in ``qibo/base``. The default simulation engine is implemented using tensorflow and is located in ``qibo/tensorflow``.

Other backends can be implemented by following the tensorflow example and adding a switcher in ``qibo/config.py``.

Regression tests, which are run by the continous integration workflow are stored in ``qibo/tests``. These tests contain several examples about how to use qibo.

The ``qibo/benchmarks`` folder contains benchmark code that has been implemented so far for specific applications.

Installation
============

In order to install you can simply clone this repository with

.. code-block:: bash

      git clone git@github.com:Quantum-TII/qibo.git

and then proceed with the installation with:

.. code-block:: bash

      python setup.py install

If you prefer to keep changes always synchronized with the code then install using the develop option:

.. code-block:: bash

      python setup.py develop


Indices and tables
==================

.. toctree::
    :maxdepth: 3
    :glob:
    :caption: Contents:

    QIBO<self>
    qibo
    examples

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
