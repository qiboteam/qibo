.. title::
      Qibo


What is Qibo?
=============

**Qibo is an open-source full stack API for quantum simulation and quantum hardware control.**

**Qibo** aims to contribute as a community driven quantum middleware software.
The project goals can be enumerated by the following concepts:

1. *Simplicity:* design agnostic primitives for quantum applications.
2. *Flexibility:* transparent mechanism to execute code on classical and quantum hardware.
3. *Community:* build a collaborative repository where users, developers and laboratories can find solutions to accelerate quantum development.
4. *Documentation:* provide an extensive description of all steps required to support new quantum devices or simulators.
5. *Applications:* maintain a large ecosystem of applications, quantum models and algorithms.

**Qibo** key features:

* Definition of a standard language for the construction and execution of quantum circuits with device agnostic approach to simulation and quantum hardware control based on plug and play backend drivers.
* A continuously growing code-base of quantum algorithms applications presented with examples and tutorials.
* Efficient simulation backends with GPU, multi-GPU and CPU with multi-threading support.
* Simple mechanism for the implementation of new simulation and hardware backend drivers.

Publications
============

*  *"Qibo: a framework for quantum simulation with hardware acceleration"*,
   Stavros Efthymiou, Sergi Ramos-Calderer, Carlos Bravo-Prieto, Adrián
   Pérez-Salinas, Diego García-Martín, Artur Garcia-Saez, José Ignacio Latorre,
   Stefano Carrazza [`arXiv:2009.01845`_].

.. _`arXiv:2009.01845`: https://arxiv.org/abs/2009.01845

Contents
========

.. toctree::
    :maxdepth: 2

    getting-started/index

.. toctree::
    :hidden:
    :maxdepth: 2

    examples
    advancedexamples
    applications

.. toctree::
    :hidden:
    :maxdepth: 2
    :glob:
    :caption: Qibo API reference

    qibo

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Developer guides

    general
    contributing
    benchmarks

.. toctree::
    :hidden:
    :maxdepth: 3
    :glob:
    :caption: Why use Qibo?

    qibo

.. toctree::
    :hidden:
    :maxdepth: 3
    :glob:
    :caption: Contributing

    qibo

.. toctree::
    :hidden:
    :maxdepth: 3
    :glob:
    :caption: Extra

    hep



Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
