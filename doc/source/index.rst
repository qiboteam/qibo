.. title::
      Qibo


What is Qibo?
=============

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3997195.svg
   :target: https://doi.org/10.5281/zenodo.3997195

Qibo is an open-source full stack API for quantum simulation and quantum hardware control.

Qibo aims to contribute as a community driven quantum middleware software with

1. *Simplicity:* agnostic design to quantum primitives.
2. *Flexibility:* transparent mechanism to execute code on classical and quantum hardware.
3. *Community:* a common place where find solutions to accelerate quantum development.
4. *Documentation:* describe all steps required to support new quantum devices or simulators.
5. *Applications:* maintain a large ecosystem of applications, quantum models and algorithms.

Qibo key features:

* Definition of a standard language for the construction and execution of quantum circuits with device agnostic approach to simulation and quantum hardware control based on plug and play backend drivers.
* A continuously growing code-base of quantum algorithms applications presented with examples and tutorials.
* Efficient simulation backends with GPU, multi-GPU and CPU with multi-threading support.
* Simple mechanism for the implementation of new simulation and hardware backend drivers.

This documentation refers to Qibo |release|.

Contents
========

.. toctree::
    :maxdepth: 2
    :caption: Introduction

    getting-started/index
    code-examples/index

.. toctree::
    :maxdepth: 2
    :caption: Main documentation

    api-reference/index
    developer-guides/index

.. toctree::
    :maxdepth: 2
    :caption: Appendix

    appendix/benchmarks
    appendix/citing-qibo


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
