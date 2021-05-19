.. title::
      Qibo

================
Welcome to Qibo!
================

Qibo is an open-source full stack API for quantum simulation and quantum hardware control.

Qibo key features:
  * Definition of a standard language for the construction and execution of quantum circuits with device agnostic approach to simulation and quantum hardware control based on plug and play backend drivers.
  * A continuously growing code-base of quantum algorithms applications presented with examples and tutorials.
  * Efficient simulation backends with GPU, multi-GPU and CPU with multi-threading support.
  * Simple mechanism for the implementation of new simulation and hardware backend drivers.

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3997195.svg
   :target: https://doi.org/10.5281/zenodo.3997195

This documentation refers to Qibo |release|.

.. toctree::
    :maxdepth: 2
    :caption: User documentation

    installation
    examples
    advancedexamples
    applications

.. toctree::
    :maxdepth: 3
    :glob:
    :caption: Features

    qibo

.. toctree::
    :maxdepth: 2
    :caption: Developer documentation

    general
    contributing
    benchmarks

* :ref:`modindex`

.. toctree::
    :maxdepth: 3
    :glob:
    :caption: Extra

    hep
