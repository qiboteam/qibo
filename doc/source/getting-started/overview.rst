Project overview
================

The Qibo project targets the development of an open-source full stack API for
quantum simulation and quantum hardware control.

Quantum technologies, such as NISQ devices, are developed by research
institutions and require a high level of knowledge of the physics and electronic
devices used to prepare, execute and retrieve measurements from the experimental
apparatus.

In this context, Qibo proposes an agnostic approach to quantum simulation and
hardware control, providing the required components and standards to quickly
connect the classical hardware and experimental setup into a software stack
which automates all aspects of a quantum computation.

In the picture below, we summarize the major components of the Qibo "ecosystem".

.. image:: overview.png

The first component is the language API, based on Python 3, which defines the
interface for the development of quantum applications, models and new
algorithms. We also provide a large code-base of models and algorithms,
presented with code examples and step-by-step tutorials. Finally, we provide
several tools for the laboratory management and quantum hardware control.

Qibo provides a plug and play mechanism of :ref:`backend drivers <backend-drivers>` which
specializes the code for quantum simulation on different classical hardware
configurations, such as multi-threading CPU, single GPU and multi-GPU, and
similarly for quantum hardware control, from superconducting to ion trap
technologies including FPGA and AWG devices.

_______________________

.. _backend-drivers:

Backend drivers
---------------

As mentioned above, we provide backends for quantum simulation on classical
hardware and quantum hardware management and control. In the image below we
present a schematic view of the currently supported backends.

.. image:: backends.png

Quantum simulation is proposed through dedicated backends for single node
multi-GPU and multi-threading CPU setups. Quantum hardware control is supported
for chips based on superconducting qubits.

_______________________

.. _packages:

Packages
--------

Following the overview description above, in this section we present the python
packages for the modules and backends presented.

Base package
^^^^^^^^^^^^

* :ref:`installing-qibo` is the base package for coding and using the API. This package contains all primitives and algorithms for start coding quantum circuits, adiabatic evolution and more (see :ref:`Components`). This package comes with a lightweight quantum simulator which works on multiple CPU architectures such as x86 and arm64.

.. _simulation-backends:

Simulation backends
^^^^^^^^^^^^^^^^^^^

We provide multiple simulation backends for Qibo, which are automatically loaded
if the corresponding packages are installed, following the hierarchy below:

* :ref:`installing-qibojit`: an efficient simulation backend for CPU, GPU and multi-GPU based on just-in-time (JIT) compiled custom operators. Install this package if you need to simulate quantum circuits with large number of qubits or complex quantum algorithms which may benefit from computing parallelism.
* :ref:`installing-qibotf`: an efficient simulation backend for CPU, GPU and multi-GPU based on TensorFlow custom operators. Install this package if you need to simulate quantum circuits with large number of qubits or complex quantum algorithms which may benefit from computing parallelism.
* :ref:`installing-tensorflow`: a pure TensorFlow implementation for quantum simulation which provides access to gradient descent optimization and the possibility to implement classical and quantum architectures together. This backend is not optimized for memory and speed, use :ref:`installing-qibotf` instead.
* :ref:`installing-numpy`: a lightweight quantum simulator shipped with the :ref:`installing-qibo` base package. Use this simulator if your CPU architecture is not supported by the other backends. Please note that the simulation performance is quite poor in comparison to other backends.

The default backend that is used is the first available from the above list.
The user can switch to a different using the ``qibo.set_backend`` method
(see :ref:`Backends <Backends>` section for more details).

The active default backend will be printed as an info message the first time
Qibo is imported in the code. If qibojit and qibotf are not installed,
an additional warning will appear prompting the user to install one of the two
for increased performance and multi-threading and/or GPU capabilities.
The logging level can be controlled using the ``QIBO_LOG_LEVEL`` environment
variable. This can be set to 3 to hide info messages or 4 to hide both info
and warning messages. The default value is 1 allowing all messages to appear.


.. _hardware-backends:

Hardware backends
^^^^^^^^^^^^^^^^^

We provide the following hardware control backends for Qibo:

* :ref:`installing-qiboicarusq` (*experimental*): a module for laboratories, containing the specifics to operate Qibo on chips based on superconducting qubits, designed specifically for the IcarusQ experiment at `CQT <https://www.quantumlah.org/>`_.

_______________________

Operating systems support
-------------------------

In the table below we summarize the status of *pre-compiled binaries
distributed with pypi* for the packages listed above.

+------------------+------+---------+------------------+------------+
| Operating System | qibo | qibojit | qibotf (cpu/gpu) | tensorflow |
+==================+======+=========+==================+============+
| Linux x86        | Yes  | Yes     | Yes/Yes          | Yes        |
+------------------+------+---------+------------------+------------+
| MacOS >= 10.15   | Yes  | Yes     | Yes/No           | Yes        |
+------------------+------+---------+------------------+------------+
| Windows          | Yes  | Yes     | No/No            | Yes        |
+------------------+------+---------+------------------+------------+

.. note::
      All packages are supported for Python >= 3.6.
