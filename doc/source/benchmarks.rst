Benchmarks
==========

Benchmark results
-----------------
In our `release publication <https://arxiv.org/abs/2009.01845>`_ we compare
Qibo performance with other publicly available libraries for quantum circuit
simulation and we provide results from different hardware configurations.
For convenience the results can be found in the following examples for various
tasks related to circuit or adiabatic evolution simulation:

.. toctree::
    :maxdepth: 1

    benchmarks/results/QFT.md
    benchmarks/results/VAR5.md
    benchmarks/results/SHOTS.md
    benchmarks/results/PRECISION.md
    benchmarks/results/ADIABATIC.md
    benchmarks/results/HARDWARE.md


The libraries used in these benchmarks are shown in the table below with their
respective default simulation precision and supported hardware configurations.

.. list-table:: Quantum libraries used in the benchmarks.
   :widths: 30 25 50
   :header-rows: 1

   * - Library
     - Precision
     - Hardware
   * - `Qibo 0.1.0 <https://github.com/Quantum-TII/qibo>`_
     - single/double
     - multi-thread CPU, GPU, multi-GPU
   * - `Cirq 0.8.1 <https://github.com/quantumlib/Cirq>`_
     - single
     - single-thread CPU
   * - `TFQ 0.3.0 <https://github.com/tensorflow/quantum>`_
     - single
     - single-thread CPU
   * - `Qiskit 0.14.2 <https://github.com/Qiskit>`_
     - double
     - single-thread CPU
   * - `PyQuil 2.20.0 <https://github.com/rigetti/pyquil>`_
     - double
     - single-thread CPU
   * - `IntelQS 0.14.2 <https://github.com/iqusoft/intel-qs>`_
     - double
     - multi-thread CPU
   * - `QCGPU 0.1.1 <https://github.com/libtangle/qcgpu>`_
     - single
     - multi-thread CPU, GPU
   * - `Qulacs 0.1.10.1 <https://github.com/qulacs/qulacs>`_
     - double
     - multi-thread CPU, GPU


The default precision and hardware configuration is used for all libraries.
Single-thread Qibo numbers were obtained using the `taskset` utility to restrict
the number of threads.

All results presented in the above pages are produced with an
`NVIDIA DGX Station <https://www.nvidia.com/en-us/data-center/dgx-station/>`_.
The machine specification includes 4x NVIDIA Tesla V100 with
32 GB of GPU memory each, and an Intel Xeon E5-2698 v4 with 2.2 GHz
(20-Core/40-Threads) with 256 GB of RAM.
The operating system of this machine is the default Ubuntu 18.04-LTS with
CUDA/``nvcc 10.1``, TensorFlow 2.2.0 and ``g++ 7.5``.

The following sections describe how to run Qibo benchmarks using the scripts
found at: https://github.com/Quantum-TII/qibo/tree/master/examples/benchmarks.


How to run circuit benchmarks?
------------------------------

The main benchmark script is ``main.py``. This can be
executed as ``python main.py (OPTIONS)`` where ``(OPTIONS)`` can be any of the
following options:

* ``--nqubits`` (``int``): Number of qubits in the circuit.

* ``--type`` (``str``): Type of benchmark circuit.
  Available circuit types are shown in the next section. Some circuit types
  support additional options which are described below.

* ``--backend`` (``str``): Qibo backend to use for the calculation.
  Available backends are ``"custom"``, ``"matmuleinsum"``, ``"defaulteinsum"``,
  ``"numpy_defaulteinsum"`` and ``"numpy_matmuleinsum"``.
  ``"custom"`` is the default backend.

* ``--precision`` (``str``): Complex number precision to use for the benchmark.
    Available options are ``'single'`` and ``'double'``.

* ``--device`` (``str``): Tensorflow device to use for the benchmarks.
  Example: ``--device /GPU:0`` or ``--device /CPU:0``.

* ``--accelerators`` (``str``): Devices to use for distributed execution of the circuit.
  Example: ``--accelerators 1/GPU:0,1/GPU:1`` will distribute the execution
  on two GPUs.

* ``--memory`` (``int``): Limits GPU memory used for execution. If no limiter is used,
  Tensorflow uses all available by default.

* ``--nshots`` (``int``): Number of measurement shots.
  This will benchmark the sampling of frequencies, not individual shot samples.
  If not given no measurements will be performed and the benchmark will
  terminate once the final state vector is found.

* ``--compile`` (``bool``): If used, the circuit will be compiled using ``tf.function``.
  Note that custom operators do not support compilation.
  Default is ``False``.

* ``--fuse`` (``bool``): Circuit gates will be fused for faster execution of some circuit
  types. Default is ``False``.

When a benchmark is executed, the total simulation time will be printed in the
terminal once the simulation finishes. Optionally execution times can be saved
by passing the ``--filename`` (``str``) flag. All benchmarks details are logged
in a Python dictionary and saved in a text file using ``json.dump``. The logs
include circuit creation and simulation times. If the given ``filename`` already
exists it will be updated, otherwise it will be created.


Available circuit types
"""""""""""""""""""""""

As explained above, the circuit to be used in the benchmarks can be selected
using the ``--type`` flag. This accepts one of the following options:

* ``qft``: Circuit for `Quantum Fourier Transform <https://en.wikipedia.org/wiki/Quantum_Fourier_transform>`_.
    The circuit contains SWAP gates that rearrange output qubits to their
    original input order.

* ``variational``: Example of a variational circuit.
    Contains layer of parametrized ``RY`` gates followed by a layer of entangling
    ``CZ`` gates. The parameters of ``RY`` gates are sampled randomly from 0 to 2pi.
    Supports the following options:
        - ``--nlayers``: Total number of layers.

* ``opt-variational``: Same as ``variational`` using the :class:`qibo.abstractions.gates.VariationalLayer`.
    This gate optimizes execution by fusing the parametrized with the entangling
    gates before applying them to the state vector.
    Supports the following options:
        - ``--nlayers``: Total number of layers.

* ``one-qubit-gate``: Single one-qubit gate applied to all qubits.
    Supports the following options:
        - ``--gate-type``: Which one-qubit gate to use.
        - ``--nlayers``: Total number of layers.
        - ``--theta``: Value of the free parameter (for parametrized gates).

* ``two-qubit-gate``: Single two-qubit gate applied to all qubits.
    Supports the following options:
        - ``--gate-type``: Which two-qubit gate to use.
        - ``--nlayers``: Total number of layers.
        - ``--theta`` (and/or ``--phi``): Value of the free parameter (for parametrized gates).

* ``ghz``: Circuit that prepares the `GHZ state <https://en.wikipedia.org/wiki/Greenberger%E2%80%93Horne%E2%80%93Zeilinger_state>`_.
    .

* ``supremacy``: Circuit inspired by the `Quantum supremacy experiment <https://www.nature.com/articles/s41586-019-1666-5>`_.
    Contains alternating layers of random one-qubit gates and ``CU1`` gates.
    One-qubit gates are randomly selected from the set ``{RX, RY, RZ}`` and
    have random phases. The total number of layers is controlled using ``--nlayers``.
    Supports the following options:
        - ``--nlayers``: Total number of layers.


How to run VQE benchmarks?
--------------------------

It is possible to run a VQE optimization benchmark using ``vqe.py``. This
supports the following options:

* ``--nqubits`` (``int``): Number of qubits in the circuit.
* ``--nlayers`` (``int``): Total number of layers in the circuit.
* ``--method`` (``str``): Optimization method.
* ``--maxiter`` (``int``): Maximum number of iterations for the optimizer.
* ``--varlayer``: If used the circuit will be created using the
  :class:`qibo.abstractions.gates.VariationalLayer` gate which fuses one and two qubits
  for efficiency.

The script will perform the VQE minimization and will print the optimal energy
found and its difference with the exact ground state energy. It will also
show the total execution time.


How to run QAOA benchmarks?
---------------------------

It is possible to run a QAOA optimization benchmark using ``qaoa.py``. This
supports the following options:

* ``--nqubits`` (``int``): Number of qubits in the circuit.
* ``--nangles`` (``int``): Number of variational parameters in the QAOA ansatz. The parameters are initialized according to uniform distribution in [0, 0.1].
* ``--trotter`` (``bool``): If ``True`` it uses the Trotter decomposition to apply the exponential operators.
* ``--solver`` (``str``): :ref:`Solver <Solvers>` to use for applying the exponential operators.
* ``--method`` (``str``): Optimization method.
* ``--maxiter`` (``int``): Maximum number of iterations for the optimizer.

The script will perform the QAOA minimization and will print the optimal energy
found and its difference with the exact ground state energy. It will also
show the total execution time.
