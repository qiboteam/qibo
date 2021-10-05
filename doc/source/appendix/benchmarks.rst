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
   * - `Qibo 0.1.0 <https://github.com/qiboteam/qibo>`_
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
found at: https://github.com/qiboteam/qibo/tree/master/examples/benchmarks.


How to run circuit benchmarks?
------------------------------

The main benchmark script is ``main.py``. This can be
executed as ``python main.py (OPTIONS)`` where ``(OPTIONS)`` can be any of the
following options:

* ``--nqubits`` (``int``): Number of qubits in the circuit.

* ``--circuit`` (``str``): Circuit to execute. Read the next section for a list
  of available circuits. Some circuit types support additional options which
  are described below. Quantum Fourier Transform is the default benchmark circuit.

* ``--backend`` (``str``): Qibo backend to use for the calculation.
  See :ref:`Simulation backends <simulation-backends>` for more information on the
  calculation backends. ``qibojit`` is the default backend.

* ``--precision`` (``str``): Complex number precision to use for the benchmark.
  Available options are single and double precision. Default is double.

* ``--nreps`` (``int``): Number of repetitions for the circuit execution.

* ``--nshots`` (``int``): Number of measurement shots.
  This will benchmark the sampling of frequencies, not individual shot samples.
  If not given no measurements will be performed and the benchmark will
  terminate once the final state vector is found.

* ``--fuse`` (``bool``): Use :ref:`Circuit fusion <circuit-fusion>` to reduce
  the number of gates in the circuit. Default is ``False``.

* ``--transfer`` (``bool``): Transfer the final state vector from GPU to CPU
  and measure the required time.

* ``--device`` (``str``): Device to use for the benchmarks.
  Example: ``--device /GPU:0`` or ``--device /CPU:0``.
  Note that GPU is not supported by all backends. If a GPU and a supporting
  backend is available it will be the default choice.

* ``--accelerators`` (``str``): Devices to use for distributed execution of the circuit.
  Example: ``--accelerators 1/GPU:0,1/GPU:1`` will distribute the execution
  on two GPUs. The coefficient of each device denotes the number of times to
  reuse this device. See :class:`qibo.core.distcircuit.DistributedCircuit` for
  more details in the distributed implementation.

* ``--memory`` (``int``): Limits GPU memory used for execution. Relevant only
  for Tensorflow backends, as Tensorflow uses the full GPU memory by default.

* ``--threading`` (``str``): Selects numba threading layer. Relevant for the
  qibojit backend on CPU only. See `Numba threading layers <https://numba.pydata.org/numba-doc/latest/user/threading-layer.html>`_
  for more details.

* ``--compile`` (``bool``): Compile the circuit using ``tf.function``.
  Available only when using the tensorflow backend. Default is ``False``.


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


How to run VQE benchmarks?
--------------------------

It is possible to run a VQE optimization benchmark using ``vqe.py``. This
supports the following options:

* ``--nqubits`` (``int``): Number of qubits in the circuit.
* ``--nlayers`` (``int``): Total number of layers in the circuit.
* ``--method`` (``str``): Optimization method. Default is scipy's Powell method.
* ``--maxiter`` (``int``): Maximum number of iterations for the optimizer. Default is ``None``.
* ``--backend`` (``str``): Qibo backend to use.
  See :ref:`Simulation backends <simulation-backends>` for more information on the
  calculation backends. Default is ``qibojit``.
* ``--varlayer`` (``bool``): If ``True`` the :class:`qibo.abstractions.gates.VariationalLayer`
  will be used to construct the circuit, otherwise plain ``RY`` and ``CZ`` gates
  will be used. Default is ``False``.
* ``--filename`` (``str``): Name of the file to save benchmark logs.

The script will perform the VQE minimization and will print the optimal energy
found and its difference with the exact ground state energy. It will also
show the total execution time.


How to run QAOA benchmarks?
---------------------------

It is possible to run a QAOA optimization benchmark using ``qaoa.py``. This
supports the following options:

* ``--nqubits`` (``int``): Number of qubits in the circuit.
* ``--nangles`` (``int``): Number of variational parameters in the QAOA ansatz.
  The parameters are initialized according to uniform distribution in [0, 0.1].
* ``--dense`` (``bool``): If ``True`` it uses the full Hamiltonian matrix to
  perform the unitaries, otherwise it will use the Trotter decomposition
  of the operators. Default is ``False``.
* ``--solver`` (``str``): :ref:`Solvers <Solvers>` to use for applying the exponential operators.
* ``--method`` (``str``): Optimization method. Default is scipy's Powell method.
* ``--maxiter`` (``int``): Maximum number of iterations for the optimizer. Default is ``None``.
* ``--filename`` (``str``): Name of the file to save benchmark logs.

The script will perform the QAOA minimization and will print the optimal energy
found and its difference with the exact ground state energy. It will also
show the total execution time.


How to run time evolution benchmarks?
-------------------------------------

Time evolution benchmarks can be run using ``evolution.py``. This performs an
adiabatic evolution with :meth:`qibo.hamiltonians.X` as the easy Hamiltonian
and :meth:`qibo.hamiltonians.TFIM` as the problem Hamiltonian and supports the
following options:

* ``--nqubits`` (``int``): Number of qubits in the circuit.
* ``--dt`` (``float``): Time step for the evolution algorithm.
* ``--solver`` (``str``): :ref:`Solvers <Solvers>` to use for evolving the state.
* ``--dense`` (``bool``): If ``True`` it uses the full Hamiltonian matrix to
  evolve the system, otherwise it will perform the Trotter decomposition.
  Default is ``False``.
* ``--accelerators`` (``str``): Devices to use for distributed execution of the circuit.
  See :class:`qibo.core.distcircuit.DistributedCircuit` for more details on the
  distributed implementation.
* ``--maxiter`` (``int``): Maximum number of iterations for the optimizer. Default is ``None``.
* ``--filename`` (``str``): Name of the file to save benchmark logs.

The script will perform the QAOA minimization and will print the optimal energy
found and its difference with the exact ground state energy. It will also
show the total execution time.
