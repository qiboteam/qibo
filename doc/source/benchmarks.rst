Benchmarks
==========

The following sections describe how to run Qibo benchmarks using the scripts
found at: https://github.com/Quantum-TII/qibo/tree/master/examples/benchmarks.

Comparison of Qibo to other libraries can be found in our
`original publication <https://arxiv.org/abs/2009.3352360>`_
and in the :ref:`Benchmark results <benchmark-results>` sections.


Circuit benchmarks
------------------

The main benchmark script is ``main.py``. This can be
executed as ``python main.py (OPTIONS)`` where ``(OPTIONS)`` can be any of the
following options:

* ``--nqubits``: Number of qubits in the circuit. Can be a single integer or
  an interval defined with a dash (``-``) as ``a-b``.
  Example: ``--nqubits 5-10`` will run the benchmark for all ``nqubits``
  from 5 to 10 inclusive.

* ``--backend``: Qibo backend to use for the calculation.
  Available backends are ``"custom"``, ``"matmuleinsum"`` and ``"defaulteinsum"``.
  ``"custom"`` is the default backend.

* ``--type``: Type of benchmark circuit.
  Available circuit types are shown in the next section. Some circuit types
  support additional options which are analyzed bellow.

* ``--nshots``: Number of measurement shots.
  If not given no measurements will be performed and the benchmark will
  terminate once the final state vector is found.

* ``--device``: Tensorflow device to use for the benchmarks.
  Example: ``--device /GPU:0`` or ``--device /CPU:0``.

* ``--accelerators``: Devices to use for distributed execution of the circuit.
  Example: ``--accelerators 1/GPU:0,1/GPU:1`` will distribute the execution
  on two GPUs, if these are available and compatible to Tensorflow.

* ``--compile``: If used, the circuit will be compiled using ``tf.function``.
  Note: custom operators do not support compilation.

* ``--precision``: Complex number precision to use for the benchmark.
  Available options are ``'single'`` and ``'double'``.

When a benchmark is executed, the total simulation time will be printed in the
terminal once the simulation finishes. Optionally execution times can be saved
in a ``.h5`` file. This can be enabled by passing the following additional flags:

* ``--directory``: Directory where the ``.h5`` will be saved.

* ``--name``: Name of the ``.h5`` file.

If the file exists in the given directory an error will be raised. The saved file
contains two arrays with the following keys:

  1. ``nqubits``: List with the number of qubits.
  2. ``creation_time``: List with the time required to create the circuit for
     each number of qubits.
  3. ``simulation_time``: List with the total execution time for each number of
     qubits.

If ``--compile`` option is used, then the measured simulation time is the second
call, while the execution time of the first call is saved as ``compile_time``.


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

* ``opt-variational``: Same as ``variational`` using the :class:`qibo.base.gates.VariationalLayer`.
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
    Contains alternating layers of random one-qubit gates and ``CZPow`` gates.
    One-qubit gates are randomly selected from the set ``{RX, RY, RZ}`` and
    have random phases. The total number of layers is controlled using ``--nlayers``.
    Supports the following options:
        - ``--nlayers``: Total number of layers.


VQE benchmark
-------------

It is possible to run a VQE optimization benchmark using ``vqe.py``. This
supports the following options:

* ``--nqubits`` (``int``): Number of qubits in the circuit.
* ``--nlayers`` (``int``): Total number of layers in the circuit.
* ``--method`` (``str``): Optimization method.
* ``--maxiter`` (``int``): Maximum number of iterations for the optimizer.
* ``--varlayer``: If used the circuit will be created using the
  :class:`qibo.base.gates.VariationalLayer` gate which fuses one and two qubits
  for efficiency.

The script will perform the VQE minimization and will print the optimal energy
found and its difference with the exact ground state energy. It will also
show the total execution time.


.. _benchmark-results:

Benchmark results
-----------------

We compare Qibo performance with other publicly available libraries for quantum
circuit simulation. We provide results from different hardware configurations.
The libraries used in the benchmarks are shown in the table below.

[TODO: Add table with libraries]

The default precision and hardware configuration is used for all libraries.
Single-thread Qibo numbers were obtained using the `taskset` utility to restrict
the number of threads.

Results are presented for the following examples:

.. toctree::
    :maxdepth: 1

    benchmarks/results/QFT.md
    benchmarks/results/VAR5.md
    benchmarks/results/HARDWARE.md
