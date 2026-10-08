.. _Parallel:

Parallelism
===========

We provide CPU multi-processing methods for circuit evaluation for multiple
input states and multiple parameters for fixed input state.

When using the methods below the ``processes`` option controls the number of
processes used by the parallel algorithms through the ``multiprocessing``
library. By default ``processes=None``, in this case the total number of logical
cores are used. Make sure to select the appropriate number of processes for your
computer specification, taking in consideration memory and physical cores. In
order to obtain optimal results you can control the number of threads used by
each process with the ``qibo.set_threads`` method. For example, for small-medium
size circuits you may benefit from single thread per process, thus set
``qibo.set_threads(1)`` before running the optimization.

.. automodule:: qibo.parallel
   :members:
   :member-order: bysource
   :exclude-members: ParallelResources
