Code overview
=============

The Qibo framework in this repository implements a common system to deal with
classical hardware and future quantum hardware.

Features
--------

The main Qibo objects are circuits, defined in ``qibo/models/circuit.py`` and
gates, defined in ``qibo/gates``. These allow the user to simulate circuits
that follow the gate-based approach of quantum computation or to execute
them on different hardware. These objects are backend agnostic, meaning that
the same circuit can be executed using different backends.
Backends are defined in ``qibo/backends`` and are used to simulate the abstract
circuits or execute them on hardware.

Qibo provides additional features that are useful for quantum applications, such
as Hamiltonians (``qibo/hamiltonians``), time evolution simulation (``qibo/models/evolution.py``)
and variational models (``qibo/models/variational.py``).

Including a new backend
-----------------------

New backends can be implemented by inheriting
:class:`qibo.backends.abstract.Backend` and implementing its abstract
methods. If the backend is for classical simulation one may prefer to
inherit :class:`qibo.backends.abstract.Simulator` instead.


Examples and tutorials
----------------------

The ``examples`` folder contains benchmark code for applications/tutorials
described in :ref:`Applications <applications>` while ``examples/benchmarks``
contains some code for benchmarking only.
