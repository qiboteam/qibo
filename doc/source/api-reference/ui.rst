User Interface
==============

The ``qibo.ui`` module provides a set of plotting utilities built on top of
``matplotlib`` to visualize circuits, states and measurement outcomes. It is
imported as ``from qibo.ui import ...``.

Circuit drawing
---------------

.. autofunction:: qibo.ui.plot_circuit

State and result visualization
------------------------------

.. autofunction:: qibo.ui.visualize_state

.. autofunction:: qibo.ui.plot_density_hist

Bloch sphere
------------

.. autoclass:: qibo.ui.bloch.BlochSphere
    :members:
    :member-order: bysource
