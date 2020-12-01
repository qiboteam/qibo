Models for High Energy Physics
==============================

The Qibo package comes with the following extra models:

* qPDFs_

_______________________

.. _qPDFs:

Quantum PDFs
------------

Qibo provides a variational circuit model for parton distribution functions,
named qPDF. This model is based on :class:`qibo.models.Circuit` and provides a
simple API to evaluate PDF flavours at specific values of the momentum fraction
x. Further details and references about this model are presented in the
``examples/qPDF`` tutorial.

qPDF circuit model
^^^^^^^^^^^^^^^^^^

.. autoclass:: qibo.hep.qPDF
    :members:
    :member-order: bysource
