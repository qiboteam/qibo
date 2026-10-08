.. _Parameter:

Parameter and Gradients
=======================

It can be useful to define custom parameters in an optimization context. For
example, the rotational angles which encodes information in a Quantum Neural Network
are usually built as a combination of features and trainable parameters. For
doing this, the :class:`qibo.parameter.Parameter` class can be used. It allows
to define custom parameters which can be inserted into a :class:`qibo.models.circuit.Circuit`.
Moreover, it automatically precomputes the analytical derivative of the parameter
function, which can be used to calculate the derivatives of a variational model
with respect to its parameters.

.. automodule:: qibo.parameter
    :members:
    :member-order: bysource


.. _Gradients:

Gradients
---------

In the context of optimization, particularly when dealing with Quantum Machine
Learning problems, it is often necessary to calculate the gradients of functions
that are to be minimized (or maximized). Hybrid methods, which are based on the
use of classical techniques for the optimization of quantum computation procedures,
have been presented in the :doc:`optimizers` section. This approach is very useful in
simulation, but some classical methods cannot be used when using real circuits:
for example, in the context of neural networks, the Back-Propagation algorithm
is used, where it is necessary to know the value of a target function during the
propagation of information within the network. Using a real circuit, we would not
be able to access this information without taking a measurement, causing the state
of the system to collapse and losing the information accumulated up to that moment.
For this reason, in `qibo` we have also implemented methods for calculating the
gradients which can be performed directly on the hardware, such as the
`Parameter Shift Rule`_.

.. automodule:: qibo.derivative
   :members:
   :member-order: bysource

.. _`Parameter Shift Rule`: https://arxiv.org/abs/1811.11184
