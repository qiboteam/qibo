.. _States:

Execution Outcomes
==================

Qibo circuits return different objects when executed depending on what the
circuit contains and on the settings of the simulation. The following table
summarizes which outcomes to expect depending on whether:

* the circuit contains noise channels
* the qubits are measured at the end of the execution
* some collapse measurement is present in the circuit
* ``density_matrix`` is set to ``True`` in simulation

.. table::

   +----------+--------------+----------+----------------+------------------------------------------+
   | Noise    | Measurements | Collapse | Density Matrix |      Outcome                             |
   +==========+==============+==========+================+==========================================+
   |    ❌    |      ❌      |    ❌    |   ❌ / ✅      | :class:`qibo.result.QuantumState`        |
   +----------+--------------+----------+----------------+------------------------------------------+
   |    ❌    |      ✅      |    ❌    |   ❌ / ✅      | :class:`qibo.result.CircuitResult`       |
   +----------+--------------+----------+----------------+------------------------------------------+
   | ❌ / ✅  |      ❌      | ❌ / ✅  |       ✅       | :class:`qibo.result.QuantumState`        |
   +----------+--------------+----------+----------------+------------------------------------------+
   | ❌ / ✅  |      ✅      | ❌ / ✅  |       ❌       | :class:`qibo.result.MeasurementOutcomes` |
   +----------+--------------+----------+----------------+------------------------------------------+
   | ❌ / ✅  |      ✅      | ❌ / ✅  |       ✅       | :class:`qibo.result.CircuitResult`       |
   +----------+--------------+----------+----------------+------------------------------------------+

Therefore, one of the three objects :class:`qibo.result.QuantumState`,
:class:`qibo.result.MeasurementOutcomes` or :class:`qibo.result.CircuitResult`
is going to be returned by the circuit execution. The first gives acces to the final
state and probabilities via the :meth:`qibo.result.QuantumState.state` and
:meth:`qibo.result.QuantumState.probabilities` methods, whereas the second
allows to retrieve the final samples, the frequencies and the probabilities (calculated
as ``frequencies/nshots``) with the :meth:`qibo.result.MeasurementOutcomes.samples`,
:meth:`qibo.result.MeasurementOutcomes.frequencies` and
:meth:`qibo.result.MeasurementOutcomes.probabilities` methods respectively. The
:class:`qibo.result.CircuitResult` object includes all the above instead.

Every time some measurement is performed at the end of the execution, the result
will be a ``CircuitResult`` unless the final state could not be represented with the
current simulation settings, i.e. if some stochasticity is present in the ciruit
(via noise channels or collapse measurements) and ``density_matrix=False``. In that
case a simple ``MeasurementOutcomes`` object is returned.

If no measurement is appended at the end of the circuit, the final ``QuantumState``
is going to be provided as output. However, if the circuit is stochastic,
``density_matrix`` should be set to ``True`` in order to recover the final state,
otherwise an error is raised.

The final result of the circuit execution can also be saved to disk and loaded back:

.. testsetup::

   from qibo import gates, Circuit

.. testcode::

   circuit = Circuit(2)
   circuit.add(gates.M(0,1))
   # this will be a CircuitResult object
   result = circuit()
   # save it to final_result.npy
   result.dump('final_result.npy')
   # can be loaded back
   from qibo.result import load_result

   loaded_result = load_result('final_result.npy')

.. autoclass:: qibo.result.QuantumState
    :members:
    :member-order: bysource

.. autoclass:: qibo.result.MeasurementOutcomes
    :members:
    :member-order: bysource

.. autoclass:: qibo.result.CircuitResult
    :members:
    :member-order: bysource
