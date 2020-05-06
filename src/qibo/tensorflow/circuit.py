# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import circuit
from qibo.config import DTYPECPX, DTYPEINT
from qibo.tensorflow import gates, measurements, callbacks
from typing import List, Optional, Tuple, Union


class TensorflowCircuit(circuit.BaseCircuit):
    """Implementation of :class:`qibo.base.circuit.BaseCircuit` in Tensorflow.

    Args:
        nqubits (int): Total number of qubits in the circuit.
        dtype: Tensorflow type for complex numbers.
            Read automatically from `config`.
    """

    def __init__(self, nqubits, dtype=DTYPECPX):
        super(TensorflowCircuit, self).__init__(nqubits)
        self.dtype = dtype
        self.compiled_execute = None
        self.callbacks = []

    def __add__(self, circuit: "TensorflowCircuit") -> "TensorflowCircuit":
        return TensorflowCircuit._circuit_addition(self, circuit)

    def _execute_func(self, state: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """Simulates the circuit gates.

        Can be compiled using `tf.function` or used as it is in Eager mode.
        """
        # Calculate callbacks for initial state
        callback_results = [[callback(state)] for callback in self.callbacks]

        for ig, gate in enumerate(self.queue):
            if gate.is_channel and not self.using_density_matrix:
                # Switch from vector to density matrix
                self.using_density_matrix = True
                state = tf.tensordot(state, tf.math.conj(state), axes=0)

            state = gate(state, is_density_matrix=self.using_density_matrix)
            for ic, callback in enumerate(self.callbacks):
                if (ig + 1) % callback.steps == 0:
                    callback_results[ic].append(callback(state))

        # Stack all results for each callback
        callback_results = [tf.stack(r) for r in callback_results]

        return state, callback_results

    def compile(self, callback: Optional[callbacks.Callback] = None):
        """Compiles the circuit as a Tensorflow graph.

        Args:
            callback: A Callback to calculate during circuit execution.
                See :class:`qibo.tensorflow.callbacks.Callback` for more details.
                User can give a single callback or list of callbacks here.
        """
        if self.compiled_execute is not None:
            raise RuntimeError("Circuit is already compiled.")
        self._add_callbacks(callback)
        self.compiled_execute = tf.function(self._execute_func)

    def execute(self,
                initial_state: Optional[Union[np.ndarray, tf.Tensor]] = None,
                nshots: Optional[int] = None,
                callback: Optional[callbacks.Callback] = None
                ) -> Union[tf.Tensor, measurements.CircuitResult]:
        """Propagates the state through the circuit applying the corresponding gates.

        In default usage the full final state vector or density matrix is returned.
        If the circuit contains measurement gates and `nshots` is given, then
        the final state is sampled and the samples are returned.
        Circuit execution uses by default state vectors but switches automatically
        to density matrices if

        Args:
            initial_state (np.ndarray): Initial state vector as a numpy array of shape ``(2 ** nqubits,)``
                or a density matrix of shape ``(2 ** nqubits, 2 ** nqubits)``.
                A Tensorflow tensor with shape ``nqubits * (2,)`` (or ``2 * nqubits * (2,)`` for density matrices)
                is also allowed as an initial state but must have the `dtype` of the circuit.
                If ``initial_state`` is ``None`` the |000...0> state will be used.
            nshots (int): Number of shots to sample if the circuit contains
                measurement gates.
                If ``nshots`` None the measurement gates will be ignored.
            callback: A Callback to calculate during circuit execution.
                See :class:`qibo.tensorflow.callbacks.Callback` for more details.
                User can give a single callback or list of callbacks here.
                Note that if the Circuit is compiled then all callbacks should
                be passed when ``compile`` is called, not during execution.
                Otherwise an ``RuntimeError`` will be raised.

        Returns:
            If ``nshots`` is given and the circuit contains measurements
                A :class:`qibo.base.measurements.CircuitResult` object that contains the measured bitstrings.
            If ``nshots`` is ``None`` or the circuit does not contain measurements.
                The final state vector as a Tensorflow tensor of shape ``(2 ** nqubits,)`` or a density matrix of shape ``(2 ** nqubits, 2 ** nqubits)``.
        """
        if initial_state is None:
            state = self._default_initial_state()

        else:
            def shape_error(shape):
                raise ValueError("Invalid initial state shape {} for circuit "
                                 "with {} qubits.".format(shape, self.nqubits))

            if isinstance(initial_state, np.ndarray):
                shape = initial_state.shape
                if len(shape) == 1:
                    # Assume state vector was given
                    if 2 ** self.nqubits != shape[0]:
                        shape_error(shape)
                    state = tf.cast(initial_state.reshape(self.nqubits * (2,)),
                                    dtype=self.dtype)
                elif len(shape) == 2:
                    # Assume density matrix was given
                    self.using_density_matrix = True
                    if 2 * (2 ** self.nqubits,) != shape:
                        shape_error(shape)
                    state = tf.cast(initial_state.reshape(2 * self.nqubits * (2,)),
                                    dtype=self.dtype)
                else:
                    shape_error(shape)

            elif isinstance(initial_state, tf.Tensor):
                shape = tuple(initial_state.shape)
                if initial_state.dtype != self.dtype:
                    raise TypeError("Circuit is of type {} but initial state is "
                                    "{}.".format(self.dtype, initial_state.dtype))

                if shape == self.nqubits * (2,):
                    state = initial_state
                elif shape == 2 * self.nqubits * (2,):
                    self.using_density_matrix = True
                    state = initial_state
                else:
                    shape_error(shape)
            else:
                raise TypeError("Initial state type {} is not recognized."
                                "".format(type(initial_state)))

        if self.compiled_execute is None:
            self._add_callbacks(callback)
            state, callback_results = self._execute_func(state)
        else:
            if callback is not None:
                raise RuntimeError("Cannot add callbacks to compiled circuit. "
                                   "Please pass the callbacks when compiling.")
            state, callback_results = self.compiled_execute(state)

        # Append callback results to callbacks
        for callback, result in zip(self.callbacks, callback_results):
            callback.append(result)

        if self.measurement_gate is None or nshots is None:
            shape = tf.cast((1+self.using_density_matrix) * (2 ** self.nqubits,),
                            dtype=DTYPEINT)
            self._final_state = tf.reshape(state, shape)
            return self._final_state

        samples = self.measurement_gate(state, nshots, samples_only=True,
                                        is_density_matrix=self.using_density_matrix)
        self._final_state = state

        self.measurement_gate_result = measurements.GateResult(
            self.measurement_gate.qubits, state, decimal_samples=samples)
        return measurements.CircuitResult(
            self.measurement_tuples, self.measurement_gate_result)

    def __call__(self, initial_state: Optional[tf.Tensor] = None,
                 nshots: Optional[int] = None,
                 callback: Optional[callbacks.Callback] = None) -> tf.Tensor:
        """Equivalent to ``circuit.execute``."""
        return self.execute(initial_state=initial_state, nshots=nshots,
                            callback=callback)

    @property
    def final_state(self) -> tf.Tensor:
        """Final state as a Tensorflow tensor of shape (2 ** nqubits,).

        The circuit has to be executed at least once before accessing this
        property, otherwise a `ValueError` is raised. If the circuit is
        executed more than once, only the last final state is returned.
        """
        if self._final_state is None:
            raise RuntimeError("Cannot access final state before the circuit "
                               "is executed.")
        if self.measurement_gate_result is None:
            return self._final_state
        shape = (1 + self.using_density_matrix) * (2 ** self.nqubits,)
        return tf.reshape(self._final_state, shape)

    def with_noise(self, noise_map: circuit.NoiseMapType,
                   measurement_noise: Optional[circuit.NoiseMapType] = None
                   ) -> "TensorflowCircuit":
        """Creates a copy of the circuit with noise gates after each gate.

        Args:
            noise_map (dict): Dictionary that maps qubit ids to noise
                probabilities (px, py, pz).
                If a tuple of probabilities (px, py, pz) is given instead of
                a dictionary, then the same probabilities will be used for all
                qubits.
            measurement_noise (dict): Optional map for using different noise
                probabilities before measurement for the qubits that are
                measured.
                If ``None`` the default probabilities specified by ``noise_map``
                will be used for all qubits.

        Returns:
            Circuit object that contains all the gates of the original circuit
            and additional noise channels on all qubits after every gate.

        Example:
            ::

                from qibo.models import Circuit
                from qibo import gates
                c = Circuit(2)
                c.add([gates.H(0), gates.H(1), gates.CNOT(0, 1)])
                noise_map = {0: (0.1, 0.0, 0.2), 1: (0.0, 0.2, 0.1)}
                noisy_c = c.with_noise(noise_map)

                # ``noisy_c`` will be equivalent to the following circuit
                c2 = Circuit(2)
                c2.add(gates.H(0))
                c2.add(gates.NoiseChannel(0, 0.1, 0.0, 0.2))
                c2.add(gates.NoiseChannel(1, 0.0, 0.2, 0.1))
                c2.add(gates.H(1))
                c2.add(gates.NoiseChannel(0, 0.1, 0.0, 0.2))
                c2.add(gates.NoiseChannel(1, 0.0, 0.2, 0.1))
                c2.add(gates.CNOT(0, 1))
                c2.add(gates.NoiseChannel(0, 0.1, 0.0, 0.2))
                c2.add(gates.NoiseChannel(1, 0.0, 0.2, 0.1))
        """
        return super(TensorflowCircuit, self).with_noise(gates.NoiseChannel,
                                                         noise_map,
                                                         measurement_noise)

    def _default_initial_state(self) -> tf.Tensor:
        """Creates the |000...0> state for default initialization."""
        initial_state = tf.zeros(2 ** self.nqubits, dtype=self.dtype)
        update = tf.constant([1], dtype=self.dtype)
        initial_state = tf.tensor_scatter_nd_update(initial_state,
                                                    tf.constant([[0]], dtype=DTYPEINT),
                                                    update)
        initial_state = tf.reshape(initial_state, self.nqubits * (2,))
        return initial_state

    def _add_callbacks(self, callback: callbacks.Callback):
        """Adds callbacks in the circuit."""
        n = len(self.callbacks)
        if isinstance(callback, list):
            self.callbacks += callback
        elif isinstance(callback, callbacks.Callback):
            self.callbacks.append(callback)
        # Set number of qubits in new callbacks
        for cb in self.callbacks[n:]:
            cb.nqubits = self.nqubits
