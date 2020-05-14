# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import gates as base_gates
from qibo.base import cache
from qibo.config import einsum, matrices, DTYPEINT, DTYPE, GPU_MEASUREMENT_CUTOFF, CPU_NAME
from typing import Optional, Sequence, Tuple


class TensorflowGate(base_gates.Gate):
    """The base Tensorflow gate.

    **Properties:**
        matrix: The matrix that represents the gate to be applied.
            This is (2, 2) for 1-qubit gates and (4, 4) for 2-qubit gates.
        qubits: List with the qubits that the gate is applied to.
    """

    dtype = matrices.dtype
    einsum = einsum

    def __init__(self):
        self.calculation_cache = None
        # For `controlled_by` gates (see `cache.ControlCache` for more details)
        self.control_cache = None
        # Gate matrices
        self.matrix = None

    def _construct_matrix(self):
        """Constructs the gate matrix as ``tf.Tensor``.

        This is called automatically by ``nqubits`` setter.
        """
        pass

    def with_backend(self, einsum_choice: str) -> "TensorflowGate":
        """Uses a different einsum backend than the one defined in config.

        Useful for testing.

        Args:
            einsum_choice: Which einsum backend to use.
                One of `DefaultEinsum` or `MatmulEinsum`.

        Returns:
            The gate object with the calculation backend switched to the
            selection.
        """
        from qibo.tensorflow import einsum
        self.einsum = getattr(einsum, einsum_choice)()
        return self

    @base_gates.Gate.nqubits.setter
    def nqubits(self, n: int):
        """Sets the number of qubit that this gate acts on.

        This is called automatically by the `Circuit.add` method if the gate
        is used on a `Circuit`. If the gate is called on a state then `nqubits`
        is set during the first `__call__`.
        When `nqubits` is set we also calculate the einsum string so that it
        is calculated only once per gate.
        """
        base_gates.Gate.nqubits.fset(self, n)
        if self.is_controlled_by:
            self.control_cache = cache.ControlCache(self)
            nactive = n - len(self.control_qubits)
            targets = self.control_cache.targets
            self.calculation_cache = self.einsum.create_cache(targets, nactive, ncontrol=len(self.control_qubits))
        else:
            self.calculation_cache = self.einsum.create_cache(self.qubits, n)
        self.calculation_cache.cast_shapes(lambda x: tf.cast(x, dtype=DTYPEINT))
        self._construct_matrix()

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        """Implements the `Gate` on a given state."""
        if self._nqubits is None:
            if is_density_matrix:
                self.nqubits = len(tuple(state.shape)) // 2
            else:
                self.nqubits = len(tuple(state.shape))

        if self.is_controlled_by:
            return self._controlled_by_call(state, is_density_matrix)

        if is_density_matrix:
            state = self.einsum(self.calculation_cache.right, state,
                                tf.math.conj(self.matrix))
            state = self.einsum(self.calculation_cache.left, state, self.matrix)
            return state

        return self.einsum(self.calculation_cache.vector, state, self.matrix)

    def _controlled_by_call(self, state: tf.Tensor,
                            is_density_matrix: bool = False) -> tf.Tensor:
        """Gate __call__ method for `controlled_by` gates."""
        ncontrol = len(self.control_qubits)
        nactive = self.nqubits - ncontrol

        transpose_order = self.control_cache.order(is_density_matrix)
        reverse_transpose_order = self.control_cache.reverse(is_density_matrix)

        state = tf.transpose(state, transpose_order)
        if is_density_matrix:
            state = tf.reshape(state, 2 * (2 ** ncontrol,) + 2 * nactive * (2,))

            updates01 = self.einsum(self.calculation_cache.right0,
                                    state[:-1, -1],
                                    tf.math.conj(self.matrix))
            updates10 = self.einsum(self.calculation_cache.left0,
                                    state[-1, :-1],
                                    self.matrix)

            updates11 = self.einsum(self.calculation_cache.right,
                                    state[-1, -1], tf.math.conj(self.matrix))
            updates11 = self.einsum(self.calculation_cache.left,
                                    updates11, self.matrix)

            updates01 = tf.concat([state[:-1, :-1], updates01[:, tf.newaxis]], axis=1)
            updates10 = tf.concat([updates10, updates11[tf.newaxis]], axis=0)
            state = tf.concat([updates01, updates10[tf.newaxis]], axis=0)
            state = tf.reshape(state, 2 * self.nqubits * (2,))

        else:
            # Apply `einsum` only to the part of the state where all controls
            # are active. This should be `state[-1]`
            state = tf.reshape(state, (2 ** ncontrol,) + nactive * (2,))
            updates = self.einsum(self.calculation_cache.vector, state[-1],
                                  self.matrix)
            # Concatenate the updated part of the state `updates` with the
            # part of of the state that remained unaffected `state[:-1]`.
            state = tf.concat([state[:-1], updates[tf.newaxis]], axis=0)
            state = tf.reshape(state, self.nqubits * (2,))

        return tf.transpose(state, reverse_transpose_order)


class H(TensorflowGate, base_gates.H):

    def __init__(self, q):
        base_gates.H.__init__(self, q)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        self.matrix = matrices.H


class X(TensorflowGate, base_gates.X):

    def __init__(self, q):
        base_gates.X.__init__(self, q)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        self.matrix = matrices.X

    def controlled_by(self, *q):
        """Fall back to CNOT and Toffoli if controls are one or two."""
        if len(q) == 1:
            gate = CNOT(q[0], self.target_qubits[0])
        elif len(q) == 2:
            gate = TOFFOLI(q[0], q[1], self.target_qubits[0])
        else:
            gate = super(X, self).controlled_by(*q)
        gate.einsum = self.einsum
        return gate


class Y(TensorflowGate, base_gates.Y):

    def __init__(self, q):
        base_gates.Y.__init__(self, q)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        self.matrix = matrices.Y


class Z(TensorflowGate, base_gates.Z):

    def __init__(self, q):
        base_gates.Z.__init__(self, q)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        self.matrix = matrices.Z


class M(TensorflowGate, base_gates.M):
    from qibo.tensorflow import measurements

    def __init__(self, *q, register_name: Optional[str] = None):
        base_gates.M.__init__(self, *q, register_name=register_name)
        TensorflowGate.__init__(self)
        self._traceout = None

    @base_gates.Gate.nqubits.setter
    def nqubits(self, n: int):
        base_gates.Gate.nqubits.fset(self, n)

    @property
    def _traceout_str(self):
        """Einsum string used to trace out when state is density matrix."""
        if self._traceout is None:
            from qibo.tensorflow.einsum import DefaultEinsum
            qubits = set(self.unmeasured_qubits)
            self._traceout = DefaultEinsum.partialtrace_str(
              qubits, self.nqubits, measuring=True)
        return self._traceout

    def _calculate_probabilities(self, state: tf.Tensor,
                                 is_density_matrix: bool = False) -> tf.Tensor:
        """Calculates probabilities from state using Born's rule.

        Args:
            state: State vector of shape nqubits * (2,) or density matrix of
                shape 2 * nqubits * (2,).
            is_density_matrix: Flag that specifies whether `state` is a state
                vector or density matrix.

        Returns:
            Probabilities for measured qubits with shape len(target_qubits)* (2,).
        """
        # Trace out unmeasured qubits
        if is_density_matrix:
            probs = tf.cast(tf.einsum(self._traceout_str, state),
                            dtype=DTYPE)
        else:
            probs = tf.reduce_sum(tf.square(tf.abs(state)),
                                  axis=self.unmeasured_qubits)
        # Bring probs in the order specified by the user
        return tf.transpose(probs, perm=self.reduced_target_qubits)

    def __call__(self, state: tf.Tensor, nshots: int,
                 samples_only: bool = False,
                 is_density_matrix: bool = False) -> tf.Tensor:
        if self._nqubits is None:
            self.nqubits = len(tuple(state.shape)) // (1 + int(is_density_matrix))

        probs_dim = 2 ** len(self.target_qubits)
        probs = self._calculate_probabilities(state, is_density_matrix)
        logits = tf.math.log(tf.reshape(probs, (probs_dim,)))

        if nshots * probs_dim < GPU_MEASUREMENT_CUTOFF:
            # Use default device to perform sampling
            samples_dec = tf.random.categorical(logits[tf.newaxis], nshots,
                                                dtype=DTYPEINT)[0]
        else:
            # Force using CPU to perform sampling because if GPU is used
            # it will cause a `ResourceExhaustedError`
            if CPU_NAME is None:
                raise RuntimeError("Cannot find CPU device to use for sampling.")
            with tf.device(CPU_NAME):
                samples_dec = tf.random.categorical(logits[tf.newaxis], nshots,
                                                    dtype=DTYPEINT)[0]
        if samples_only:
            return samples_dec
        return self.measurements.GateResult(
            self.qubits, state, decimal_samples=samples_dec)


class RX(TensorflowGate, base_gates.RX):

    def __init__(self, q, theta):
        base_gates.RX.__init__(self, q, theta)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        th = tf.cast(self.theta, dtype=self.dtype)
        self.matrix = (tf.cos(th / 2.0) * matrices.I -
                       1j * tf.sin(th / 2.0) * matrices.X)


class RY(TensorflowGate, base_gates.RY):

    def __init__(self, q, theta):
        base_gates.RY.__init__(self, q, theta)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        th = tf.cast(self.theta, dtype=self.dtype)
        self.matrix = (tf.cos(th / 2.0) * matrices.I -
                       1j * tf.sin(th / 2.0) * matrices.Y)


class RZ(TensorflowGate, base_gates.RZ):

    def __init__(self, q, theta):
        base_gates.RZ.__init__(self, q, theta)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        th = tf.cast(self.theta, dtype=self.dtype)
        phase = tf.exp(1j * th / 2.0)[tf.newaxis]
        diag = tf.concat([tf.math.conj(phase), phase], axis=0)
        self.matrix = tf.linalg.diag(diag)


class CNOT(TensorflowGate, base_gates.CNOT):

    def __init__(self, q0, q1):
        base_gates.CNOT.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        self.matrix = matrices.CNOT


class CZPow(TensorflowGate, base_gates.CZPow):

    def __init__(self, q0, q1, theta):
        base_gates.CZPow.__init__(self, q0, q1, theta)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        th = tf.cast(self.theta, dtype=self.dtype)
        phase = tf.exp(1j * th)[tf.newaxis]
        diag = tf.concat([tf.ones(3, dtype=self.dtype), phase], axis=0)
        crz = tf.linalg.diag(diag)
        self.matrix = tf.reshape(crz, 4 * (2,))


class SWAP(TensorflowGate, base_gates.SWAP):

    def __init__(self, q0, q1):
        base_gates.SWAP.__init__(self, q0, q1)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        self.matrix = matrices.SWAP


class TOFFOLI(TensorflowGate, base_gates.TOFFOLI):

    def __init__(self, q0, q1, q2):
        base_gates.TOFFOLI.__init__(self, q0, q1, q2)
        TensorflowGate.__init__(self)

    def _construct_matrix(self):
        self.matrix = matrices.TOFFOLI


class Unitary(TensorflowGate, base_gates.Unitary):

    def __init__(self, unitary, *q, name: Optional[str] = None):
        base_gates.Unitary.__init__(self, unitary, *q, name=name)
        TensorflowGate.__init__(self)

        rank = len(self.target_qubits)
        shape = tuple(self.unitary.shape)
        if shape != (2 ** rank, 2 ** rank):
            raise ValueError("Invalid shape {} of unitary matrix acting on "
                             "{} target qubits.".format(shape, rank))

    def _construct_matrix(self):
        rank = len(self.target_qubits)
        self.matrix = tf.convert_to_tensor(self.unitary, dtype=self.dtype)
        self.matrix = tf.reshape(self.matrix, 2 * rank * (2,))


class Flatten(TensorflowGate, base_gates.Flatten):

    def __init__(self, coefficients):
        base_gates.Flatten.__init__(self, coefficients)

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        if self._nqubits is None:
            if is_density_matrix:
                self.nqubits = len(tuple(state.shape)) // 2
            else:
                self.nqubits = len(tuple(state.shape))

        if is_density_matrix:
            shape = 2 * self.nqubits * (2,)
        else:
            shape = self.nqubits * (2,)

        _state = np.array(self.coefficients).reshape(shape)
        return tf.convert_to_tensor(_state, dtype=self.dtype)


class TensorflowChannel(TensorflowGate):
    """Base Tensorflow channels.

    All channels should inherit this class.
    """

    def with_backend(self, einsum_choice: str) -> "TensorflowChannel":
        super(TensorflowChannel, self).with_backend(einsum_choice)
        for gate in self.gates:
            gate.einsum = self.einsum
        return self

    @TensorflowGate.nqubits.setter
    def nqubits(self, n: int):
        self._nqubits = n
        self._nstates = 2 ** n
        for gate in self.gates:
            gate.nqubits = n

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = True
                 ) -> tf.Tensor:
        if not is_density_matrix:
            raise ValueError("Noise channel can only be applied to density "
                             "matrices.")
        if self._nqubits is None:
            self.nqubits = len(tuple(state.shape)) // 2

        return self._krauss_sum(state)

    def _krauss_sum(self, state: tf.Tensor) -> tf.Tensor:
        """Loops over `self.gates` to calculate sum of Krauss operators."""
        raise NotImplementedError


class NoiseChannel(TensorflowChannel, base_gates.NoiseChannel):

    def __init__(self, q: int, px: float = 0, py: float = 0, pz: float = 0):
        base_gates.NoiseChannel.__init__(self, q, px, py, pz)
        TensorflowChannel.__init__(self)
        classes = (X, Y, Z)
        self.gates = [cl(q) for p, cl in zip(self.p, classes) if p > 0]

    def _krauss_sum(self, state: tf.Tensor) -> tf.Tensor:
        new_state = tf.zeros_like(state)
        for p, gate in zip(self.p, self.gates):
            new_state += p * gate(state, is_density_matrix=True)
        return (1 - self.total_p) * state + new_state


class GeneralChannel(TensorflowChannel, base_gates.GeneralChannel):

    def __init__(self, A: Sequence[Tuple[Tuple[int], np.ndarray]]):
        base_gates.GeneralChannel.__init__(self, A)
        TensorflowChannel.__init__(self)
        self.gates = [Unitary(m, *list(q)) for q, m in A]

    def _krauss_sum(self, state: tf.Tensor) -> tf.Tensor:
        new_state = tf.zeros_like(state)
        for gate in self.gates:
            new_state += gate(state, is_density_matrix=True)
        return new_state
