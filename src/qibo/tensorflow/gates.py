# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import gates as base_gates
from qibo.config import einsum, matrices, DTYPEINT, GPU_MEASUREMENT_CUTOFF, CPU_NAME
from typing import List, Optional, Sequence, Tuple


class _ControlCache:
    """Helper tools for `controlled_by` gates.

    This class contains:
      A) an `order` that is used to transpose `state`
         so that control legs are moved in the front
      B) a `targets` list which is equivalent to the
         `target_qubits` tuple but each index is reduced
         by the amount of control qubits that preceed it.
    This method is called by the `nqubits` setter so that the loop runs
    once per gate (and not every time the gate is called).
    """

    def __init__(self, gate):
        self.nqubits = gate.nqubits
        self.order, self.targets = self.calculate(gate)
        # Calculate the reverse order for transposing the state legs so that
        # control qubits are back to their original positions
        self.reverse = self.revert(self.order)

    @staticmethod
    def calculate(gate):
        loop_start = 0
        order = list(gate.control_qubits)
        targets = list(gate.target_qubits)
        for control in gate.control_qubits:
            for i in range(loop_start, control):
                order.append(i)
            loop_start = control + 1

            for i, t in enumerate(gate.target_qubits):
                if t > control:
                    targets[i] -= 1
        for i in range(loop_start, gate.nqubits):
            order.append(i)

        return order, targets

    def revert(self, transpose_order) -> List[int]:
        reverse_order = self.nqubits * [0]
        for i, r in enumerate(transpose_order):
            reverse_order[r] = i
        return reverse_order


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
        # For `controlled_by` gates (see `_ControlCache` for more details)
        self.control_cache = None
        # Gate matrices
        self.matrix = None
        self._matrix_dagger = None

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
            self.control_cache = _ControlCache(self)
            nactive = n - len(self.control_qubits)
            targets = self.control_cache.targets
            self.calculation_cache = self.einsum.create_cache(targets, nactive)
        else:
            self.calculation_cache = self.einsum.create_cache(self.qubits, n)

    @property
    def matrix_dagger(self):
        if self._matrix_dagger is not None:
            return self._matrix_dagger

        n = len(tuple(self.matrix.shape)) // 2
        ids = tuple(range(n, 2 * n)) + tuple(range(n))
        self._matrix_dagger = tf.math.conj(tf.transpose(self.matrix, ids))
        return self._matrix_dagger

    def _is_density_matrix(self, state: tf.Tensor) -> bool:
        shape = tuple(state.shape)
        if len(shape) == self.nqubits:
            return False
        if len(shape) == 2 * self.nqubits:
            return True
        raise ValueError("Gate for {} qubits cannot be applied to a state "
                          "of shape {}.".format(self.nqubits, shape))

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        """Implements the `Gate` on a given state."""
        if self._nqubits is None:
            raise ValueError("Cannot apply gate {} with unspecified number "
                             "of qubits.".format(self.name))

        if self.is_controlled_by:
            return self._controlled_by_call(state)

        return self.einsum(self.calculation_cache, state, self.matrix,
                           is_density_matrix=self._is_density_matrix(state))

    def _controlled_by_call(self, state: tf.Tensor) -> tf.Tensor:
        """Gate __call__ method for `controlled_by` gates."""
        ncontrol = len(self.control_qubits)
        nactive = self.nqubits - ncontrol

        if self._is_density_matrix(state):
            raise NotImplementedError

        # Apply `einsum` only to the part of the state where all controls
        # are active. This should be `state[-1]`
        state = tf.transpose(state, self.control_cache.order)
        state = tf.reshape(state, (2 ** ncontrol,) + nactive * (2,))
        updates = self.einsum(self.calculation_cache, state[-1], self.matrix)

        # Concatenate the updated part of the state `updates` with the
        # part of of the state that remained unaffected `state[:-1]`.
        state = tf.concat([state[:-1], updates[tf.newaxis]], axis=0)
        state = tf.reshape(state, self.nqubits * (2,))
        return tf.transpose(state, self.control_cache.reverse)


class H(TensorflowGate, base_gates.H):

    def __init__(self, q):
        base_gates.H.__init__(self, q)
        TensorflowGate.__init__(self)
        self.matrix = matrices.H


class X(TensorflowGate, base_gates.X):

    def __init__(self, q):
        base_gates.X.__init__(self, q)
        TensorflowGate.__init__(self)
        self.matrix = matrices.X

    def controlled_by(self, *q):
        """Fall back to CNOT and Toffoli if controls are one or two."""
        gate = super(X, self).controlled_by(*q)
        if len(q) == 1:
            return CNOT(q[0], self.target_qubits[0])
        elif len(q) == 2:
            return TOFFOLI(q[0], q[1], self.target_qubits[0])
        return gate


class Y(TensorflowGate, base_gates.Y):

    def __init__(self, q):
        base_gates.Y.__init__(self, q)
        TensorflowGate.__init__(self)
        self.matrix = matrices.Y


class Z(TensorflowGate, base_gates.Z):

    def __init__(self, q):
        base_gates.Z.__init__(self, q)
        TensorflowGate.__init__(self)
        self.matrix = matrices.Z


class M(TensorflowGate, base_gates.M):
    from qibo.tensorflow import measurements

    def __init__(self, *q, register_name: Optional[str] = None):
        base_gates.M.__init__(self, *q, register_name=register_name)
        TensorflowGate.__init__(self)

    @base_gates.Gate.nqubits.setter
    def nqubits(self, n: int):
        base_gates.Gate.nqubits.fset(self, n)

    def __call__(self, state: tf.Tensor, nshots: int,
                 samples_only: bool = False) -> tf.Tensor:
        if self._nqubits is None:
            self.nqubits = len(tuple(state.shape))

        # Trace out unmeasured qubits
        probs_dim = 2 ** len(self.target_qubits)
        probs = tf.reduce_sum(tf.square(tf.abs(state)),
                              axis=self.unmeasured_qubits)
        # Bring probs in the order specified by the user
        probs = tf.transpose(probs, perm=self.reduced_target_qubits)
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

        theta = tf.cast(self.theta, dtype=self.dtype)
        phase = tf.exp(1j * np.pi * theta / 2.0)
        cos = tf.cast(tf.math.real(phase), dtype=self.dtype)
        sin = tf.cast(tf.math.imag(phase), dtype=self.dtype)
        self.matrix = phase * (cos * matrices.I - 1j * sin * matrices.X)


class RY(TensorflowGate, base_gates.RY):

    def __init__(self, q, theta):
        base_gates.RY.__init__(self, q, theta)
        TensorflowGate.__init__(self)

        theta = tf.cast(self.theta, dtype=self.dtype)
        phase = tf.exp(1j * np.pi * theta / 2.0)
        cos = tf.cast(tf.math.real(phase), dtype=self.dtype)
        sin = tf.cast(tf.math.imag(phase), dtype=self.dtype)
        self.matrix = phase * (cos * matrices.I - 1j * sin * matrices.Y)


class RZ(TensorflowGate, base_gates.RZ):

    def __init__(self, q, theta):
        base_gates.RZ.__init__(self, q, theta)
        TensorflowGate.__init__(self)

        theta = tf.cast(self.theta, dtype=self.dtype)
        phase = tf.exp(1j * np.pi * theta)
        rz = tf.eye(2, dtype=self.dtype)
        self.matrix = tf.tensor_scatter_nd_update(rz, [[1, 1]], [phase])

    def controlled_by(self, *q):
        """Fall back to CRZ if control is one."""
        gate = super(RZ, self).controlled_by(*q)
        if len(q) == 1:
            return CRZ(q[0], self.target_qubits[0], self.theta)
        return gate


class CNOT(TensorflowGate, base_gates.CNOT):

    def __init__(self, q0, q1):
        base_gates.CNOT.__init__(self, q0, q1)
        TensorflowGate.__init__(self)
        self.matrix = matrices.CNOT


class CRZ(TensorflowGate, base_gates.CRZ):

    def __init__(self, q0, q1, theta):
        base_gates.CRZ.__init__(self, q0, q1, theta)
        TensorflowGate.__init__(self)

        theta = tf.cast(self.theta, dtype=self.dtype)
        phase = tf.exp(1j * np.pi * theta)
        crz = tf.eye(4, dtype=self.dtype)
        crz = tf.tensor_scatter_nd_update(crz, [[3, 3]], [phase])
        self.matrix = tf.reshape(crz, 4 * (2,))


class SWAP(TensorflowGate, base_gates.SWAP):

    def __init__(self, q0, q1):
        base_gates.SWAP.__init__(self, q0, q1)
        TensorflowGate.__init__(self)
        self.matrix = matrices.SWAP


class TOFFOLI(TensorflowGate, base_gates.TOFFOLI):

    def __init__(self, q0, q1, q2):
        base_gates.TOFFOLI.__init__(self, q0, q1, q2)
        TensorflowGate.__init__(self)
        self.matrix = matrices.TOFFOLI


class Unitary(TensorflowGate, base_gates.Unitary):

    def __init__(self, unitary, *q, name: Optional[str] = None):
        base_gates.Unitary.__init__(self, unitary, *q, name=name)
        TensorflowGate.__init__(self)

        rank = 2 * len(self.target_qubits)
        # This reshape will raise an error if the number of target qubits
        # given is incompatible to the shape of the given unitary.
        self.matrix = tf.convert_to_tensor(self.unitary, dtype=self.dtype)
        self.matrix = tf.reshape(self.matrix, rank * (2,))


class Flatten(TensorflowGate, base_gates.Flatten):

    def __init__(self, coefficients):
        base_gates.Flatten.__init__(self, coefficients)

    def __call__(self, state):
        if self.nqubits is None:
            self.nqubits = len(tuple(state.shape))

        if len(self.coefficients) != 2 ** self.nqubits:
                raise ValueError(
                    "Circuit was created with {} qubits but the "
                    "flatten layer state has {} coefficients."
                    "".format(self.nqubits, self.coefficients)
                )

        _state = np.array(self.coefficients).reshape(self.nqubits * (2,))
        return tf.convert_to_tensor(_state, dtype=state.dtype)
