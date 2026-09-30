import math

import networkx as nx

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import PRECISION_TOL, log, raise_error
from qibo.gates.abstract import SpecialGate
from qibo.models import Circuit
from qibo.transpiler.abstract import Optimizer

# Gates of :class:`qibo.transpiler.optimizer.ParametrizedGateFusion` that are merged
# by composing their angles as the angles of a :class:`qibo.gates.U3` gate.
_EULER_GATES = (gates.U1, gates.U2, gates.U3, gates.CU1, gates.CU2, gates.CU3)


# Fusion rules of :class:`qibo.transpiler.optimizer.ParametrizedGateFusion`.
# Each gate class maps to a tuple ``(added, equal, reversal)``, where ``added`` holds
# the indices of the parameters that add up when two gates of the class are applied in
# sequence, ``equal`` holds the indices of the parameters that must coincide for the
# fusion to be exact, and ``reversal`` describes what happens when the second gate
# lists the same qubits in reversed order: ``1`` means the gate does not change,
# ``-1`` means it is the same gate with the sign of the added parameters flipped, and
# ``None`` means that the reversed gate is a different gate, so it cannot be fused.
_FUSION_RULES = {
    gates.RX: ((0,), (), None),
    gates.RY: ((0,), (), None),
    gates.RZ: ((0,), (), None),
    gates.U1: ((0,), (), None),
    gates.PRX: ((0,), (1,), None),
    gates.U1q: ((0,), (1,), None),
    gates.CRX: ((0,), (), None),
    gates.CRY: ((0,), (), None),
    gates.CRZ: ((0,), (), None),
    gates.CU1: ((0,), (), None),
    gates.RXX: ((0,), (), 1),
    gates.RYY: ((0,), (), 1),
    gates.RZZ: ((0,), (), 1),
    gates.RZX: ((0,), (), None),
    gates.RXXYY: ((0,), (), 1),
    gates.RBS: ((0,), (), -1),
    gates.GIVENS: ((0,), (), -1),
    gates.fSim: ((0, 1), (), 1),
}


# Gates replacing ``T ** k`` for ``k = 0, ..., 7``, in the order they are applied.
_T_RULES = (
    (),
    (gates.T,),
    (gates.S,),
    (gates.S, gates.T),
    (gates.Z,),
    (gates.Z, gates.T),
    (gates.SDG,),
    (gates.TDG,),
)


class InverseCancellation(Optimizer):
    """Cancels pairs of consecutive gates whose product is the identity.

    Two gates cancel when no other gate acts on their qubits in between, both act on
    the same qubits in the same order with the same control qubits, and the second gate
    undoes the first one, i.e. the product of their matrices is the identity matrix up
    to ``atol``. Cancellation is applied repeatedly, so nested pairs are removed
    completely: a Pauli-X gate, a Pauli-Y gate, a second Pauli-Y gate and a second
    Pauli-X gate applied in this order on one qubit all disappear.

    The product must equal the identity itself, not merely the identity times a global
    phase, because for gates built with ``controlled_by`` a phase on the target matrix
    is a relative phase on the control qubits. Measurements, alignments, barriers, noise
    channels and fused gates are never cancelled and stop the cancellation across
    them on their qubits. Gates are compared through their matrices at their current parameter
    values, so a cancelled pair of parametrized gates is not tracked by the returned
    circuit anymore.

    Args:
        atol (float, optional): Tolerance on the Frobenius norm (square root of the sum
            of the squared absolute values of all entries) of the difference between
            the product of the two gate matrices and the identity matrix.
            Defaults to :math:`10^{-12}`.

    Example:

        The pair of controlled-NOT (CNOT) gates cancels first, which makes the two
        Hadamard gates on the first qubit consecutive so they cancel as well. The two
        rotations around the X axis on the second qubit cancel because their angles
        add up to zero. The Hadamard gates on the second qubit stay, because the CNOT
        gate between them acts on that qubit.

        .. testcode::

            from qibo import Circuit, gates
            from qibo.transpiler import InverseCancellation

            circuit = Circuit(2)
            circuit.add(gates.H(0))
            circuit.add(gates.CNOT(0, 1))
            circuit.add(gates.CNOT(0, 1))
            circuit.add(gates.H(0))
            circuit.add(gates.RX(1, 0.3))
            circuit.add(gates.RX(1, -0.3))
            circuit.add(gates.X(0))
            circuit.add(gates.H(1))
            circuit.add(gates.CNOT(0, 1))
            circuit.add(gates.H(1))

            circuit.draw()
            print()
            InverseCancellation()(circuit).draw()

        .. testoutput::

            0: ─H─o─o─H──X────o───
            1: ───X─X─RX─RX─H─X─H─

            0: ─X─o───
            1: ─H─X─H─
    """

    def __init__(self, atol: float = 1e-12):
        if atol < 0:
            raise_error(ValueError, f"``atol`` must be non-negative, but got {atol}.")
        self.atol = atol

    def __call__(self, circuit: Circuit, backend: Backend = None) -> Circuit:
        """Remove pairs of consecutive gates that multiply to the identity.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be optimized.
            backend (:class:`qibo.backends.abstract.Backend`, optional): Backend used to
                build the gate matrices. If ``None``, defaults to the global backend.
                Defaults to ``None``.

        Returns:
            :class:`qibo.models.circuit.Circuit`: Circuit without the cancelled pairs.
        """
        backend = _check_backend(backend)

        uncancellable = (gates.M, gates.Align, gates.Channel, SpecialGate)
        # ``kept`` holds the surviving gates in order (cancelled ones are set to
        # ``None``) and ``stacks`` holds, for each qubit, the positions in ``kept``
        # of the surviving gates acting on it. The last position is the latest gate.
        kept = []
        stacks = {qubit: [] for qubit in range(circuit.nqubits)}
        for gate in circuit.queue:
            latest = {
                stacks[qubit][-1] if stacks[qubit] else None for qubit in gate.qubits
            }
            index = latest.pop() if len(latest) == 1 else None
            partner = None if index is None else kept[index]

            cancels = (
                partner is not None
                and not isinstance(gate, uncancellable)
                and not isinstance(partner, uncancellable)
                and partner.qubits == gate.qubits
                and partner.control_qubits == gate.control_qubits
            )
            if cancels:
                first = partner.matrix(backend)
                second = gate.matrix(backend)
                cancels = first.shape == second.shape and (
                    backend.matrix_norm(
                        second @ first - backend.matrices.I(first.shape[0]),
                        order="fro",
                    )
                    <= self.atol
                )

            if cancels:
                kept[index] = None
                for qubit in gate.qubits:
                    stacks[qubit].pop()
            else:
                kept.append(gate)
                for qubit in gate.qubits:
                    stacks[qubit].append(len(kept) - 1)

        new = Circuit(**circuit.init_kwargs)
        new.add([gate for gate in kept if gate is not None])

        return new


class ParametrizedGateFusion(Optimizer):
    """Merges consecutive rotation gates of the same kind into a single gate.

    Two gates are merged when no other gate acts on their qubits in between and both
    are of the same gate class acting on the same qubits with the same control qubits.
    The merged gate has the sum of the rotation angles, for example a
    :class:`qibo.gates.RY` gate with angle :math:`\\alpha` followed by a
    :class:`qibo.gates.RY` gate with angle :math:`\\beta` on the same qubit becomes one
    :class:`qibo.gates.RY` gate with angle :math:`\\alpha + \\beta`. Runs of more than
    two gates are merged completely. The merge is exact, including the global phase.

    The following gates are merged this way:

    * the single-qubit rotations :class:`qibo.gates.RX`, :class:`qibo.gates.RY`,
      :class:`qibo.gates.RZ` and :class:`qibo.gates.U1`;
    * the controlled rotations :class:`qibo.gates.CRX`, :class:`qibo.gates.CRY`,
      :class:`qibo.gates.CRZ` and :class:`qibo.gates.CU1`;
    * the two-qubit rotations :class:`qibo.gates.RXX`, :class:`qibo.gates.RYY`,
      :class:`qibo.gates.RZZ`, :class:`qibo.gates.RZX`, :class:`qibo.gates.RXXYY`,
      :class:`qibo.gates.RBS` and :class:`qibo.gates.GIVENS`;
    * :class:`qibo.gates.fSim`, where both the swap angle and the phase add up;
    * :class:`qibo.gates.PRX` and :class:`qibo.gates.U1q`, where the rotation angle
      adds up as long as the phase angle of both gates is the same up to ``atol``.

    Any of these gates can also be controlled on further qubits with
    ``controlled_by``, as long as both gates have the same control qubits. Gates that
    are symmetric under exchanging their two qubits (such as
    :class:`qibo.gates.RZZ`) are also merged when their qubits are listed in a
    different order, and so are :class:`qibo.gates.RBS` and
    :class:`qibo.gates.GIVENS`, whose angle changes sign when the qubits are
    exchanged.

    The general single-qubit gates :class:`qibo.gates.U2` and :class:`qibo.gates.U3`,
    and their controlled versions :class:`qibo.gates.CU2` and :class:`qibo.gates.CU3`,
    are merged in any combination into a single :class:`qibo.gates.U3` gate, whose
    angles are not the sum of the angles of the two gates. A U3 gate with angles
    :math:`\\theta`, :math:`\\phi` and :math:`\\lambda` applies a rotation around the
    Z axis by :math:`\\lambda`, then a rotation around the Y axis by :math:`\\theta`
    and finally a rotation around the Z axis by :math:`\\phi`. A U2 gate is the U3 gate
    with :math:`\\theta = \\pi / 2`. For a first gate with angles
    :math:`(\\theta_1, \\phi_1, \\lambda_1)` and a second gate with angles
    :math:`(\\theta_2, \\phi_2, \\lambda_2)`, let :math:`h = (\\lambda_2 + \\phi_1) / 2`
    and let :math:`\\alpha` and :math:`\\beta` be the complex numbers

    .. math::
        \\alpha = \\cos h \\, \\cos\\frac{\\theta_1 + \\theta_2}{2}
            - i \\sin h \\, \\cos\\frac{\\theta_2 - \\theta_1}{2}, \\qquad
        \\beta = \\cos h \\, \\sin\\frac{\\theta_1 + \\theta_2}{2}
            - i \\sin h \\, \\sin\\frac{\\theta_2 - \\theta_1}{2}.

    Then the merged gate is the U3 gate with angles

    .. math::
        \\theta' = 2 \\, \\mathrm{atan2}(|\\beta|, |\\alpha|), \\qquad
        \\phi' = \\phi_2 + \\arg\\beta - \\arg\\alpha, \\qquad
        \\lambda' = \\lambda_1 - \\arg\\alpha - \\arg\\beta,

    where :math:`|z|` and :math:`\\arg z` are the modulus and the phase angle of the
    complex number :math:`z`, and :math:`\\mathrm{atan2}` is the two-argument
    arctangent. This merge is exact, including the global phase, for gates without
    control qubits and for controlled gates alike.

    The phase gate :class:`qibo.gates.U1` and its controlled version
    :class:`qibo.gates.CU1` differ from the U3 gate with :math:`\\theta = 0` and
    :math:`\\phi = 0` by the global phase :math:`e^{i \\lambda / 2}`. Therefore
    merging one of them with a U2 or U3 gate, which gives a U2 gate for a U2 gate and
    a U3 gate for a U3 gate, changes the global phase of the circuit. This is only
    done if ``up_to_global_phase`` is ``True`` and the gates have no control qubits,
    because for controlled gates the global phase becomes a relative phase.

    All other gates, such as :class:`qibo.gates.GPI2` or :class:`qibo.gates.U1q` and
    :class:`qibo.gates.PRX` with different phase angles, are left untouched, because
    the product of two of them is not a single gate of the same kind with added
    parameters, or, for :class:`qibo.gates.MS`, could leave the range of allowed
    angles.

    Only gates with the same ``trainable`` flag are merged, and gates with sympy
    parameters are skipped. The merged gate is a new gate, and the input circuit is not
    modified. The merged gate holds fewer parameters than the two gates it replaces, so
    the list of parameters of the returned circuit is shorter than the one of the input
    circuit. Merging is done for the current parameter values, so a later
    ``set_parameters`` call on the returned circuit does not update the angles of the
    original gates independently.

    Args:
        atol (float, optional): Tolerance for deciding that the phase angles of two
            :class:`qibo.gates.PRX` or :class:`qibo.gates.U1q` gates are the same, and
            that the global phase changed by a merge is zero. Defaults to
            :math:`10^{-12}`.
        up_to_global_phase (bool, optional): If ``True``, gates without control
            qubits are also merged when this changes the global phase of the circuit,
            which happens when a :class:`qibo.gates.U1` gate is merged with a
            :class:`qibo.gates.U2` or :class:`qibo.gates.U3` gate. Defaults to
            ``False``.

    Example:

        The two :class:`qibo.gates.RY` gates on the first qubit are merged. The two
        :class:`qibo.gates.RBS` gates act on the same qubits in reversed order, so the
        angle of the second one enters with the opposite sign. The two
        :class:`qibo.gates.RZ` gates on the first qubit are not merged, because the
        CNOT gate acts on that qubit in between.

        .. testcode::

            from qibo import Circuit, gates
            from qibo.transpiler import ParametrizedGateFusion

            circuit = Circuit(3)
            circuit.add(gates.RY(0, 0.1))
            circuit.add(gates.RY(0, 0.2))
            circuit.add(gates.RBS(1, 2, 0.3))
            circuit.add(gates.RBS(2, 1, 0.4))
            circuit.add(gates.RZ(0, 0.5))
            circuit.add(gates.CNOT(0, 1))
            circuit.add(gates.RZ(0, 0.6))

            fused = ParametrizedGateFusion()(circuit)
            for gate in fused.queue:
                print(gate.name, gate.qubits, [round(p, 3) for p in gate.parameters])

        .. testoutput::

            ry (0,) [0.3]
            rbs (1, 2) [-0.1]
            rz (0,) [0.5]
            cx (0, 1) []
            rz (0,) [0.6]

        A :class:`qibo.gates.U3` gate followed by a :class:`qibo.gates.U2` gate gives
        one :class:`qibo.gates.U3` gate, which has the same matrix as the two gates.

        .. testcode::

            import numpy as np

            circuit = Circuit(1)
            circuit.add(gates.U3(0, 0.3, 0.2, 0.1))
            circuit.add(gates.U2(0, 0.4, 0.5))

            fused = ParametrizedGateFusion()(circuit)
            gate = fused.queue[0]
            print(gate.name, [round(float(p), 3) for p in gate.parameters])
            print(np.allclose(fused.unitary(), circuit.unitary()))

        .. testoutput::

            u3 [1.799, 0.597, 0.823]
            True
    """

    def __init__(self, atol: float = 1e-12, up_to_global_phase: bool = False):
        if atol < 0:
            raise_error(ValueError, f"``atol`` must be non-negative, but got {atol}.")
        self.atol = atol
        self.up_to_global_phase = up_to_global_phase

    def __call__(self, circuit: Circuit, backend: Backend = None) -> Circuit:
        """Merge consecutive rotation gates of the same kind.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be optimized.
            backend (:class:`qibo.backends.abstract.Backend`, optional): Backend used to
                compute the angles of the merged gates. If ``None``, defaults to the
                global backend. Defaults to ``None``.

        Returns:
            :class:`qibo.models.circuit.Circuit`: Circuit with the merged gates.
        """
        backend = _check_backend(backend)

        # ``kept`` holds the surviving gates in order and ``stacks`` holds, for each
        # qubit, the positions in ``kept`` of the surviving gates acting on it.
        # The last position is the latest gate.
        kept = []
        stacks = {qubit: [] for qubit in range(circuit.nqubits)}
        for gate in circuit.queue:
            latest = {
                stacks[qubit][-1] if stacks[qubit] else None for qubit in gate.qubits
            }
            index = latest.pop() if len(latest) == 1 else None
            partner = None if index is None else kept[index]

            rule = _FUSION_RULES.get(type(gate))
            same_class = rule is not None and type(partner) is type(gate)
            both_euler = type(gate) in _EULER_GATES and type(partner) in _EULER_GATES
            compatible = (
                (same_class or both_euler)
                and set(partner.qubits) == set(gate.qubits)
                and partner.control_qubits == gate.control_qubits
                and partner.trainable == gate.trainable
                and not partner.symbolic_parameters
                and not gate.symbolic_parameters
            )

            merged = None
            if compatible and same_class:
                added, equal, reversal = rule
                sign = 1 if partner.qubits == gate.qubits else reversal
                if sign is not None and all(
                    backend.abs(partner.parameters[i] - gate.parameters[i]) <= self.atol
                    for i in equal
                ):
                    merged = partner.on_qubits(
                        {qubit: qubit for qubit in partner.qubits}
                    )
                    merged.parameters = tuple(
                        (
                            partner.parameters[i] + sign * gate.parameters[i]
                            if i in added
                            else partner.parameters[i]
                        )
                        for i in range(len(partner.parameters))
                    )
            elif compatible:
                # Different classes among U1, U2 and U3 (or their controlled versions).
                # U1(lam) is U3(0, 0, lam) and U2(phi, lam) is U3(pi / 2, phi, lam),
                # up to the global phase of U1 given in ``_EULER_GATES``.
                angles = []
                for parameters in (partner.parameters, gate.parameters):
                    padding = {1: (0.0, 0.0), 2: (math.pi / 2,), 3: ()}[len(parameters)]
                    angles.append(padding + tuple(parameters))
                (theta1, phi1, lam1), (theta2, phi2, lam2) = angles

                half = (lam2 + phi1) / 2
                plus, minus = (theta1 + theta2) / 2, (theta2 - theta1) / 2
                alpha_re = backend.cos(half) * backend.cos(plus)
                alpha_im = -backend.sin(half) * backend.cos(minus)
                beta_re = backend.cos(half) * backend.sin(plus)
                beta_im = -backend.sin(half) * backend.sin(minus)
                arg_alpha = backend.arctan2(alpha_im, alpha_re)
                arg_beta = backend.arctan2(beta_im, beta_re)
                theta = 2 * backend.arctan2(
                    backend.sqrt(beta_re**2 + beta_im**2),
                    backend.sqrt(alpha_re**2 + alpha_im**2),
                )
                phi = phi2 + arg_beta - arg_alpha
                lam = lam1 - arg_alpha - arg_beta

                # The merge changes the global phase by half the sum of the U1 angles,
                # and this phase is zero when the sine of a quarter of that sum is.
                phase = sum(
                    p[0] for p in (partner.parameters, gate.parameters) if len(p) == 1
                )
                exact = backend.abs(backend.sin(phase / 4)) <= self.atol
                if exact or (self.up_to_global_phase and not gate.control_qubits):
                    if {len(partner.parameters), len(gate.parameters)} == {1, 2}:
                        merged = gates.U2(
                            *gate.target_qubits, phi, lam, trainable=gate.trainable
                        )
                    else:
                        merged = gates.U3(
                            *gate.target_qubits,
                            theta,
                            phi,
                            lam,
                            trainable=gate.trainable,
                        )
                    if gate.control_qubits:
                        merged = merged.controlled_by(*gate.control_qubits)

            if merged is None:
                kept.append(gate)
                for qubit in gate.qubits:
                    stacks[qubit].append(len(kept) - 1)
            else:
                kept[index] = merged

        new = Circuit(**circuit.init_kwargs)
        new.add(kept)

        return new


class Preprocessing(Optimizer):
    """Pad the circuit with unused qubits to match the number of physical qubits.

    Args:
        connectivity (:class:`networkx.Graph`): Hardware connectivity as a graph.
    """

    def __init__(self, connectivity: nx.Graph | None = None):
        self.connectivity = connectivity

    def __call__(self, circuit: Circuit) -> Circuit:
        if not all(qubit in self.connectivity.nodes for qubit in circuit.wire_names):
            default_wire_names = list(self.connectivity.nodes)[: circuit.nqubits]
            log.warning(
                f"Some wire_names in the circuit are not in the connectivity graph. Using wire name {default_wire_names}."
            )
            circuit.wire_names = default_wire_names

        physical_qubits = self.connectivity.number_of_nodes()
        logical_qubits = circuit.nqubits
        if logical_qubits > physical_qubits:
            raise_error(
                ValueError,
                f"The number of qubits in the circuit ({logical_qubits}) "
                + f"can't be greater than the number of physical qubits ({physical_qubits}).",
            )
        if logical_qubits == physical_qubits:
            return circuit

        new_wire_names = circuit.wire_names + list(
            self.connectivity.nodes - circuit.wire_names
        )

        new_circuit = Circuit(nqubits=physical_qubits, wire_names=new_wire_names)
        for gate in circuit.queue:
            new_circuit.add(gate)

        return new_circuit


class Rearrange(Optimizer):
    """Rearranges gates using ``qibo``'s fusion algorithm.
    May reduce number of :class:`qibo.gates.SWAP` when fixing for connectivity
    but this has not been tested.

    Args:
        max_qubits (int, optional): Maximum number of qubits to fuse.
            Defaults to :math:`1`.
    """

    def __init__(self, max_qubits: int = 1):
        self.max_qubits = max_qubits

    def __call__(self, circuit: Circuit, backend: Backend | None = None) -> Circuit:
        backend = _check_backend(backend)
        fused_circuit = circuit.fuse(max_qubits=self.max_qubits)
        new = circuit.__class__(nqubits=circuit.nqubits, wire_names=circuit.wire_names)
        for fgate in fused_circuit.queue:
            if isinstance(fgate, gates.FusedGate):
                new.add(gates.Unitary(fgate.matrix(backend), *fgate.qubits))
            else:
                new.add(fgate)

        return new


class RemoveFinalReset(Optimizer):
    """Removes the resets at the end of a circuit.

    A reset is the single-qubit :class:`qibo.gates.ResetChannel` that sends the
    qubit to the :math:`|0\\rangle` state with probability one, that is with
    :math:`p_0 = 1` and :math:`p_1 = 0`. A reset is removed when no other operation
    (gate, measurement or barrier) acts on its qubit afterwards. Resets that are
    only followed by other final resets are removed as well.

    Such a reset does not change the outcome of any measurement made before it,
    nor the state of the other qubits, but it does change the state of its own qubit
    in the final state returned by the circuit execution.

    Example:

        The last reset is removed, while the first one is kept because the
        Hadamard gate acts on the qubit after it.

        .. testcode::

            from qibo import Circuit, gates
            from qibo.transpiler import RemoveFinalReset

            circuit = Circuit(1, density_matrix=True)
            circuit.add(gates.ResetChannel(0, [1.0, 0.0]))
            circuit.add(gates.H(0))
            circuit.add(gates.ResetChannel(0, [1.0, 0.0]))

            print(RemoveFinalReset()(circuit).ngates)

        .. testoutput::

            2
    """

    def __call__(self, circuit: Circuit) -> Circuit:
        """Remove the resets that are not followed by any operation on their qubit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be optimized.

        Returns:
            :class:`qibo.models.circuit.Circuit`: Circuit without the final resets.
        """
        used = set()
        kept = []
        for gate in reversed(circuit.queue):
            if (
                isinstance(gate, gates.ResetChannel)
                and abs(1 - gate.init_kwargs["p_0"]) <= PRECISION_TOL
                and abs(gate.init_kwargs["p_1"]) <= PRECISION_TOL
                and not used.intersection(gate.qubits)
            ):
                continue
            used.update(gate.qubits)
            kept.append(gate)

        new = Circuit(**circuit.init_kwargs)
        new.add(kept[::-1])

        return new


class RemoveIdentityEquivalent(Optimizer):
    """Removes gates whose action is equivalent to the identity.

    A gate :math:`U`, acting on :math:`n` qubits in total (targets and controls),
    is removed when its average gate fidelity with the identity,

    .. math::
        F(U) = \\frac{|\\text{Tr}(U)|^2 + D}{D \\, (D + 1)},

    satisfies :math:`1 - F(U) \\leq \\epsilon`. Here :math:`D = 2^n` is the
    dimension of the Hilbert space, :math:`\\text{Tr}` is the matrix trace and
    :math:`\\epsilon` is the tolerance. Since :math:`|\\text{Tr}(U)|` does not
    depend on a global phase, an uncontrolled gate that equals the identity up to
    a phase, e.g. a rotation around the Z axis by :math:`2\\pi`, is removed. For a
    gate built with ``controlled_by`` that phase is a relative phase between the
    control states, so the gate is removed only if its target matrix is the identity.

    Measurements, alignments, barriers, noise channels and fused gates are never
    removed. Gates are evaluated at their current parameter values, so a removed
    parametrized gate is not tracked by the returned circuit anymore.

    Args:
        approximation_degree (float, optional): Value in :math:`[0, 1]` that sets the
            tolerance as :math:`\\epsilon = 1 -` ``approximation_degree``. With the
            default value only gates equal to the identity up to numerical precision
            (:math:`\\epsilon = 10^{-14}`) are removed. Defaults to :math:`1.0`.

    Example:

        The rotation by :math:`10^{-3}` radians differs from the identity by an
        infidelity of about :math:`1.7 \\times 10^{-7}`, so it is removed only when
        the tolerance allows it. The identity gate and the rotation by
        :math:`2\\pi` (minus the identity) are removed in both cases.

        .. testcode::

            from math import pi

            from qibo import Circuit, gates
            from qibo.transpiler import RemoveIdentityEquivalent

            circuit = Circuit(2)
            circuit.add(gates.I(0))
            circuit.add(gates.RZ(0, 2 * pi))
            circuit.add(gates.RX(1, 1e-3))
            circuit.add(gates.CNOT(0, 1))

            print(RemoveIdentityEquivalent()(circuit).ngates)
            print(RemoveIdentityEquivalent(approximation_degree=1 - 1e-6)(circuit).ngates)

        .. testoutput::

            2
            1
    """

    def __init__(self, approximation_degree: float = 1.0):
        if not 0.0 <= approximation_degree <= 1.0:
            raise_error(
                ValueError,
                f"``approximation_degree`` must be in [0, 1], but got {approximation_degree}.",
            )
        self.approximation_degree = approximation_degree

    def __call__(self, circuit: Circuit, backend: Backend = None) -> Circuit:
        """Remove the gates that are equivalent to the identity.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be optimized.
            backend (:class:`qibo.backends.abstract.Backend`, optional): Backend used to
                build the gate matrices. If ``None``, defaults to the global backend.
                Defaults to ``None``.

        Returns:
            :class:`qibo.models.circuit.Circuit`: Circuit without the removed gates.
        """
        backend = _check_backend(backend)

        tolerance = max(1 - self.approximation_degree, 1e-14)
        unremovable = (gates.M, gates.Align, gates.Channel, SpecialGate)

        kept = []
        for gate in circuit.queue:
            removable = False
            if not isinstance(gate, unremovable + (gates.FusedGate,)):
                matrix = gate.matrix(backend)
                target_dim = matrix.shape[0]
                control_dim = 2 ** len(gate.control_qubits)
                dim = target_dim * control_dim
                # the controlled gate acts as the identity when a control is off
                trace = (control_dim - 1) * target_dim + backend.trace(matrix)
                fidelity = (backend.abs(trace) ** 2 + dim) / (dim * (dim + 1))
                removable = bool(1 - fidelity <= tolerance)
            if not removable:
                kept.append(gate)

        new = Circuit(**circuit.init_kwargs)
        new.add(kept)

        return new


class RemoveResetInZeroState(Optimizer):
    """Removes the resets acting on qubits that are still in the zero state.

    A reset is the single-qubit :class:`qibo.gates.ResetChannel` that sends the
    qubit to the :math:`|0\\rangle` state with probability one, that is with
    :math:`p_0 = 1` and :math:`p_1 = 0`. A qubit is known to be in the
    :math:`|0\\rangle` state only until the first operation (gate, measurement or
    barrier) acts on it, so only the resets that precede every other operation on
    their qubit are removed. Consecutive resets at the start of a qubit are all
    removed.

    Example:

        The reset at the start of the circuit is removed, while the one after the
        Hadamard gate is kept.

        .. testcode::

            from qibo import Circuit, gates
            from qibo.transpiler import RemoveResetInZeroState

            circuit = Circuit(1, density_matrix=True)
            circuit.add(gates.ResetChannel(0, [1.0, 0.0]))
            circuit.add(gates.H(0))
            circuit.add(gates.ResetChannel(0, [1.0, 0.0]))

            print(RemoveResetInZeroState()(circuit).ngates)

        .. testoutput::

            2
    """

    def __call__(self, circuit: Circuit) -> Circuit:
        """Remove the resets acting on qubits that no operation has touched yet.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be optimized.

        Returns:
            :class:`qibo.models.circuit.Circuit`: Circuit without the removed resets.
        """
        touched = set()
        kept = []
        for gate in circuit.queue:
            if (
                isinstance(gate, gates.ResetChannel)
                and abs(1 - gate.init_kwargs["p_0"]) <= PRECISION_TOL
                and abs(gate.init_kwargs["p_1"]) <= PRECISION_TOL
                and not touched.intersection(gate.qubits)
            ):
                continue
            touched.update(gate.qubits)
            kept.append(gate)

        new = Circuit(**circuit.init_kwargs)
        new.add(kept)

        return new


class ResetAfterMeasureSimplification(Optimizer):
    """Replaces a reset that follows a measurement by a gate conditioned on its outcome.

    A reset is the single-qubit :class:`qibo.gates.ResetChannel` that sends the
    qubit to the :math:`|0\\rangle` state with probability one, that is with
    :math:`p_0 = 1` and :math:`p_1 = 0`. When a reset directly follows a measurement
    of the same qubit, with no other operation on that qubit in between, the measurement
    has already collapsed the qubit to the state :math:`|m\\rangle`, where :math:`m`
    is the measured bit. The reset is then equivalent to a Pauli-X gate applied
    only if :math:`m = 1`. The pass replaces it by a :class:`qibo.gates.U3` gate
    with angles :math:`\\theta = \\lambda = \\pi m` and :math:`\\phi = 0`, which is the
    identity for :math:`m = 0` and exactly the Pauli-X gate for :math:`m = 1`, with
    :math:`m` given by the ``symbols`` of the measurement result, see
    :ref:`collapse-examples`.

    Measurements followed by a gate on the same qubit are collapsing measurements
    in ``qibo``, so every measurement that precedes a reset is handled.

    Example:

        The reset is replaced by a conditional gate, and the circuit ends in the
        zero state for both measurement outcomes.

        .. testcode::

            from qibo import Circuit, gates
            from qibo.transpiler import ResetAfterMeasureSimplification

            circuit = Circuit(1, density_matrix=True)
            circuit.add(gates.H(0))
            circuit.add(gates.M(0))
            circuit.add(gates.ResetChannel(0, [1.0, 0.0]))

            simplified = ResetAfterMeasureSimplification()(circuit)
            print([gate.name for gate in simplified.queue])

        .. testoutput::

            ['h', 'measure', 'u3']
    """

    def __call__(self, circuit: Circuit) -> Circuit:
        """Replace the resets that follow a measurement by conditional gates.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be optimized.

        Returns:
            :class:`qibo.models.circuit.Circuit`: Circuit with the replaced resets.
        """
        kept = []
        # position in ``kept`` of the latest operation on each qubit
        latest = {}
        for gate in circuit.queue:
            if (
                isinstance(gate, gates.ResetChannel)
                and abs(1 - gate.init_kwargs["p_0"]) <= PRECISION_TOL
                and abs(gate.init_kwargs["p_1"]) <= PRECISION_TOL
            ):
                (qubit,) = gate.qubits
                previous = kept[latest[qubit]] if qubit in latest else None
                if isinstance(previous, gates.M):
                    index = sorted(previous.target_qubits).index(qubit)
                    angle = math.pi * previous.result.symbols[index]
                    gate = gates.U3(qubit, theta=angle, phi=0.0, lam=angle)
            kept.append(gate)
            latest.update({qubit: len(kept) - 1 for qubit in gate.qubits})

        new = Circuit(**circuit.init_kwargs)
        new.add(kept)

        return new


class TGateRules(Optimizer):
    """Replaces runs of consecutive :class:`qibo.gates.gates.T` gates by shorter equivalents.

    The :math:`T` gate is the diagonal matrix :math:`\\mathrm{diag}(1, e^{i \\pi / 4})`,
    so :math:`k` consecutive :math:`T` gates on the same qubit give
    :math:`\\mathrm{diag}(1, e^{i k \\pi / 4})`. This only depends on :math:`k \bmod 8`,
    hence eight rules are enough to simplify any number of consecutive :math:`T`
    gates, using the Clifford gates :math:`S = T^2` and :math:`Z = T^4`:

    .. list-table::
        :header-rows: 1

        * - :math:`k \bmod 8`
          - Replacement
        * - 0
          - identity (no gate)
        * - 1
          - :math:`T`
        * - 2
          - :math:`S`
        * - 3
          - :math:`S T`
        * - 4
          - :math:`Z`
        * - 5
          - :math:`Z T`
        * - 6
          - :math:`S^\\dagger`
        * - 7
          - :math:`T^\\dagger`

    All rules are exact, i.e. they hold without any global phase, so they are also
    correct inside larger circuits. Only :math:`T` gates without control qubits are
    replaced. A run of :math:`T` gates on a qubit ends as soon as any other gate,
    including a measurement or a barrier, acts on that qubit; gates acting on other
    qubits do not end it.

    Example:

        Five :math:`T` gates on the first qubit become :math:`Z T`. On the second
        qubit, the controlled-NOT (CNOT) gate ends the run, so the first :math:`T`
        gate stays and the two :math:`T` gates after the CNOT become an :math:`S` gate.

        .. testcode::

            from qibo import Circuit, gates
            from qibo.transpiler import TGateRules

            circuit = Circuit(2)
            circuit.add(gates.T(0) for _ in range(5))
            circuit.add(gates.T(1))
            circuit.add(gates.CNOT(0, 1))
            circuit.add(gates.T(1))
            circuit.add(gates.T(1))

            circuit.draw()
            print()
            TGateRules()(circuit).draw()

        .. testoutput::

            0: ─T─T─T─T─T─o─────
            1: ─T─────────X─T─T─

            0: ─Z─T─o───
            1: ─T───X─S─
    """

    def __call__(self, circuit: Circuit) -> Circuit:
        """Replace runs of consecutive :math:`T` gates using the rules for powers of :math:`T`.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be optimized.

        Returns:
            :class:`qibo.models.circuit.Circuit`: Circuit with the runs of :math:`T`
            gates replaced.
        """
        new = Circuit(**circuit.init_kwargs)
        # ``powers`` maps each qubit to the number of consecutive T gates pending on it.
        # The final ``None`` flushes the runs still pending at the end of the circuit.
        powers = {}
        for gate in [*circuit.queue, None]:
            if isinstance(gate, gates.T) and not gate.control_qubits:
                powers[gate.qubits[0]] = powers.get(gate.qubits[0], 0) + 1
                continue

            for qubit in list(powers) if gate is None else gate.qubits:
                new.add(rule(qubit) for rule in _T_RULES[powers.pop(qubit, 0) % 8])

            if gate is not None:
                new.add(gate)

        return new
