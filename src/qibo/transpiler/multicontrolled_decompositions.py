"""Decompositions of multi-controlled single-qubit gates, with or without auxiliary qubits."""

import math

from numpy.typing import ArrayLike

from qibo import gates
from qibo.backends import Backend, _check_backend, _numpy_backend
from qibo.config import PRECISION_TOL, raise_error
from qibo.transpiler.unitary_decompositions import u3_decomposition


def multi_controlled_decomposition(
    unitary: ArrayLike,
    controls: tuple[int, ...],
    target: int,
    free: tuple[int, ...] = (),
    *,
    clean: tuple[int, ...] = (),
    minimize_toffolis: bool = False,
    minimize_depth: bool = False,
    backend: Backend = None,
) -> list[gates.Gate]:
    """Decomposes a multi-controlled single-qubit gate, with or without auxiliary qubits.

    The decomposition is exact, including the global phase. By default, no
    auxiliary qubits are used. Gates with :math:`\\det(U) = 1` are decomposed with
    a number of CNOTs that grows linearly with the number of controls following
    Ref. [1], and other gates quadratically following Ref. [2]. If ``free`` qubits
    are given, a multi-controlled :math:`X` gate uses them as dirty auxiliary
    qubits, which makes its cost linear following Ref. [3]. Other gates do not
    benefit from auxiliary qubits, and ``free`` and ``clean`` are ignored. This is
    the decomposition used by :meth:`qibo.gates.abstract.Gate.decompose` for gates controlled
    by more than one qubit.

    If ``minimize_toffolis`` is ``True``, a multi-controlled :math:`X` gate with at
    least one auxiliary qubit is decomposed following Ref. [4] instead. This uses
    :math:`2k - 3` Toffoli gates for :math:`k` controls with clean qubits, and
    :math:`4k - 8` with dirty qubits, which is the lowest count, but more CNOTs
    than the default. Two auxiliary qubits make the depth logarithmic in :math:`k`,
    and one makes it linear. Clean qubits are preferred over dirty ones.

    If ``minimize_depth`` is ``True``, a multi-controlled :math:`X` gate with at least
    one auxiliary qubit is decomposed with a depth that is logarithmic in :math:`k`.
    With two or more auxiliary qubits this follows Ref. [4], and with one follows
    Ref. [5] (see also Ref. [6]). The number of gates is linear in :math:`k` in both
    cases, but larger than the default for one auxiliary qubit, so the depth is
    lower only for tens of controls or more.

    Args:
        unitary (ArrayLike): :math:`2 \\times 2` unitary matrix of the target gate.
        controls (tuple[int, ...]): Ids of the control qubits.
        target (int): Id of the target qubit.
        free (tuple[int, ...], optional): Ids of free qubits that can be used as dirty
            auxiliary qubits, that is, they can be in any state and are left
            unchanged. Defaults to ``()``, which uses no auxiliary qubits.
        clean (tuple[int, ...], optional): Ids of free qubits that are known to be in
            the state :math:`\\ket{0}`, and are left in that state. They are used as
            dirty qubits unless ``minimize_toffolis`` is ``True``. Defaults to ``()``.
        minimize_toffolis (bool, optional): If ``True``, a multi-controlled :math:`X`
            gate with auxiliary qubits minimizes the number of Toffoli gates instead
            of the number of CNOTs. Defaults to ``False``.
        minimize_depth (bool, optional): If ``True``, a multi-controlled :math:`X`
            gate with auxiliary qubits minimizes the depth instead of the number of
            CNOTs. It cannot be used together with ``minimize_toffolis``.
            Defaults to ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Backend of
            ``unitary``, which is only used to move it to the CPU, since the
            decomposition is always computed with NumPy. If ``None``, it uses the
            current backend. Defaults to ``None``.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Gates that have the same effect as
        the multi-controlled gate.

    References:
        1. R. Vale, T. M. D. Azevedo, I. C. S. Araújo, I. F. Araujo, and A. J. da Silva,
        *Circuit Decomposition of Multi-Controlled Special Unitary Single-Qubit Gates*,
        `IEEE Trans. Comput.-Aided Des. Integr. Circuits Syst.
        <https://doi.org/10.1109/TCAD.2023.3327102>`_

        2. A. J. da Silva and D. K. Park, *Linear-depth quantum circuits for multiqubit
        controlled gates*, `Phys. Rev. A 106, 042602 (2022)
        <https://doi.org/10.1103/PhysRevA.106.042602>`_.

        3. R. Iten, R. Colbeck, I. Kukuljan, J. Home, and M. Christandl,
        *Quantum circuits for isometries*, `Phys. Rev. A 93, 032318 (2016)
        <https://doi.org/10.1103/PhysRevA.93.032318>`_.

        4. T. Khattar and C. Gidney, *Rise of conditionally clean ancillae for efficient
        quantum circuit constructions*, `Quantum 9, 1752 (2025)
        <https://doi.org/10.22331/q-2025-05-21-1752>`_.

        5. J. Nie, W. Zi, and X. Sun, *Quantum circuit for multi-qubit Toffoli gate with
        optimal resource*, `arXiv:2402.05053 <https://arxiv.org/abs/2402.05053>`_.

        6. V. Vandaele, *Asymptotically optimal quantum circuits for comparators and
        incrementers*, `arXiv:2603.12917 <https://arxiv.org/abs/2603.12917>`_.
    """
    backend = _check_backend(backend)

    # The decomposition of a ``2 x 2`` matrix does not need an accelerator, so it
    # is computed on the CPU with NumPy whatever the backend of ``unitary`` is.
    unitary, backend = backend.to_numpy(unitary), _numpy_backend()

    controls, free, clean = tuple(controls), tuple(free), tuple(clean)
    if not controls:
        raise_error(ValueError, "At least one control qubit is needed.")

    if minimize_toffolis and minimize_depth:
        raise_error(
            ValueError,
            "``minimize_toffolis`` and ``minimize_depth`` cannot be used together.",
        )

    if set(free) & {*controls, target} or set(clean) & {*controls, target, *free}:
        raise_error(
            ValueError,
            "Free qubits cannot coincide with the target or control qubits, "
            + "or be both clean and dirty.",
        )

    if len(controls) == 1:
        decomposition = _controlled_u2(unitary, controls[0], target, backend)
    elif (
        (free or clean)
        and len(controls) > 2
        and backend.matrix_norm(unitary - backend.matrices.X) < PRECISION_TOL
    ):
        if minimize_depth and len(free) + len(clean) >= 2:
            decomposition = _mcx_gidney_log_depth(
                controls, target, (*clean, *free)[:2], len(clean) >= 2
            )
        elif minimize_depth:
            decomposition = _mcx_nie_log_depth(
                controls, target, (*clean, *free)[0], bool(clean)
            )
        elif minimize_toffolis and len(clean) >= 2:
            decomposition = _mcx_gidney_log_depth(controls, target, clean[:2], True)
        elif minimize_toffolis and clean:
            decomposition = _mcx_gidney_linear_depth(controls, target, clean[0], True)
        elif minimize_toffolis and len(free) >= 2:
            decomposition = _mcx_gidney_log_depth(controls, target, free[:2], False)
        elif minimize_toffolis:
            decomposition = _mcx_gidney_linear_depth(controls, target, free[0], False)
        elif len(free) + len(clean) >= len(controls) - 2:
            decomposition = _mcx_vchain_dirty(controls, target, (*free, *clean))
        else:
            decomposition = _linear_mcx(controls, target, (*free, *clean)[0])
    elif backend.abs(backend.det(unitary) - 1.0) < PRECISION_TOL:
        decomposition = _ldmcsu(unitary, controls, target, backend)
    else:
        decomposition = _ldmcu(unitary, controls, target, backend)

    return decomposition


def _c3x(controls: tuple[int, ...], target: int) -> list[gates.Gate]:
    """Decomposes a three-controlled :math:`X` gate into :math:`14` CNOTs.

    Args:
        controls (tuple[int, ...]): Ids of the three control qubits.
        target (int): Id of the target qubit.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Gates that have the same effect as
        the original gate.
    """
    qubits = [*controls, target]
    angle = math.pi / 8
    # ``("p", q, s)`` is a phase gate on ``qubits[q]`` by ``s * angle``,
    # and ``("cx", c, t)`` is a CNOT from ``qubits[c]`` to ``qubits[t]``.
    steps = [("p", q, 1) for q in range(4)]
    steps.extend([
        ("cx", 0, 1), ("p", 1, -1), ("cx", 0, 1), ("cx", 1, 2), ("p", 2, -1),
        ("cx", 0, 2), ("p", 2, 1), ("cx", 1, 2), ("p", 2, -1), ("cx", 0, 2),
        ("cx", 2, 3), ("p", 3, -1), ("cx", 1, 3), ("p", 3, 1), ("cx", 2, 3),
        ("p", 3, -1), ("cx", 0, 3), ("p", 3, 1), ("cx", 2, 3), ("p", 3, -1),
        ("cx", 1, 3), ("p", 3, 1), ("cx", 2, 3), ("p", 3, -1), ("cx", 0, 3),
    ])  # fmt: skip

    c3x = [gates.H(target)]
    c3x.extend(
        (
            gates.U1(qubits[first], second * angle)
            if kind == "p"
            else gates.CNOT(qubits[first], qubits[second])
        )
        for kind, first, second in steps
    )
    c3x.append(gates.H(target))

    return c3x


def _c4x(controls: tuple[int, ...], target: int) -> list[gates.Gate]:
    """Decomposes a four-controlled :math:`X` gate into :math:`36` CNOTs.

    The gate is a relative-phase three-controlled :math:`X` gate (Ref. [2]), its inverse
    and a three-controlled :math:`\\sqrt{X}` gate, surrounded by controlled phases, following
    Lemma 7.5 of Ref. [1].

    Args:
        controls (tuple[int, ...]): Ids of the four control qubits.
        target (int): Id of the target qubit.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Gates that have the same effect as
        the original gate.

    References:
        1. A. Barenco *et al.*, *Elementary gates for quantum computation*,
        `Phys. Rev. A 52, 3457 (1995) <https://doi.org/10.1103/PhysRevA.52.3457>`_.

        2. D. Maslov, *Advantages of using relative-phase Toffoli gates with an application to
        multiple control Toffoli optimization*, `Phys. Rev. A 93, 022311 (2016)
        <https://doi.org/10.1103/PhysRevA.93.022311>`_.
    """
    a, b, c, d = controls
    angle = math.pi / 8

    relative_phase_c3x = [
        gates.H(d), gates.T(d), gates.CNOT(c, d), gates.TDG(d), gates.H(d),
        gates.CNOT(a, d), gates.T(d), gates.CNOT(b, d), gates.TDG(d),
        gates.CNOT(a, d), gates.T(d), gates.CNOT(b, d), gates.TDG(d), gates.H(d),
        gates.T(d), gates.CNOT(c, d), gates.TDG(d), gates.H(d),
    ]  # fmt: skip
    relative_phase_c3x_dagger = [
        gates.H(d), gates.T(d), gates.CNOT(c, d), gates.TDG(d), gates.H(d),
        gates.T(d), gates.CNOT(b, d), gates.TDG(d), gates.CNOT(a, d), gates.T(d),
        gates.CNOT(b, d), gates.TDG(d), gates.CNOT(a, d), gates.H(d), gates.T(d),
        gates.CNOT(c, d), gates.TDG(d), gates.H(d),
    ]  # fmt: skip

    # Three-controlled square root of X. ``("p", q, s)`` is a controlled phase from
    # ``q`` to the target by ``s * angle``, and ``("cx", q, r)`` is a CNOT from ``q``
    # to ``r``. Each step is preceded by a Hadamard gate on the target.
    steps = [
        ("p", a, 1), ("cx", a, b), ("p", b, -1), ("cx", a, b), ("p", b, 1),
        ("cx", b, c), ("p", c, -1), ("cx", a, c), ("p", c, 1), ("cx", b, c),
        ("p", c, -1), ("cx", a, c), ("p", c, 1),
    ]  # fmt: skip
    c3sqrt_x = []
    for kind, first, second in steps:
        c3sqrt_x.append(gates.H(target))
        c3sqrt_x.append(
            gates.CU1(first, target, second * angle)
            if kind == "p"
            else gates.CNOT(first, second)
        )
    c3sqrt_x.append(gates.H(target))

    c4x = [gates.H(target), gates.CU1(d, target, math.pi / 2), gates.H(target)]
    c4x.extend(relative_phase_c3x)
    c4x.extend([gates.H(target), gates.CU1(d, target, -math.pi / 2), gates.H(target)])
    c4x.extend(relative_phase_c3x_dagger)
    c4x.extend(c3sqrt_x)

    return c4x


def _controlled_u2(
    unitary: ArrayLike, control: int, target: int, backend: Backend = None
) -> list[gates.Gate]:
    """Decomposes a singly-controlled single-qubit gate into :math:`2` CNOTs.

    The gate is :math:`e^{i \\gamma} U_{3}`, and the phase :math:`\\gamma` becomes a
    relative phase once controlled, so it is applied as a
    :class:`qibo.gates.U1` gate on the control.

    Args:
        unitary (ArrayLike): :math:`2 \\times 2` unitary matrix of the target gate.
        control (int): Id of the control qubit.
        target (int): Id of the target qubit.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Gates that have the same effect as
        the original gate.
    """
    backend = _check_backend(backend)

    theta, phi, lam = u3_decomposition(unitary, backend)
    u3_matrix = gates.U3(target, theta, phi, lam).matrix(backend)
    overlap = backend.trace(
        backend.matmul(backend.conj(backend.transpose(u3_matrix)), unitary)
    )

    return [
        gates.U1(control, backend.angle(overlap)),
        gates.CU3(control, target, theta, phi, lam),
    ]


def _eig_u2(unitary: ArrayLike, backend: Backend = None) -> tuple:
    """Eigendecomposition of a :math:`2 \\times 2` unitary with unitary eigenvectors.

    A general eigensolver loses orthogonality between the eigenvectors when the
    eigenvalues are close. Here the second eigenvector is set to the exact
    orthogonal complement of the first.

    Args:
        unitary (ArrayLike): :math:`2 \\times 2` unitary matrix.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        tuple(ArrayLike, ArrayLike): Eigenvalues, and a unitary matrix that has
        the eigenvectors as columns.
    """
    backend = _check_backend(backend)

    eigenvalues, eigenvectors = backend.eig(unitary)
    first = eigenvectors[:, 0]
    first = first / backend.sqrt(
        backend.real(backend.matmul(backend.conj(first), first))
    )
    second = [-backend.conj(first[1]), backend.conj(first[0])]
    eigenvectors = backend.cast(
        [[first[0], second[0]], [first[1], second[1]]], dtype=backend.dtype
    )

    return eigenvalues, eigenvectors


def _gidney_log_depth_ladder(
    qubits: tuple[int, ...], skip_conditionally_clean: bool = False
) -> tuple[list[gates.Gate], list[int]]:
    """Accumulates the AND of the controls in logarithmic depth, as in Fig. 4b of Ref. [1].

    The accumulation uses conditionally clean ancillae, which are qubits that are in
    a known state if the qubit that stores the AND of the controls they depend on is on.

    Args:
        qubits (tuple[int, ...]): Ids of the control qubits, followed by the id of the
            auxiliary qubit.
        skip_conditionally_clean (bool, optional): If ``True``, the first Toffoli gate,
            which writes the AND of two controls on the auxiliary qubit, is not
            included. Defaults to ``False``.

    Returns:
        tuple(list[:class:`qibo.gates.abstract.Gate`], list[int]): Gates of the ladder,
        and positions in ``qubits`` of the controls that still have to be combined with
        the auxiliary qubit.

    References:
        1. T. Khattar and C. Gidney, *Rise of conditionally clean ancillae for efficient
        quantum circuit constructions*, `Quantum 9, 1752 (2025)
        <https://doi.org/10.22331/q-2025-05-21-1752>`_.
    """
    auxiliary = len(qubits) - 1
    controls = list(range(auxiliary))
    ancillas, final_controls, ladder = [auxiliary], [], []

    while len(controls) > 1:
        batch_size = min(len(ancillas) + 1, len(controls))
        controls, batch = controls[batch_size:], controls[:batch_size]
        new_ancillas = []
        while len(batch) > 1:
            num_toffolis = len(batch) // 2
            offset = len(batch) % 2
            firsts = batch[offset : offset + num_toffolis]
            seconds = batch[offset + num_toffolis :]
            targets = ancillas[-num_toffolis:]
            if targets != [auxiliary]:
                for first, second, tgt in zip(firsts, seconds, targets):
                    ladder.extend(
                        [
                            gates.X(qubits[tgt]),
                            gates.TOFFOLI(qubits[first], qubits[second], qubits[tgt]),
                        ]
                    )
            elif not skip_conditionally_clean:
                ladder.append(
                    gates.TOFFOLI(
                        qubits[firsts[0]], qubits[seconds[0]], qubits[auxiliary]
                    )
                )
            new_ancillas.extend(batch[offset:])
            targets.extend(batch[:offset])
            batch = targets
            ancillas = ancillas[:-num_toffolis]

        ancillas.extend(new_ancillas)
        ancillas.sort()
        final_controls.extend(batch)

    final_controls.extend(controls)

    return ladder, sorted(final_controls)[:-1]


def _ldmcsu(
    unitary: ArrayLike,
    controls: tuple[int, ...],
    target: int,
    backend: Backend = None,
) -> list[gates.Gate]:
    """Decomposes a multi-controlled gate with :math:`\\det(U) = 1` using the controls
    themselves as dirty auxiliary qubits.

    The controls are split in two halves, and each half borrows the qubits of the
    other for a multi-controlled :math:`X` gate, following Ref. [1].

    Args:
        unitary (ArrayLike): :math:`2 \\times 2` unitary matrix with determinant one.
        controls (tuple[int, ...]): Ids of the control qubits. At least two are needed.
        target (int): Id of the target qubit.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Gates that have the same effect as
        the original gate.

    References:
        1. R. Vale, T. M. D. Azevedo, I. C. S. Araújo, I. F. Araujo, and A. J. da Silva,
        *Circuit Decomposition of Multi-Controlled Special Unitary Single-Qubit Gates*,
        `IEEE Trans. Comput.-Aided Des. Integr. Circuits Syst.
        <https://doi.org/10.1109/TCAD.2023.3327102>`_
    """
    backend = _check_backend(backend)

    atol = 1e-12
    main_is_real = (
        backend.abs(backend.imag(unitary[0, 0])) < atol
        and backend.abs(backend.imag(unitary[1, 1])) < atol
    )
    secondary_is_real = (
        backend.abs(backend.imag(unitary[0, 1])) < atol
        and backend.abs(backend.imag(unitary[1, 0])) < atol
    )

    # The construction needs a matrix with a real secondary diagonal, with entries
    # ``x_value`` (upper right) and ``z_value`` (lower right). Other matrices are
    # brought to this form by a change of basis: Hadamard if only the main diagonal
    # is real, or the eigenbasis (which gives ``x_value = 0``) otherwise. The
    # eigenbasis is also used near ``z_value = -1``, where the closed form of
    # the gate ``A`` below is ill-conditioned.
    if secondary_is_real:
        x_value, z_value = unitary[0, 1], unitary[1, 1]
    else:
        x_value = -backend.real(unitary[0, 1])
        z_value = unitary[1, 1] - 1j * backend.imag(unitary[0, 1])

    if (not main_is_real and not secondary_is_real) or (
        backend.real(z_value) + 1.0 < 1e-2
    ):
        eigenvalues, eigenvectors = _eig_u2(unitary, backend)
        basis_change = u3_decomposition(eigenvectors, backend)
        before = [gates.U3(target, *basis_change).dagger()]
        after = [gates.U3(target, *basis_change)]
        x_value, z_value = 0.0, eigenvalues[1]
    elif not secondary_is_real:
        before, after = [gates.H(target)], [gates.H(target)]
    else:
        before, after = [], []

    # Special unitary ``A``, with entries ``alpha`` and ``beta``, such that the
    # sequence of blocks below multiplies to the original gate.
    if backend.abs(x_value) < atol:
        alpha = backend.exp(1j * backend.angle(z_value) / 4)
        beta = 0.0
    else:
        offset = backend.sqrt((backend.real(z_value) + 1.0) / 2.0)
        denominator = 2.0 * backend.sqrt((backend.real(z_value) + 1.0) * (offset + 1.0))
        alpha = (
            backend.sqrt((offset + 1.0) / 2.0)
            + 1j * backend.imag(z_value) / denominator
        )
        beta = x_value / denominator
    matrix_a = backend.cast(
        [[alpha, -backend.conj(beta)], [beta, backend.conj(alpha)]],
        dtype=backend.dtype,
    )
    params_a = u3_decomposition(matrix_a, backend)

    num_first = (len(controls) + 1) // 2
    num_second = len(controls) // 2
    first = (
        controls[:num_first],
        target,
        controls[num_first : 2 * num_first - 2],
    )
    second = (
        controls[num_first:],
        target,
        controls[num_first - num_second + 2 : num_first],
    )

    decomposition = before
    for step in range(2):
        second_gates = _mcx_vchain_dirty(*second)
        decomposition.extend(_mcx_vchain_dirty(*first))
        decomposition.append(gates.U3(target, *params_a))
        decomposition.extend(
            [gate.dagger() for gate in second_gates[::-1]]
            if step == 0
            else second_gates
        )
        decomposition.append(gates.U3(target, *params_a).dagger())
    decomposition.extend(after)

    return decomposition


def _ldmcu(
    unitary: ArrayLike,
    controls: tuple[int, ...],
    target: int,
    backend: Backend = None,
) -> list[gates.Gate]:
    """Decomposes a multi-controlled single-qubit gate using controlled rotations.

    The roots :math:`U^{1 / 2^{k}}` of the gate, which are applied controlled by
    one qubit, are obtained from its eigendecomposition, following Ref. [1].

    Args:
        unitary (ArrayLike): :math:`2 \\times 2` unitary matrix of the target gate.
        controls (tuple[int, ...]): Ids of the control qubits. At least two are needed.
        target (int): Id of the target qubit.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Gates that have the same effect as
        the original gate.

    References:
        1. A. J. da Silva and D. K. Park, *Linear-depth quantum circuits for multiqubit
        controlled gates*, `Phys. Rev. A 106, 042602 (2022)
        <https://doi.org/10.1103/PhysRevA.106.042602>`_.
    """
    backend = _check_backend(backend)

    qubits = [*controls, target]
    nqubits = len(qubits)
    eigenvalues, eigenvectors = _eig_u2(unitary, backend)
    eigenangles = backend.angle(eigenvalues)
    eigenvectors_dagger = backend.conj(backend.transpose(eigenvectors))

    body = []
    # Each pass is (number of qubits, uses controlled roots, direction).
    for size, first, step in (
        (nqubits, True, 1),
        (nqubits, True, -1),
        (nqubits - 1, False, 1),
        (nqubits - 1, False, -1),
    ):
        pairs = sorted(
            (
                (control, tgt)
                for tgt in range(size)
                for control in range(0 if step == 1 else 1, tgt)
            ),
            key=lambda pair: pair[0] + pair[1],
            reverse=step == 1,
        )

        for control, tgt in pairs:
            exponent = tgt - control - (control == 0)
            signal = step * (-1 if control == 0 and not first else 1)
            root_index = 2**exponent

            if tgt == size - 1 and first:
                phases = backend.exp(1j * signal * eigenangles / root_index)
                root = backend.matmul(
                    backend.matmul(eigenvectors, backend.diag(phases)),
                    eigenvectors_dagger,
                )
                body.extend(_controlled_u2(root, qubits[control], qubits[tgt], backend))
            else:
                body.append(
                    gates.CRX(
                        qubits[control], qubits[tgt], signal * math.pi / root_index
                    )
                )

    return body


def _linear_mcx(
    controls: tuple[int, ...], target: int, ancilla: int
) -> list[gates.Gate]:
    """Decomposes a multi-controlled :math:`X` gate with one dirty auxiliary qubit.

    This is Lemma 9 of Ref. [1]. The controls are split in two
    groups, and the gate is two pairs of smaller multi-controlled :math:`X` gates,
    the first of each pair targeting the auxiliary qubit, which can be in any state
    and is left unchanged.

    Args:
        controls (tuple[int, ...]): Ids of the control qubits. At least four are needed.
        target (int): Id of the target qubit.
        ancilla (int): Id of the dirty auxiliary qubit.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Gates that have the same effect as
        the original gate.

    References:
        1. R. Iten, R. Colbeck, I. Kukuljan, J. Home, and M. Christandl, *Quantum circuits for
        isometries*, `Phys. Rev. A 93, 032318 (2016) <https://doi.org/10.1103/PhysRevA.93.032318>`_.
    """
    num_controls = len(controls)

    if num_controls == 4:
        return _c4x(controls, target)

    if num_controls == 5:
        decomposition = _c3x(controls[:3], ancilla)
        decomposition.extend(_c3x((*controls[3:], ancilla), target))
        decomposition *= 2

        return decomposition

    num_second = math.ceil((num_controls + 2) / 2)
    num_first = num_controls - num_second + 1
    first = (controls[:num_first], ancilla, controls[num_first : 2 * num_first - 2])
    second = (
        (*controls[num_first:], ancilla),
        target,
        controls[num_first - num_second + 2 : num_first],
    )

    decomposition = _mcx_vchain_dirty(*first, relative_phase=True)
    decomposition.extend(_mcx_vchain_dirty(*second))
    decomposition.extend(_mcx_vchain_dirty(*first, relative_phase=True))
    decomposition.extend(_mcx_vchain_dirty(*second))

    return decomposition


def _mcx_gidney_linear_depth(
    controls: tuple[int, ...], target: int, ancilla: int, clean: bool
) -> list[gates.Gate]:
    """Decomposes a multi-controlled :math:`X` gate with one auxiliary qubit in linear depth.

    This is Fig. 3 of Ref. [1] if the ancilla is clean, with :math:`2k - 3` Toffoli
    gates for :math:`k` controls, and Fig. 5 if it is dirty, with :math:`4k - 8`.

    Args:
        controls (tuple[int, ...]): Ids of the control qubits. At least three are needed.
        target (int): Id of the target qubit.
        ancilla (int): Id of the auxiliary qubit.
        clean (bool): If ``True``, ``ancilla`` is in the state :math:`\\ket{0}`. Otherwise
            it can be in any state. It is left unchanged.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Gates that have the same effect as
        the original gate.

    References:
        1. T. Khattar and C. Gidney, *Rise of conditionally clean ancillae for efficient
        quantum circuit constructions*, `Quantum 9, 1752 (2025)
        <https://doi.org/10.22331/q-2025-05-21-1752>`_.
    """
    qubits = (ancilla, *controls)
    size = len(qubits)

    ladder = []
    for i in range(2, size - 2, 2):
        ladder.extend(
            [
                gates.TOFFOLI(qubits[i + 1], qubits[i + 2], qubits[i]),
                gates.X(qubits[i]),
            ]
        )
    first, second, last = (
        (size - 3, size - 5, size - 6) if size % 2 else (size - 1, size - 4, size - 5)
    )
    if last > 0:
        ladder.extend(
            [
                gates.TOFFOLI(qubits[first], qubits[second], qubits[last]),
                gates.X(qubits[last]),
            ]
        )
    for i in range(last, 2, -2):
        ladder.extend(
            [
                gates.TOFFOLI(qubits[i], qubits[i - 1], qubits[i - 2]),
                gates.X(qubits[i - 2]),
            ]
        )

    middle = [gates.TOFFOLI(ancilla, controls[max(0, 6 - size)], target)]
    block = ladder.copy()
    block.extend(middle)
    block.extend(gate.dagger() for gate in ladder[::-1])

    toffoli = gates.TOFFOLI(controls[0], controls[1], ancilla)
    decomposition = [toffoli]
    decomposition.extend(block)
    decomposition.append(toffoli)
    if not clean:
        decomposition.extend(block)

    return decomposition


def _mcx_gidney_log_depth(
    controls: tuple[int, ...], target: int, ancillas: tuple[int, int], clean: bool
) -> list[gates.Gate]:
    """Decomposes a multi-controlled :math:`X` gate with two auxiliary qubits in logarithmic depth.

    This is Fig. 4 of Ref. [1] if the ancillae are clean, with :math:`2k - 3` Toffoli
    gates for :math:`k` controls, and Fig. 6 if they are dirty, with :math:`4k - 8`.

    Args:
        controls (tuple[int, ...]): Ids of the control qubits. At least three are needed.
        target (int): Id of the target qubit.
        ancillas (tuple[int, int]): Ids of the two auxiliary qubits.
        clean (bool): If ``True``, ``ancillas`` are in the state :math:`\\ket{0}`.
            Otherwise they can be in any state. They are left unchanged.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Gates that have the same effect as
        the original gate.

    References:
        1. T. Khattar and C. Gidney, *Rise of conditionally clean ancillae for efficient
        quantum circuit constructions*, `Quantum 9, 1752 (2025)
        <https://doi.org/10.22331/q-2025-05-21-1752>`_.
    """
    body = []
    for skip_conditionally_clean in (False,) if clean else (False, True):
        ladder, positions = _gidney_log_depth_ladder(
            (*controls, ancillas[0]), skip_conditionally_clean
        )
        final_controls = tuple(controls[i] for i in positions)
        if len(final_controls) == 1:
            middle = [gates.TOFFOLI(ancillas[0], final_controls[0], target)]
        else:
            middle = _mcx_gidney_linear_depth(
                (ancillas[0], *final_controls), target, ancillas[1], True
            )
        body.extend(ladder)
        body.extend(middle)
        body.extend(gate.dagger() for gate in ladder[::-1])

    return body


def _mcx_nie_log_depth(
    controls: tuple[int, ...], target: int, ancilla: int, clean: bool
) -> list[gates.Gate]:
    """Decomposes a multi-controlled :math:`X` gate with one auxiliary qubit in logarithmic depth.

    Four controls are first accumulated on ``ancilla``, which makes them conditionally
    clean. Two of them are targets and the other two are auxiliary qubits of two
    parallel decompositions of half the size, in which the remaining controls are
    accumulated. For a dirty ``ancilla`` the construction is repeated, as in Ref. [1].
    The number of gates is linear in the number of controls.

    Args:
        controls (tuple[int, ...]): Ids of the control qubits.
        target (int): Id of the target qubit.
        ancilla (int): Id of the auxiliary qubit.
        clean (bool): If ``True``, ``ancilla`` is in the state :math:`\\ket{0}`. Otherwise
            it can be in any state. It is left unchanged.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Gates that have the same effect as
        the original gate.

    References:
        1. J. Nie, W. Zi, and X. Sun, *Quantum circuit for multi-qubit Toffoli gate with optimal
        resource*, `arXiv:2402.05053 <https://arxiv.org/abs/2402.05053>`_.

        2. V. Vandaele, *Asymptotically optimal quantum circuits for comparators and
        incrementers*, `arXiv:2603.12917 <https://arxiv.org/abs/2603.12917>`_.
    """
    accumulate, workspace, flip = _nie_first_half(controls, target, ancilla)
    if not accumulate:
        return flip

    block = workspace.copy()
    block.extend(flip)
    block.extend(gate.dagger() for gate in workspace[::-1])

    decomposition = []
    for part in (
        (accumulate, block, accumulate)
        if clean
        else (block, accumulate, block, accumulate)
    ):
        decomposition.extend(part)

    return decomposition


def _mcx_vchain_dirty(
    controls: tuple[int, ...],
    target: int,
    ancillas: tuple[int, ...],
    relative_phase: bool = False,
) -> list[gates.Gate]:
    """Decomposes a multi-controlled :math:`X` gate with :math:`k - 2` dirty auxiliary qubits.

    This is Lemma 8 of Ref. [1], using Toffoli gates up to a relative phase
    (Ref. [2]) wherever the phases cancel. The auxiliary qubits can be in any
    state and are left unchanged.

    Args:
        controls (tuple[int, ...]): Ids of the :math:`k` control qubits.
        target (int): Id of the target qubit.
        ancillas (tuple[int, ...]): Ids of the dirty auxiliary qubits. At least
            :math:`k - 2` are needed if :math:`k \\geq 3`.
        relative_phase (bool, optional): If ``True``, the gate is implemented up to
            a diagonal operator, with fewer CNOTs. Defaults to ``False``.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Gates that have the same effect as
        the original gate.

    References:
        1. R. Iten, R. Colbeck, I. Kukuljan, J. Home, and M. Christandl, *Quantum circuits for
        isometries*, `Phys. Rev. A 93, 032318 (2016) <https://doi.org/10.1103/PhysRevA.93.032318>`_.

        2. D. Maslov, *Advantages of using relative-phase Toffoli gates with an application to
        multiple control Toffoli optimization*, `Phys. Rev. A 93, 022311 (2016)
        <https://doi.org/10.1103/PhysRevA.93.022311>`_.
    """
    num_controls = len(controls)

    if num_controls == 1:
        return [gates.CNOT(controls[0], target)]

    if num_controls == 2:
        return [gates.TOFFOLI(*controls, target)]

    if num_controls == 3 and not relative_phase:
        return _c3x(controls, target)

    ancillas = ancillas[: num_controls - 2]
    targets = [target, *ancillas[::-1]]
    body = []
    for step in range(2):
        for i in range(num_controls - 2):
            if i > 0 or relative_phase:
                cancel = "left" if relative_phase and i == 0 and step == 1 else "right"
                body.extend(
                    _toffoli((controls[-1 - i], ancillas[-1 - i]), targets[i], cancel)
                )
            else:
                body.append(gates.TOFFOLI(controls[-1], ancillas[-1], target))
        body.extend(_toffoli(controls[:2], targets[-1]))
        for i in range(num_controls - 3):
            body.extend(
                _toffoli((controls[2 + i], ancillas[i]), ancillas[i + 1], "left")
            )

    return body


def _nie_first_half(
    controls: tuple[int, ...], target: int, ancilla: int
) -> tuple[list[gates.Gate], list[gates.Gate], list[gates.Gate]]:
    """First half of the logarithmic-depth decomposition of Ref. [1] of a multi-controlled :math:`X`.

    It computes the AND of the controls on ``target`` if ``target`` and ``ancilla``
    are in the state :math:`\\ket{0}`, and leaves the other qubits in a state that
    the inverse of the second list restores. With five controls or less, the gate is
    decomposed directly, and the first two lists are empty.

    Args:
        controls (tuple[int, ...]): Ids of the control qubits.
        target (int): Id of the target qubit.
        ancilla (int): Id of the auxiliary qubit.

    Returns:
        tuple(list[:class:`qibo.gates.abstract.Gate`], ...): Gates that accumulate four
        controls on ``ancilla``, gates that accumulate the others on two of the four
        controls, and gates that write the result on ``target``.

    References:
        1. J. Nie, W. Zi, and X. Sun, *Quantum circuit for multi-qubit Toffoli gate with optimal
        resource*, `arXiv:2402.05053 <https://arxiv.org/abs/2402.05053>`_.
    """
    num_controls = len(controls)
    if num_controls <= 5:
        if num_controls == 1:
            decomposition = [gates.CNOT(controls[0], target)]
        elif num_controls == 2:
            decomposition = [gates.TOFFOLI(*controls, target)]
        elif num_controls == 3:
            decomposition = _c3x(controls, target)
        elif num_controls == 4:
            decomposition = _c4x(controls, target)
        else:
            decomposition = _linear_mcx(controls, target, ancilla)

        return [], [], decomposition

    first, rest = controls[:4], controls[4:]
    half = len(rest) // 2
    workspace = [gates.X(qubit) for qubit in first]
    for group, group_target, group_ancilla in (
        (rest[:half], first[0], first[1]),
        (rest[half:], first[2], first[3]),
    ):
        for part in _nie_first_half(group, group_target, group_ancilla):
            workspace.extend(part)

    return (
        _c4x(first, ancilla),
        workspace,
        _c3x((ancilla, first[0], first[2]), target),
    )


def _toffoli(
    controls: tuple[int, int], target: int, cancel: str | None = None
) -> list[gates.Gate]:
    """Toffoli gate up to a relative phase, with :math:`3` CNOTs instead of :math:`6`.

    The circuit is two identical halves around a central CNOT, so a half can be
    removed if it is cancelled by a neighbouring circuit. See Sec. 6.2 of Ref. [1] and
    Ref. [2].

    Args:
        controls (tuple[int, int]): Ids of the two control qubits.
        target (int): Id of the target qubit.
        cancel (str, optional): If ``"left"`` (``"right"``), the first (second)
            half is removed. If ``None``, the full circuit is returned.
            Defaults to ``None``.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Gates of the requested circuit.

    References:
        1. A. Barenco *et al.*, *Elementary gates for quantum computation*,
        `Phys. Rev. A 52, 3457 (1995) <https://doi.org/10.1103/PhysRevA.52.3457>`_.

        2. D. Maslov, *Advantages of using relative-phase Toffoli gates with an application to
        multiple control Toffoli optimization*, `Phys. Rev. A 93, 022311 (2016)
        <https://doi.org/10.1103/PhysRevA.93.022311>`_.
    """
    angle = math.pi / 4
    first_half = [
        gates.RY(target, -angle),
        gates.CNOT(controls[0], target),
        gates.RY(target, -angle),
    ]
    second_half = [
        gates.RY(target, angle),
        gates.CNOT(controls[0], target),
        gates.RY(target, angle),
    ]

    toffoli = []
    if cancel != "left":
        toffoli.extend(first_half)

    toffoli.append(gates.CNOT(controls[1], target))

    if cancel != "right":
        toffoli.extend(second_half)

    return toffoli
