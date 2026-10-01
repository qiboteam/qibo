"""Decompositions of multi-controlled single-qubit gates without auxiliary qubits."""

import math

from numpy.typing import ArrayLike

import qibo
from qibo import gates
from qibo.backends import Backend
from qibo.config import PRECISION_TOL, raise_error
from qibo.transpiler.unitary_decompositions import u3_decomposition


def multi_controlled_decomposition(
    unitary: ArrayLike,
    controls: tuple[int, ...],
    target: int,
    use_toffolis: bool = True,
    backend: Backend = None,
) -> list[gates.Gate]:
    """Decomposes a multi-controlled single-qubit gate without auxiliary qubits.

    The decomposition is exact, including the global phase. Gates with
    :math:`\\det(U) = 1` are decomposed with a number of CNOTs that grows linearly
    with the number of controls following Ref. [1], and other gates quadratically
    following Ref. [2]. This is the decomposition used by
    :meth:`qibo.gates.Gate.decompose` for gates controlled by more than one qubit.

    Args:
        unitary (ArrayLike): :math:`2 \\times 2` unitary matrix of the target gate.
        controls (tuple[int, ...]): Ids of the control qubits.
        target (int): Id of the target qubit.
        use_toffolis (bool, optional): If ``False``, :class:`qibo.gates.TOFFOLI`
            gates are decomposed into :class:`qibo.gates.CNOT` and one-qubit gates.
            Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Gates that have the same effect as
        the multi-controlled gate.

    References:
        1. R. Vale, T. M. D. Azevedo, I. C. S. Araújo, I. F. Araujo, and A. J. da Silva,
        *Circuit Decomposition of Multi-Controlled Special Unitary Single-Qubit Gates*,
        `IEEE Trans. Comput.-Aided Des. Integr. Circuits Syst.
        <https://doi.org/10.1109/TCAD.2023.3327102>`_.

        2. A. J. da Silva and D. K. Park, *Linear-depth quantum circuits for multiqubit
        controlled gates*, `Phys. Rev. A 106, 042602 (2022)
        <https://doi.org/10.1103/PhysRevA.106.042602>`_.
    """
    backend = qibo.backends._check_backend(backend)

    controls = tuple(controls)
    if not controls:
        raise_error(ValueError, "At least one control qubit is needed.")

    if len(controls) == 1:
        decomposition = _controlled_u2(unitary, controls[0], target, backend)
    elif backend.abs(backend.det(unitary) - 1.0) < PRECISION_TOL:
        decomposition = _ldmcsu(unitary, controls, target, backend)
    else:
        decomposition = _ldmcu(unitary, controls, target, backend)

    if use_toffolis:
        return decomposition

    return [
        sub_gate
        for gate in decomposition
        for sub_gate in (
            gate.decompose() if isinstance(gate, gates.TOFFOLI) else [gate]
        )
    ]


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
    steps = [("p", q, 1) for q in range(4)] + [
        ("cx", 0, 1), ("p", 1, -1), ("cx", 0, 1), ("cx", 1, 2), ("p", 2, -1),
        ("cx", 0, 2), ("p", 2, 1), ("cx", 1, 2), ("p", 2, -1), ("cx", 0, 2),
        ("cx", 2, 3), ("p", 3, -1), ("cx", 1, 3), ("p", 3, 1), ("cx", 2, 3),
        ("p", 3, -1), ("cx", 0, 3), ("p", 3, 1), ("cx", 2, 3), ("p", 3, -1),
        ("cx", 1, 3), ("p", 3, 1), ("cx", 2, 3), ("p", 3, -1), ("cx", 0, 3),
    ]  # fmt: skip

    return [
        gates.H(target),
        *(
            (
                gates.U1(qubits[first], second * angle)
                if kind == "p"
                else gates.CNOT(qubits[first], qubits[second])
            )
            for kind, first, second in steps
        ),
        gates.H(target),
    ]


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
    backend = qibo.backends._check_backend(backend)

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
    backend = qibo.backends._check_backend(backend)

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


def _ldmcsu(
    unitary: ArrayLike,
    controls: tuple[int, ...],
    target: int,
    backend: Backend = None,
) -> list[gates.Gate]:
    """Decomposes a multi-controlled gate with :math:`\\det(U) = 1` using the controls
    themselves as dirty auxiliary qubits.

    The controls are split in two halves, and each half borrows the qubits of the
    other for a multi-controlled :math:`X` gate (see Ref. [1] in
    :func:`qibo.transpiler.multicontrolled_decompositions.multi_controlled_decomposition`).

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
    """
    backend = qibo.backends._check_backend(backend)

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

    body = []
    for step in range(2):
        second_gates = _mcx_vchain_dirty(*second)
        body += [
            *_mcx_vchain_dirty(*first),
            gates.U3(target, *params_a),
            *(
                [gate.dagger() for gate in second_gates[::-1]]
                if step == 0
                else second_gates
            ),
            gates.U3(target, *params_a).dagger(),
        ]

    return [*before, *body, *after]


def _ldmcu(
    unitary: ArrayLike,
    controls: tuple[int, ...],
    target: int,
    backend: Backend = None,
) -> list[gates.Gate]:
    """Decomposes a multi-controlled single-qubit gate using controlled rotations.

    The roots :math:`U^{1 / 2^{k}}` of the gate, which are applied controlled by
    one qubit, are obtained from its eigendecomposition.

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
    """
    backend = qibo.backends._check_backend(backend)

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
                body += _controlled_u2(root, qubits[control], qubits[tgt], backend)
            else:
                body.append(
                    gates.CRX(
                        qubits[control], qubits[tgt], signal * math.pi / root_index
                    )
                )

    return body


def _mcx_vchain_dirty(
    controls: tuple[int, ...], target: int, ancillas: tuple[int, ...]
) -> list[gates.Gate]:
    """Decomposes a multi-controlled :math:`X` gate with :math:`k - 2` dirty auxiliary qubits.

    This is Lemma 8 of `Iten et al., Phys. Rev. A 93, 032318 (2016)
    <https://doi.org/10.1103/PhysRevA.93.032318>`_, using Toffoli gates up to a
    relative phase wherever the phases cancel. The auxiliary qubits can be in any
    state and are left unchanged.

    Args:
        controls (tuple[int, ...]): Ids of the :math:`k` control qubits.
        target (int): Id of the target qubit.
        ancillas (tuple[int, ...]): Ids of the dirty auxiliary qubits. At least
            :math:`k - 2` are needed if :math:`k \\geq 3`.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Gates that have the same effect as
        the original gate.
    """
    num_controls = len(controls)

    if num_controls == 1:
        return [gates.CNOT(controls[0], target)]

    if num_controls == 2:
        return [gates.TOFFOLI(*controls, target)]

    if num_controls == 3:
        return _c3x(controls, target)

    ancillas = ancillas[: num_controls - 2]
    targets = [target, *ancillas[::-1]]
    body = []
    for _ in range(2):
        body.append(gates.TOFFOLI(controls[-1], ancillas[-1], target))
        for i in range(1, num_controls - 2):
            body += _toffoli((controls[-1 - i], ancillas[-1 - i]), targets[i], "right")
        body += _toffoli(controls[:2], targets[-1])
        for i in range(num_controls - 3):
            body += _toffoli((controls[2 + i], ancillas[i]), ancillas[i + 1], "left")

    return body


def _toffoli(
    controls: tuple[int, int], target: int, cancel: str | None = None
) -> list[gates.Gate]:
    """Toffoli gate up to a relative phase, with :math:`3` CNOTs instead of :math:`6`.

    The circuit is two identical halves around a central CNOT, so a half can be
    removed if it is cancelled by a neighbouring circuit.

    Args:
        controls (tuple[int, int]): Ids of the two control qubits.
        target (int): Id of the target qubit.
        cancel (str, optional): If ``"left"`` (``"right"``), the first (second)
            half is removed. If ``None``, the full circuit is returned.
            Defaults to ``None``.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Gates of the requested circuit.
    """
    angle = math.pi / 4

    return [
        *(
            []
            if cancel == "left"
            else [
                gates.RY(target, -angle),
                gates.CNOT(controls[0], target),
                gates.RY(target, -angle),
            ]
        ),
        gates.CNOT(controls[1], target),
        *(
            []
            if cancel == "right"
            else [
                gates.RY(target, angle),
                gates.CNOT(controls[0], target),
                gates.RY(target, angle),
            ]
        ),
    ]
