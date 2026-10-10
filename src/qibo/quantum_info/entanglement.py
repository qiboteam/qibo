"""Module with entanglement measures."""

import math

import numpy as np
from numpy.typing import ArrayLike

from qibo.backends import Backend, _check_backend
from qibo.config import PRECISION_TOL, raise_error
from qibo.models.circuit import Circuit
from qibo.quantum_info.linalg_operations import (
    matrix_power,
    partial_trace,
    partial_transpose,
)
from qibo.quantum_info.metrics import fidelity, purity


def concurrence(
    state: ArrayLike,
    bipartition: list[int] | tuple[int, ...],
    check_purity: bool = True,
    backend: Backend | None = None,
) -> float:
    """Calculates concurrence of a pure bipartite quantum state
    :math:`\\rho \\in \\mathcal{H}_{A} \\otimes \\mathcal{H}_{B}` as

    .. math::
        C(\\rho) = \\sqrt{2 \\, (\\text{tr}^{2}(\\rho) - \\text{tr}(\\rho_{A}^{2}))} \\, ,

    where :math:`\\rho_{A} = \\text{tr}_{B}(\\rho)` is the reduced density operator
    obtained by tracing out the qubits in the ``bipartition`` :math:`B`.

    Args:
        state (ArrayLike): statevector or density matrix.
        bipartition (list or tuple): qubits in the subsystem to be traced out.
        check_purity (bool, optional): if ``True``, checks if ``state`` is pure. If ``False``,
            it assumes ``state`` is pure. Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Concurrence of :math:`\\rho`.
    """
    backend = _check_backend(backend)

    if (
        (len(state.shape) not in [1, 2])
        or (len(state) == 0)
        or (len(state.shape) == 2 and state.shape[0] != state.shape[1])
    ):
        raise_error(
            TypeError,
            f"state must have dims either (k,) or (k,k), but have dims {state.shape}.",
        )

    if not isinstance(check_purity, bool):
        raise_error(
            TypeError,
            f"check_purity must be type bool, but it is type {type(check_purity)}.",
        )

    if check_purity is True:
        purity_total_system = purity(state, backend=backend)

        if abs(purity_total_system - 1.0) > PRECISION_TOL:
            raise_error(
                NotImplementedError,
                "concurrence only implemented for pure quantum states.",
            )

    reduced_density_matrix = partial_trace(state, bipartition, backend=backend)

    purity_reduced = purity(reduced_density_matrix, backend=backend)

    # purity can exceed 1 due to numerical noise
    return math.sqrt(2 * max(1.0 - purity_reduced, 0.0))


def entanglement_fidelity(
    channel: ArrayLike,
    nqubits: int,
    state: ArrayLike | None = None,
    backend: Backend | None = None,
) -> float:
    """Entanglement fidelity :math:`F_{\\mathcal{E}}` of a ``channel`` :math:`\\mathcal{E}`
    on ``state`` :math:`\\rho` is given by

    .. math::
        F_{\\mathcal{E}}(\\rho) = F(\\rho_{f}, \\rho)

    where :math:`F` is the :func:`qibo.quantum_info.fidelity` function for states,
    and :math:`\\rho_{f} = \\mathcal{E}_{A} \\otimes I_{B}(\\rho)`
    is the state after the channel :math:`\\mathcal{E}` was applied to
    partition :math:`A`.

    Args:
        channel (:class:`qibo.gates.channels.Channel`): quantum channel
            acting on partition :math:`A`.
        nqubits (int): total number of qubits in ``state``.
        state (ArrayLike, optional): statevector or density matrix to be evolved
            by ``channel``. If ``None``, defaults to the maximally entangled state
            :math:`\\frac{1}{\\sqrt{d}} \\, \\sum_{k = 0}^{d - 1} \\, \\ket{k}\\ket{k}`,
            where :math:`d = 2^{n / 2}` and :math:`n` is ``nqubits``, which has to be even.
            The first :math:`n / 2` qubits are paired with the last :math:`n / 2` qubits.
            Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Entanglement fidelity :math:`F_{\\mathcal{E}}`.
    """
    if not isinstance(nqubits, int):
        raise_error(
            TypeError, f"nqubits must be type int, but it is type {type(nqubits)}."
        )

    if nqubits <= 0:
        raise_error(
            ValueError, f"nqubits must be a positive integer, but it is {nqubits}."
        )

    if state is not None and (
        (len(state.shape) not in [1, 2])
        or (len(state) == 0)
        or (len(state.shape) == 2 and state.shape[0] != state.shape[1])
    ):
        raise_error(
            TypeError,
            f"state must have dims either (k,) or (k,k), but have dims {state.shape}.",
        )

    if state is not None and state.shape[-1] != 2**nqubits:
        raise_error(
            ValueError,
            f"state must have dimension 2**nqubits = {2**nqubits}, "
            + f"but it has dimension {state.shape[-1]}.",
        )

    if state is None and nqubits % 2 != 0:
        raise_error(
            ValueError,
            f"nqubits must be even to build the default state, but it is {nqubits}.",
        )

    backend = _check_backend(backend)

    if state is None:
        dim = 2 ** (nqubits // 2)
        state = backend.reshape(backend.identity(dim), (-1,)) / math.sqrt(dim)

    # real-valued states are not compatible with complex Kraus operators
    state = backend.cast(state)

    # apply_channel samples a single trajectory for statevectors,
    # so it has to default to density matrices to be deterministic
    if len(state.shape) == 1:
        state = backend.outer(state, backend.conj(state))

    # some backends apply gates in place, which would overwrite ``state``
    state_final = backend.apply_channel(
        channel, backend.cast(state, copy=True), nqubits
    )

    entang_fidelity = fidelity(state_final, state, backend=backend)

    return entang_fidelity


def entanglement_of_formation(
    state: ArrayLike,
    bipartition: list[int] | tuple[int, ...],
    base: float = 2,
    check_purity: bool = True,
    backend: Backend | None = None,
) -> float:
    """Calculates the entanglement of formation :math:`E_{f}` of a pure bipartite
    quantum state :math:`\\rho`, which is given by

    .. math::
        E_{f} = H([1 - x, x]) \\, ,

    where

    .. math::
        x = \\frac{1 + \\sqrt{1 - C^{2}(\\rho)}}{2} \\, ,

    :math:`C(\\rho)` is the :func:`qibo.quantum_info.concurrence` of :math:`\\rho`,
    and :math:`H` is the :func:`qibo.quantum_info.entropies.shannon_entropy`.

    .. note::
        This expression is valid only if at least one of the two subsystems is a single
        qubit. For pure states of larger subsystems, see
        :func:`qibo.quantum_info.entanglement_entropy`.

    Args:
        state (ArrayLike): statevector or density matrix.
        bipartition (list or tuple): qubits in the subsystem to be traced out.
        base (float): the base of the log in
            :func:`qibo.quantum_info.entropies.shannon_entropy`. Defaults to :math:`2`.
        check_purity (bool, optional): if ``True``, checks if ``state`` is pure. If ``False``,
            it assumes ``state`` is pure. Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: entanglement of formation of state :math:`\\rho`.

    References:
        1. W. K. Wootters, *Entanglement of Formation of an Arbitrary State of Two Qubits*,
           `Phys. Rev. Lett. 80, 2245 <https://doi.org/10.1103/PhysRevLett.80.2245>`_ (1998).
    """
    from qibo.quantum_info.entropies import shannon_entropy

    backend = _check_backend(backend)

    concur = concurrence(
        state, bipartition=bipartition, check_purity=check_purity, backend=backend
    )
    nqubits = int(math.log2(state.shape[-1]))
    if min(len(bipartition), nqubits - len(bipartition)) != 1:
        raise_error(
            NotImplementedError,
            "entanglement of formation only implemented when at least one of the "
            + "subsystems is a single qubit.",
        )

    # concurrence can exceed 1 due to numerical noise
    prob = (1 + math.sqrt(max(1.0 - concur**2, 0.0))) / 2
    probabilities = [1 - prob, prob]

    ent_of_form = shannon_entropy(probabilities, base=base, backend=backend)

    return ent_of_form


def entangling_capability(
    circuit: Circuit,
    samples: int,
    seed: int | None = None,
    backend: Backend | None = None,
) -> float:
    """Return the entangling capability :math:`\\text{Ent}` of a parametrized circuit.

    It is defined as the average Meyer-Wallach entanglement :math:`Q`
    (:func:`qibo.quantum_info.meyer_wallach_entanglement`) of the ``circuit``, i.e.

    .. math::
        \\text{Ent} = \\frac{1}{|\\mathcal{S}|}\\sum_{\\theta_{k} \\in \\mathcal{S}}
            \\, Q(\\rho_{k}) \\, ,

    where :math:`\\mathcal{S}` is the set of sampled circuit parameters,
    and :math:`\\rho_{k}` is the state prepared by the circuit with uniformly-sampled
    parameters :math:`\\theta_{k}`.

    .. note::
        Currently, function does not work with ``circuit`` that contains noisy channels.

    .. note::
        The parameters of ``circuit`` are overwritten by the sampling. After the call,
        ``circuit`` holds the last sampled parameters.

    Args:
        circuit (:class:`qibo.models.Circuit`): Parametrized circuit.
        samples (int): number of sampled circuit parameter vectors :math:`|\\mathcal{S}|`.
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of random
            numbers or a fixed seed to initialize a generator. If ``None``, initializes
            a generator with a random seed. Default: ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Entangling capability :math:`\\text{Ent}`.

    References:
        1. S. Sim, P. D. Johnson, and A. Aspuru-Guzik, *Expressibility and entangling
           capability of parameterized quantum circuits for hybrid quantum-classical
           algorithms*, `Adv. Quantum Technol. 2, 1900070
           <https://doi.org/10.1002/qute.201900070>`_ (2019).
    """

    if not isinstance(samples, int):
        raise_error(
            TypeError, f"samples must be type int, but it is type {type(samples)}."
        )

    if samples <= 0:
        raise_error(
            ValueError, f"samples must be a positive integer, but it is {samples}."
        )

    if (
        seed is not None
        and not isinstance(seed, int)
        and not isinstance(seed, np.random.Generator)
    ):
        raise_error(
            TypeError, "seed must be either type int or numpy.random.Generator."
        )

    backend = _check_backend(backend)

    local_state = (
        backend.default_rng(seed) if seed is None or isinstance(seed, int) else seed
    )

    capability = []
    for _ in range(samples):
        params = backend.random_uniform(
            -math.pi,
            math.pi,
            size=circuit.trainable_gates.nparams,
            seed=local_state,
        )
        circuit.set_parameters(params)
        state = backend.execute_circuit(circuit).state()
        entanglement = meyer_wallach_entanglement(state, backend=backend)
        capability.append(entanglement)

    return np.real(np.sum(capability)) / samples


def logarithmic_negativity(
    state: ArrayLike,
    bipartition: list[int] | tuple[int, ...],
    base: float = 2,
    backend: Backend | None = None,
) -> float:
    """Logarithmic negativity of a bipartite quantum state.

    Given a bipartite state :math:`\\rho \\in \\mathcal{H}_{A} \\otimes \\mathcal{H}_{B}`,
    the logarithmic negativity :math:`E_{N}(\\rho)` is given by

    .. math::
        E_{N}(\\rho) = \\log_{b}\\left( \\|\\rho^{T_{B}}\\|_{1} \\right)
            = \\log_{b}\\left( 2 \\, \\operatorname{Neg}(\\rho) + 1 \\right) \\, ,

    where :math:`b` is ``base``, :math:`\\rho^{T_{B}}` is the partial transpose of
    :math:`\\rho` with respect to ``bipartition`` :math:`B`, :math:`\\|\\cdot\\|_{1}`
    is the Schatten :math:`1`-norm, and :math:`\\operatorname{Neg}(\\rho)` is the
    :func:`qibo.quantum_info.negativity`.

    Args:
        state (ArrayLike): statevector or density matrix.
        bipartition (list or tuple): indices of qubits in partition :math:`B`,
            which is partially transposed.
        base (float, optional): the base of the log. Defaults to :math:`2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Logarithmic negativity :math:`E_{N}(\\rho)` of ``state`` :math:`\\rho`.

    References:
        1. G. Vidal, R. F. Werner, *Computable measure of entanglement*,
           `Phys. Rev. A 65, 032314 <https://doi.org/10.1103/PhysRevA.65.032314>`_ (2002).
    """
    backend = _check_backend(backend)

    if base <= 0.0 or base == 1.0:
        raise_error(ValueError, "log base must be positive and not equal to 1.")

    neg = negativity(state, bipartition, backend=backend)

    return backend.log2(2 * neg + 1) / math.log2(base)


def meyer_wallach_entanglement(
    state: ArrayLike, backend: Backend | None = None
) -> float:
    """Compute the Meyer-Wallach entanglement :math:`Q` of a ``state``,

    .. math::
        Q(\\rho) = 2\\left(1 - \\frac{1}{N} \\, \\sum_{k} \\,
            \\text{tr}\\left(\\rho_{k}^{2}\\right)\\right) \\, ,

    where :math:`\\rho_{k}` is the reduced density matrix of qubit :math:`k`,
    and :math:`N` is the total number of qubits in ``state``.
    We use the definition of the Meyer-Wallach entanglement as the average purity
    proposed in `Brennen (2003) <https://dl.acm.org/doi/10.5555/2011556.2011561>`_,
    which is equivalent to the definition introduced in `Meyer and Wallach (2002)
    <https://doi.org/10.1063/1.1497700>`_.

    Args:
        state (ArrayLike): statevector or density matrix of a pure state.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Meyer-Wallach entanglement :math:`Q`.

    References:
        1. G. K. Brennen, *An observable measure of entanglement for pure states of
           multi-qubit systems*, `Quantum Information and Computation, vol. 3 (6), 619-626
           <https://dl.acm.org/doi/10.5555/2011556.2011561>`_ (2003).

        2. D. A. Meyer and N. R. Wallach, *Global entanglement in multiparticle systems*,
           `J. Math. Phys. 43, 4273–4278 <https://doi.org/10.1063/1.1497700>`_ (2002).
    """
    backend = _check_backend(backend)

    if (
        (len(state.shape) not in [1, 2])
        or (len(state) == 0)
        or (len(state.shape) == 2 and state.shape[0] != state.shape[1])
    ):
        raise_error(
            TypeError,
            f"state must have dims either (k,) or (k,k), but have dims {state.shape}.",
        )

    if abs(purity(state, backend=backend) - 1.0) > PRECISION_TOL:
        raise_error(
            NotImplementedError,
            "meyer_wallach_entanglement only implemented for pure quantum states.",
        )

    nqubits = int(math.log2(state.shape[-1]))

    entanglement = 0
    for j in range(nqubits):
        trace_q = list(range(nqubits))
        trace_q.pop(j)

        rho_r = partial_trace(state, trace_q, backend=backend)

        trace = purity(rho_r, backend=backend)

        entanglement += trace

    return 2 * (1 - entanglement / nqubits)


def negativity(
    state: ArrayLike,
    bipartition: list[int] | tuple[int, ...],
    backend: Backend | None = None,
) -> float:
    """Calculates the negativity of a bipartite quantum state.

    Given a bipartite state :math:`\\rho \\in \\mathcal{H}_{A} \\otimes \\mathcal{H}_{B}`,
    the negativity :math:`\\operatorname{Neg}(\\rho)` is given by

    .. math::
        \\operatorname{Neg}(\\rho) = \\frac{1}{2} \\,
            \\left( \\|\\rho^{T_{B}}\\|_{1} - 1 \\right) \\, ,

    where :math:`\\rho^{T_{B}}` is the partial transpose of :math:`\\rho` with respect to
    partition :math:`B`, and :math:`\\|\\cdot\\|_{1}` is the Schatten :math:`1`-norm
    (also known as nuclear norm or trace norm).

    Args:
        state (ArrayLike): statevector or density matrix.
        bipartition (list or tuple): indices of qubits in partition :math:`B`,
            which is partially transposed.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Negativity :math:`\\operatorname{Neg}(\\rho)` of state :math:`\\rho`.

    References:
        1. G. Vidal, R. F. Werner, *Computable measure of entanglement*,
           `Phys. Rev. A 65, 032314 <https://doi.org/10.1103/PhysRevA.65.032314>`_ (2002).
    """
    backend = _check_backend(backend)

    reduced = partial_transpose(state, bipartition, backend=backend)
    reduced = backend.conj(reduced.T) @ reduced
    norm = backend.trace(matrix_power(reduced, 1 / 2, backend=backend))

    return backend.real((norm - 1) / 2)
