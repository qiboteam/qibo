"""Submodules with entanglement measures."""

import numpy as np

from qibo.backends import _check_backend
from qibo.config import PRECISION_TOL, raise_error
from qibo.quantum_info.linalg_operations import (
    matrix_power,
    partial_trace,
    partial_transpose,
)
from qibo.quantum_info.metrics import fidelity, purity


def concurrence(state, bipartition, check_purity: bool = True, backend=None):
    """Calculates concurrence of a pure bipartite quantum state
    :math:`\\rho \\in \\mathcal{H}_{A} \\otimes \\mathcal{H}_{B}` as

    .. math::
        C(\\rho) = \\sqrt{2 \\, (\\text{tr}^{2}(\\rho) - \\text{tr}(\\rho_{A}^{2}))} \\, ,

    where :math:`\\rho_{A} = \\text{tr}_{B}(\\rho)` is the reduced density operator
    obtained by tracing out the qubits in the ``bipartition`` :math:`B`.

    Args:
        state (ndarray): statevector or density matrix.
        bipartition (list or tuple or ndarray): qubits in the subsystem to be traced out.
        check_purity (bool, optional): if ``True``, checks if ``state`` is pure. If ``False``,
            it assumes ``state`` is pure . Defaults to ``True``.
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

        mixed = bool(abs(purity_total_system - 1.0) > PRECISION_TOL)
        if mixed is True:
            raise_error(
                NotImplementedError,
                "concurrence only implemented for pure quantum states.",
            )

    reduced_density_matrix = partial_trace(state, bipartition, backend=backend)

    purity_reduced = purity(reduced_density_matrix, backend=backend)
    if purity_reduced - 1.0 > 0.0:
        purity_reduced = round(purity_reduced, 7)

    concur = np.sqrt(2 * (1 - purity_reduced))

    return concur


def entanglement_of_formation(
    state, bipartition, base: float = 2, check_purity: bool = True, backend=None
):
    """Calculates the entanglement of formation :math:`E_{f}` of a pure bipartite
    quantum state :math:`\\rho`, which is given by

    .. math::
        E_{f} = H([1 - x, x]) \\, ,

    where

    .. math::
        x = \\frac{1 + \\sqrt{1 - C^{2}(\\rho)}}{2} \\, ,

    :math:`C(\\rho)` is the :func:`qibo.quantum_info.concurrence` of :math:`\\rho`,
    and :math:`H` is the :func:`qibo.quantum_info.entropies.shannon_entropy`.

    Args:
        state (ndarray): statevector or density matrix.
        bipartition (list or tuple or ndarray): qubits in the subsystem to be traced out.
        base (float): the base of the log in :func:`qibo.quantum_info.entropies.shannon_entropy`.
            Defaults to  :math:`2`.
        check_purity (bool, optional): if ``True``, checks if ``state`` is pure. If ``False``,
            it assumes ``state`` is pure . Default: ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.


    Returns:
        float: entanglement of formation of state :math:`\\rho`.
    """
    from qibo.quantum_info.entropies import shannon_entropy  # pylint: disable=C0415

    backend = _check_backend(backend)

    concur = concurrence(
        state, bipartition=bipartition, check_purity=check_purity, backend=backend
    )
    concur = (1 + np.sqrt(1 - concur**2)) / 2
    probabilities = [1 - concur, concur]

    ent_of_form = shannon_entropy(probabilities, base=base, backend=backend)

    return ent_of_form


def negativity(state, bipartition, backend=None):
    """Calculates the negativity of a bipartite quantum state.

    Given a bipartite state :math:`\\rho \\in \\mathcal{H}_{A} \\otimes \\mathcal{H}_{B}`,
    the negativity :math:`\\operatorname{Neg}(\\rho)` is given by

    .. math::
        \\operatorname{Neg}(\\rho) = \\frac{1}{2} \\,
            \\left( \\norm{\\rho_{B}}_{1} - 1 \\right) \\, ,

    where :math:`\\rho_{B}` is the reduced density matrix after tracing out qubits in
    partition :math:`A`, and :math:`\\norm{\\cdot}_{1}` is the Schatten :math:`1`-norm
    (also known as nuclear norm or trace norm).

    Args:
        state (ndarray): statevector or density matrix.
        bipartition (list or tuple or ndarray): qubits in the subsystem to be traced out.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Negativity :math:`\\operatorname{Neg}(\\rho)` of state :math:`\\rho`.
    """
    backend = _check_backend(backend)

    reduced = partial_transpose(state, bipartition, backend)
    reduced = backend.np.conj(reduced.T) @ reduced
    norm = backend.np.trace(matrix_power(reduced, 1 / 2, backend=backend))

    return backend.np.real((norm - 1) / 2)


def entanglement_fidelity(
    channel, nqubits: int, state=None, check_hermitian: bool = False, backend=None
):
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
        state (ndarray, optional): statevector or density matrix to be evolved
            by ``channel``. If ``None``, defaults to the maximally entangled state
            :math:`\\frac{1}{2^{n}} \\, \\sum_{k} \\, \\ket{k}\\ket{k}`, where
            :math:`n` is ``nqubits``. Defaults to ``None``.
        check_hermitian (bool, optional): if ``True``, checks if the final state
            :math:`\\rho_{f}` is Hermitian. If ``False``, it assumes it is Hermitian.
            Defaults to ``False``.
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

    if not isinstance(check_hermitian, bool):
        raise_error(
            TypeError,
            f"check_hermitian must be type bool, but it is type {type(check_hermitian)}.",
        )

    backend = _check_backend(backend)

    if state is None:
        state = backend.plus_density_matrix(nqubits)

    # necessary because this function do support repeated execution,
    # so it has to default to density matrices
    if len(state.shape) == 1:
        state = np.outer(state, np.conj(state))

    state_final = backend.apply_channel_density_matrix(channel, state, nqubits)

    entang_fidelity = fidelity(
        state_final, state, check_hermitian=check_hermitian, backend=backend
    )

    return entang_fidelity


def meyer_wallach_entanglement(state, backend=None):
    """Compute the Meyer-Wallach entanglement :math:`Q` of a ``state``,

    .. math::
        Q(\\rho) = 2\\left(1 - \\frac{1}{N} \\, \\sum_{k} \\,
            \\text{tr}\\left(\\rho_{k}^{2}\\right)\\right) \\, ,

    where :math:`\\rho_{k}^{2}` is the reduced density matrix of qubit :math:`k`,
    and :math:`N` is the total number of qubits in ``state``.
    We use the definition of the Meyer-Wallach entanglement as the average purity
    proposed in `Brennen (2003) <https://dl.acm.org/doi/10.5555/2011556.2011561>`_,
    which is equivalent to the definition introduced in `Meyer and Wallach (2002)
    <https://doi.org/10.1063/1.1497700>`_.

    Args:
        state (ndarray): statevector or density matrix.
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

    nqubits = int(np.log2(state.shape[-1]))

    entanglement = 0
    for j in range(nqubits):
        trace_q = list(range(nqubits))
        trace_q.pop(j)

        rho_r = partial_trace(state, trace_q, backend=backend)

        trace = purity(rho_r, backend=backend)

        entanglement += trace

    return 2 * (1 - entanglement / nqubits)


def entangling_capability(circuit, samples: int, seed=None, backend=None):
    """Return the entangling capability :math:`\\text{Ent}` of a parametrized circuit.

    It is defined as the average Meyer-Wallach entanglement :math:`Q`
    (:func:`qibo.quantum_info.meyer_wallach_entanglement`) of the ``circuit``, i.e.

    .. math::
        \\text{Ent} = \\frac{2}{|\\mathcal{S}|}\\sum_{\\theta_{k} \\in \\mathcal{S}}
            \\, Q(\\rho_{k}) \\, ,

    where :math:`\\mathcal{S}` is the set of sampled circuit parameters,
    and :math:`\\rho_{k}` is the state prepared by the circuit with uniformily-sampled
    parameters :math:`\\theta_{k}`.

    .. note::
        Currently, function does not work with ``circuit`` that contains noisy channels.

    Args:
        circuit (:class:`qibo.models.Circuit`): Parametrized circuit.
        samples (int): number of sampled circuit parameter vectors :math:`|S|`
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of random
            numbers or a fixed seed to initialize a generator. If ``None``, initializes
            a generator with a random seed. Default: ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Entangling capability :math:`\\text{Ent}`.
    """

    if not isinstance(samples, int):
        raise_error(
            TypeError, f"samples must be type int, but it is type {type(samples)}."
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
        np.random.default_rng(seed) if seed is None or isinstance(seed, int) else seed
    )

    capability = []
    for _ in range(samples):
        params = local_state.uniform(-np.pi, np.pi, circuit.trainable_gates.nparams)
        circuit.set_parameters(params)
        state = backend.execute_circuit(circuit).state()
        entanglement = meyer_wallach_entanglement(state, backend=backend)
        capability.append(entanglement)

    return 2 * np.real(np.sum(capability)) / samples
