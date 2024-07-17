"""Submodules with entanglement measures."""

import numpy as np

from qibo.backends import _check_backend
from qibo.config import PRECISION_TOL, raise_error
from qibo.quantum_info.linalg_operations import partial_trace
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
            in the execution. If ``None``, it uses :class:`qibo.backends.GlobalBackend`.
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
            in the execution. If ``None``, it uses :class:`qibo.backends.GlobalBackend`.
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
            in the execution. If ``None``, it uses :class:`qibo.backends.GlobalBackend`.
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


def meyer_wallach_entanglement(circuit, backend=None):
    """Computes the Meyer-Wallach entanglement Q of the `circuit`,

    .. math::
        Q(\\theta) = 1 - \\frac{1}{N} \\, \\sum_{k} \\,
            \\text{tr}\\left(\\rho_{k^{2}}(\\theta)\\right) \\, .

    Args:
        circuit (:class:`qibo.models.Circuit`): Parametrized circuit.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses :class:`qibo.backends.GlobalBackend`.
            Defaults to ``None``.

    Returns:
        float: Meyer-Wallach entanglement.
    """

    backend = _check_backend(backend)

    circuit.density_matrix = True
    nqubits = circuit.nqubits

    rho = backend.execute_circuit(circuit).state()

    ent = 0
    for j in range(nqubits):
        trace_q = list(range(nqubits))
        trace_q.pop(j)

        rho_r = partial_trace(rho, trace_q, backend=backend)

        trace = purity(rho_r, backend=backend)

        ent += trace

    entanglement = 1 - ent / nqubits

    return entanglement


def entangling_capability(circuit, samples: int, seed=None, backend=None):
    """Returns the entangling capability :math:`\\text{Ent}` of a parametrized
    circuit, which is average Meyer-Wallach entanglement Q of the circuit, i.e.

    .. math::
        \\text{Ent} = \\frac{2}{S}\\sum_{k}Q_k \\, ,

    where :math:`S` is the number of samples.

    Args:
        circuit (:class:`qibo.models.Circuit`): Parametrized circuit.
        samples (int): number of samples to estimate the integral.
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of random
            numbers or a fixed seed to initialize a generator. If ``None``, initializes
            a generator with a random seed. Default: ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses :class:`qibo.backends.GlobalBackend`.
            Defaults to ``None``.

    Returns:
        float: Entangling capability.
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

    res = []
    for _ in range(samples):
        params = local_state.uniform(-np.pi, np.pi, circuit.trainable_gates.nparams)
        circuit.set_parameters(params)
        entanglement = meyer_wallach_entanglement(circuit, backend=backend)
        res.append(entanglement)

    capability = 2 * np.real(np.sum(res)) / samples

    return capability
