"""Submodules with entanglement measures."""

from typing import List, Tuple, Union

import numpy as np

from qibo.backends import _check_backend
from qibo.config import PRECISION_TOL, raise_error
from qibo.quantum_info.linalg_operations import (
    matrix_power,
    partial_trace,
    partial_transpose,
)
from qibo.quantum_info.metrics import fidelity, purity


def concurrence(
    state,
    partition: Union[List[int], Tuple[int, ...]],
    check_purity: bool = True,
    backend=None,
):
    """Calculate concurrence of a pure bipartite quantum state.

    For a pure bipartite quantum state
    :math:`\\rho \\in \\mathcal{H}_{A} \\otimes \\mathcal{H}_{B}`,
    the concurrence :math:`C(\\rho)` can be calculate as

    .. math::
        \\operatorname{C}(\\rho) = \\sqrt{2 \\, (\\operatorname{Tr}^{2}(\\rho) -
            \\operatorname{Tr}(\\rho_{B}^{2}))} \\, ,

    where :math:`\\rho_{B} = \\operatorname{Tr}_{A}(\\rho)` is the reduced density operator
    obtained by tracing out the qubits in the ``partition`` :math:`A`.

    Args:
        state (ndarray): statevector or density matrix.
        partition (list or tuple): qubits in the partition :math:`A` to be traced out.
        check_purity (bool, optional): if ``True``, checks if ``state`` is pure. If ``False``,
            it assumes ``state`` is pure . Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Concurrence :math:`\\operatorname{C}`.
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

    reduced_density_matrix = partial_trace(state, partition, backend=backend)

    purity_reduced = purity(reduced_density_matrix, backend=backend)
    if purity_reduced - 1.0 > 0.0:
        purity_reduced = round(purity_reduced, 7)

    concur = np.sqrt(2 * (1 - purity_reduced))

    return concur


def entanglement_of_formation(
    state,
    partition: Union[List[int], Tuple[int, ...]],
    base: float = 2,
    check_purity: bool = True,
    backend=None,
):
    """Calculate the entanglement of formation of a pure bipartite quantum state.


    For a pure bipartite quantumm state
    :math:`\\rho \\in \\mathcal{H}_{A} \\otimes \\mathcal{H}_{B}`,
    the entanglement of formation :math:`E_{f}` can be calculated as
    function of its :func:`qibo.quantum_info.concurrence`.
    Given a random variable :math:`\\chi \\in \\{0, \\, 1\\}` with
    a Bernoulli probability distribution :math:`[1 - x(\\rho), \\, x(\\rho)]`,
    where

    .. math::
        x(\\rho) = \\frac{1 + \\sqrt{1 - \\operatorname{C}^{2}(\\rho)}}{2} \\, ,

    then the entanglement of formation :math:`\\operatorname{E}_{f}` of state :math:`\\rho`
    is given by

    .. math::
        \\operatorname{E}_{f} = \\operatorname{H}_{2}(\\chi) \\, .

    :math:`\\operatorname{C}(\\rho)` is the :func:`qibo.quantum_info.concurrence` of :math:`\\rho`,
    and :math:`\\operatorname{H}_{2}` is the base-:math:`2`
    :func:`qibo.quantum_info.shannon_entropy`.

    Args:
        state (ndarray): statevector or density matrix :math:`\\rho`.
        partition (list or tuple): qubits in the partition :math:`B` to be traced out.
        base (float): the base of the :math:`\\log` in :func:`qibo.quantum_info.shannon_entropy`.
            Defaults to  :math:`2`.
        check_purity (bool, optional): if ``True``, checks if ``state`` is pure. If ``False``,
            it assumes ``state`` is pure . Default: ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.


    Returns:
        float: Entanglement of formation :math:`\\operatorname{E}_{f}`.
    """
    from qibo.quantum_info.entropies import shannon_entropy  # pylint: disable=C0415

    backend = _check_backend(backend)

    concur = concurrence(
        state, partition=partition, check_purity=check_purity, backend=backend
    )
    concur = (1 + np.sqrt(1 - concur**2)) / 2
    probabilities = [1 - concur, concur]

    ent_of_form = shannon_entropy(probabilities, base=base, backend=backend)

    return ent_of_form


def negativity(state, partition: Union[List[int], Tuple[int, ...]], backend=None):
    """Calculate the negativity of a bipartite quantum state.

    Given a bipartite state :math:`\\rho \\in \\mathcal{H}_{A} \\otimes \\mathcal{H}_{B}`,
    the negativity :math:`\\operatorname{Neg}(\\rho)` is given by

    .. math::
        \\operatorname{Neg}(\\rho) = \\frac{\\|\\rho_{B}\\|_{1} - 1}{2} \\, ,

    where :math:`\\rho_{B}` is the reduced density matrix after tracing out qubits in
    partition :math:`A`, and :math:`\\|\\cdot\\|_{1}` is the Schatten :math:`1`-norm
    (also known as nuclear norm or trace norm).

    Args:
        state (ndarray): statevector or density matrix :math:``.
        partition (list or tuple): qubits in the partition :math:`A` to be traced out.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses it uses the current backend.
            Defaults to ``None``.

    Returns:
        float: Negativity :math:`\\operatorname{Neg}`.
    """
    backend = _check_backend(backend)

    reduced = partial_transpose(state, partition, backend)
    reduced = backend.np.conj(reduced.T) @ reduced
    norm = backend.np.trace(matrix_power(reduced, 1 / 2, backend=backend))

    return backend.np.real((norm - 1) / 2)


def entanglement_fidelity(
    channel, nqubits: int, state=None, check_hermitian: bool = False, backend=None
):
    """Calculate entanglement fidelity of a quantum channel w.r.t. a quantum state.

    Given a quantum ``channel`` :math:`\\mathcal{E}` and a quantum ``state``
    :math:`\\rho`, the entanglement fidelity :math:`F_{\\mathcal{E}}` is given by

    .. math::
        \\operatorname{F}_{\\mathcal{E}}(\\rho) =
            \\operatorname{F}(\\mathcal{E}(\\rho), \\rho) \\, ,

    where :math:`\\operatorname{F}` is the state :func:`qibo.quantum_info.fidelity`,
    and :math:`\\mathcal{E}(\\rho) \\equiv (\\mathcal{E}_{A} \\otimes I_{B})(\\rho)`
    is the state after the channel :math:`\\mathcal{E}` was applied to partition :math:`A`.

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
        \\operatorname{Q}(\\rho) = 2\\left(1 - \\frac{1}{n} \\, \\sum_{k=0}^{n-1} \\,
            \\text{Tr}\\left(\\rho_{k}^{2}\\right)\\right) \\, ,

    where :math:`\\rho_{k}` is the reduced density matrix of the :math:`k`-th qubit,
    and :math:`n` is the total number of qubits in ``state``.

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
    """Calculate the entangling capability :math:`\\text{Ent}` of a parametrized circuit.

    It is defined as the average Meyer-Wallach entanglement :math:`\\operatorname{Q}`
    (:func:`qibo.quantum_info.meyer_wallach_entanglement`) of the ``circuit``, *i.e.*

    .. math::
        \\operatorname{Ent}(\\rho) = \\frac{2}{|\\mathcal{S}|} \\,
            \\sum_{\\theta_{k} \\in \\mathcal{S}} \\,
            \\operatorname{Q}(\\rho(\\theta_{k})) \\, ,

    where :math:`\\mathcal{S}` is the set of sampled circuit parameters,
    :math:`|\\mathcal{S}|` its cardinality, and :math:`\\rho_{k}` is the
    state prepared by the circuit with uniformily-sampled parameters :math:`\\theta_{k}`.

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
