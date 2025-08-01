"""Module with functions that encode classical data into quantum circuits."""

import math
from inspect import signature
from re import finditer
from typing import List, Optional, Union

import numpy as np
from scipy.special import binom

from qibo import gates
from qibo.backends import _check_backend
from qibo.config import raise_error
from qibo.models.circuit import Circuit


def comp_basis_encoder(
    basis_element: Union[int, str, list, tuple], nqubits: Optional[int] = None, **kwargs
):
    """Create circuit that performs encoding of bitstrings into computational basis states.

    Args:
        basis_element (int or str or list or tuple): bitstring to be encoded.
            If ``int``, ``nqubits`` must be specified.
            If ``str``, must be composed of only :math:`0`s and :math:`1`s.
            If ``list`` or ``tuple``, must be composed of :math:`0`s and
            :math:`1`s as ``int`` or ``str``.
        nqubits (int, optional): total number of qubits in the circuit.
            If ``basis_element`` is ``int``, ``nqubits`` must be specified.
            If ``nqubits`` is ``None``, ``nqubits`` defaults to length of ``basis_element``.
            Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
       :class:`qibo.models.circuit.Circuit`: Circuit encoding computational basis element.
    """
    if not isinstance(basis_element, (int, str, list, tuple)):
        raise_error(
            TypeError,
            "basis_element must be either type int or str or list or tuple, "
            + f"but it is type {type(basis_element)}.",
        )

    if isinstance(basis_element, (str, list, tuple)):
        if any(elem not in ["0", "1", 0, 1] for elem in basis_element):
            raise_error(ValueError, "all elements must be 0 or 1.")

    if nqubits is not None and not isinstance(nqubits, int):
        raise_error(
            TypeError, f"``nqubits`` must be type int, but it is type {type(nqubits)}."
        )

    if nqubits is None:
        if isinstance(basis_element, int):
            raise_error(
                ValueError,
                "``nqubits`` must be specified when ``basis_element`` is type int.",
            )
        else:
            nqubits = len(basis_element)

    if isinstance(basis_element, int):
        basis_element = f"{basis_element:0{nqubits}b}"

    if isinstance(basis_element, (str, tuple)):
        basis_element = list(basis_element)

    basis_element = list(map(int, basis_element))

    circuit = Circuit(nqubits, **kwargs)
    for qubit, elem in enumerate(basis_element):
        if elem == 1:
            circuit.add(gates.X(qubit))

    return circuit


def phase_encoder(data, rotation: str = "RY", backend=None, **kwargs):
    """Create circuit that performs the phase encoding of ``data``.

    Args:
        data (ndarray or list): :math:`1`-dimensional array of phases to be loaded.
        rotation (str, optional): If ``"RX"``, uses :class:`qibo.gates.gates.RX` as rotation.
            If ``"RY"``, uses :class:`qibo.gates.gates.RY` as rotation.
            If ``"RZ"``, uses :class:`qibo.gates.gates.RZ` as rotation.
            Defaults to ``"RY"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads ``data`` in phase encoding.
    """
    if not isinstance(rotation, str):
        raise_error(
            TypeError,
            f"``rotation`` must be type str, but it is type {type(rotation)}.",
        )

    backend = _check_backend(backend)

    if isinstance(data, list):
        # TODO: Fix this mess with qibo native dtypes
        try:
            type_test = data[0].dtype
        except AttributeError:  # pragma: no cover
            type_test = type(data[0])

        data = backend.cast(data, dtype=type_test)

    if rotation not in ["RX", "RY", "RZ"]:
        raise_error(ValueError, f"``rotation`` {rotation} not found.")

    nqubits = len(data)
    gate = getattr(gates, rotation.upper())

    circuit = Circuit(nqubits, **kwargs)
    circuit.add(gate(qubit, 0.0) for qubit in range(nqubits))
    circuit.set_parameters(data)

    return circuit


def sparse_encoder(
    data, method: str = "li", nqubits: int = None, backend=None, **kwargs
):
    """Create circuit that encodes :math:`1`-dimensional data in a subset of amplitudes
    of the computational basis.

    Consider a sparse-access model, where for a data vector
    :math:`\\mathbf{x} \\in \\mathbb{C}^{d}`, with :math:`d = 2^{n}` and
    :math:`s` non-zero amplitudes, one has access to the data vector
    :math:`\\mathbf{y}` of the form

    .. math::
        \\mathbf{y} = \\left\\{ (b_{1}, x_{1}), \\, \\dots, \\, (b_{s}, x_{s}) \\right\\} \\, ,


    where :math:`\\{x_{j}\\}_{j\\in[s]}` is the non-zero components of :math:`\\mathbf{x}`
    and :math:`\\{b_{j}\\}_{j\\in[s]}` is the set of addresses associated with these values.
    Then, this function generates a quantum circuit  :math:`s\\text{-}\\mathrm{Load}` that encodes
    :math:`\\mathbf{x}` in the amplitudes of an :math:`n`-qubit quantum state as

    .. math::
        s\\text{-}\\mathrm{Load}(\\mathbf{y}) \\, \\ket{0}^{\\otimes \\, n} = \\sum_{j\\in[s]} \\,
            \\frac{x_{j}}{\\|\\mathbf{x}\\|_{2}} \\, \\ket{b_{j}} \\, ,

    where :math:`\\|\\cdot\\|_{2}` is the Euclidean norm.

    The resulting circuit parametrizes ``data`` in hyperspherical coordinates
    in the :math:`(2^{n} - 1)`-unit sphere.


    Args:
        data (ndarray or list or zip): sequence of tuples of the form :math:`(b_{j}, x_{j})`.
            The addresses :math:`b_{j}` can be either integers or in bitstring
            format of size :math:`n`.
        method (str, optional): method to be used, either ``li`` or ``farias``. They refer to
            methods in references [1] and [2], respectively.
            Defaults to ``li``
        nqubits (int, optional): total number of qubits in the system.
            To be used when :math:`b_j` are integers. If :math:`b_j` are strings and
            ``nqubits`` is ``None``, defaults to the length of the strings :math:`b_{j}`.
            Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads sparse :math:`\\mathbf{x}`.

    References:
        1. L. Li, and J. Luo,
        *Nearly Optimal Circuit Size for Sparse Quantum State Preparation*
        `arXiv:2406.16142 (2024) <https://doi.org/10.48550/arXiv.2406.16142>`_.
        2. R. M. S. Farias, T. O. Maciel, G. Camilo, R. Lin, S. Ramos-Calderer, and L. Aolita,
        *Quantum encoder for fixed-Hamming-weight subspaces*,
        `Phys. Rev. Applied 23, 044014 (2025) <https://doi.org/10.1103/PhysRevApplied.23.044014>`_.

        3. `Hyperpherical coordinates <https://en.wikipedia.org/wiki/N-sphere>`_.
    """
    backend = _check_backend(backend)

    if isinstance(data, zip):
        data = list(data)

    # TODO: Fix this mess with qibo native dtypes
    try:
        type_test = bool("int" in str(data[0][0].dtype))
    except AttributeError:
        type_test = bool("int" in str(type(data[0][0])))

    if type_test and nqubits is None:
        raise_error(
            ValueError,
            "``nqubits`` must be specified when computational basis states are "
            + "indidated by integers.",
        )

    if isinstance(data[0][0], str) and nqubits is None:
        nqubits = len(data[0][0])

    if method not in ("li", "farias"):
        raise_error(
            ValueError,
            f"``method`` should be either ``li`` or ``farias``, but it is {method}",
        )

    func = _sparse_encoder_farias if method == "farias" else _sparse_encoder_li

    return func(data, nqubits, backend, **kwargs)


def _sparse_encoder_li(data, nqubits: int, backend=None, **kwargs):
    """Create circuit that encodes :math:`1`-dimensional data in a subset of amplitudes
    of the computational basis.

    Consider a sparse-access model, where for a data vector
    :math:`\\mathbf{x} \\in \\mathbb{C}^{d}`, with :math:`d = 2^{n}` and
    :math:`s` non-zero amplitudes, one has access to the data vector
    :math:`\\mathbf{y}` of the form

    .. math::
        \\mathbf{y} = \\left\\{ (b_{1}, x_{1}), \\, \\dots, \\, (b_{s}, x_{s}) \\right\\} \\, ,


    where :math:`\\{x_{j}\\}_{j\\in[s]}` are the non-zero components of :math:`\\mathbf{x}`
    and :math:`\\{b_{j}\\}_{j\\in[s]}` is the set of addresses associated with these values.
    Then, this function generates a quantum circuit  :math:`s\\text{-}\\mathrm{Load}` that encodes
    :math:`\\mathbf{x}` in the amplitudes of an :math:`n`-qubit quantum state as

    .. math::
        s\\text{-}\\mathrm{Load}(\\mathbf{y}) \\, \\ket{0}^{\\otimes \\, n} = \\sum_{j\\in[s]} \\,
            \\frac{x_{j}}{\\|\\mathbf{x}\\|_{2}} \\, \\ket{b_{j}} \\, ,

    where :math:`\\|\\cdot\\|_{2}` is the l2-norm.

    The resulting circuit parametrizes ``data`` in hyperspherical coordinates
    in the :math:`(2^{n} - 1)`-unit sphere.


    Args:
        data (ndarray or list or zip): sequence of tuples of the form :math:`(b_{j}, x_{j})`.
            The addresses :math:`b_{j}` can be either integers or in bitstring
            format of size :math:`n`.
        nqubits (int, optional): total number of qubits in the system.
            To be used when :math:`b_j` are integers. If :math:`b_j` are strings and
            ``nqubits`` is ``None``, defaults to the length of the strings :math:`b_{j}`.
            Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads sparse :math:`\\mathbf{x}`.

    References:
        1. L. Li, and J. Luo,
        *Nearly Optimal Circuit Size for Sparse Quantum State Preparation*
        `arXiv:2406.16142 (2024) <https://doi.org/10.48550/arXiv.2406.16142>`_.

        2. `Hyperpherical coordinates <https://en.wikipedia.org/wiki/N-sphere>`_.
    """
    backend = _check_backend(backend)
    data_sorted, bitstrings_sorted = _sort_data_sparse(data, nqubits, backend)
    bitstrings_sorted = backend.cast(
        [int("".join(map(str, string)), 2) for string in bitstrings_sorted],
        dtype=backend.np.int8,
    )

    dim = len(data_sorted)
    sigma = np.arange(2**nqubits)

    flag = backend.np.zeros(dim, dtype=backend.np.int8)
    indexes = list(
        backend.to_numpy(bitstrings_sorted[bitstrings_sorted < dim]).astype(int)
    )
    flag[indexes] = 1

    data_binary = []
    for bi_int, xi in zip(bitstrings_sorted, data_sorted):
        bi_int = int(bi_int)
        if bi_int >= dim:
            for k in range(dim):
                if flag[k] == 0:
                    flag[k] = 1
                    sigma[[bi_int, k]] = [k, bi_int]
                    data_binary.append((k, xi))
                    break
        else:
            data_binary.append((bi_int, xi))

    sigma = list(sigma)

    # binary enconder on \sum_i = xi |sigma^{-1}(b_i)>
    circuit_binary = sparse_encoder(
        data_binary, method="farias", nqubits=nqubits, backend=backend, **kwargs
    )
    circuit_permutation = permutation_synthesis(sigma, **kwargs)

    return circuit_binary + circuit_permutation


def _sparse_encoder_farias(data, nqubits: int, backend=None, **kwargs):
    """Create circuit that encodes :math:`1`-dimensional data in a subset of amplitudes
    of the computational basis.

    Consider a sparse-access model, where for a data vector
    :math:`\\mathbf{x} \\in \\mathbb{C}^{d}`, with :math:`d = 2^{n}` and
    :math:`s` non-zero amplitudes, one has access to the data vector
    :math:`\\mathbf{y}` of the form

    .. math::
        \\mathbf{y} = \\left\\{ (b_{1}, x_{1}), \\, \\dots, \\, (b_{s}, x_{s}) \\right\\} \\, ,


    where :math:`\\{x_{j}\\}_{j\\in[s]}` are the non-zero components of :math:`\\mathbf{x}`
    and :math:`\\{b_{j}\\}_{j\\in[s]}` is the set of addresses associated with these values.
    Then, this function generates a quantum circuit  :math:`s\\text{-}\\mathrm{Load}` that encodes
    :math:`\\mathbf{x}` in the amplitudes of an :math:`n`-qubit quantum state as

    .. math::
        s\\text{-}\\mathrm{Load}(\\mathbf{y}) \\, \\ket{0}^{\\otimes \\, n} = \\sum_{j\\in[s]} \\,
            \\frac{x_{j}}{\\|\\mathbf{x}\\|_{2}} \\, \\ket{b_{j}} \\, ,

    where :math:`\\|\\cdot\\|_{2}` is the l2-norm.

    Resulting circuit parametrizes ``data`` in hyperspherical coordinates
    in the :math:`(2^{n} - 1)`-unit sphere.


    Args:
        data (ndarray or list or zip): sequence of tuples of the form :math:`(b_{j}, x_{j})`.
            The addresses :math:`b_{j}` can be either integers or in bitstring
            format of size :math:`n`.
        nqubits (int, optional): total number of qubits in the system.
            To be used when :math:`b_j` are integers. If :math:`b_j` are strings and
            ``nqubits`` is ``None``, defaults to the length of the strings :math:`b_{j}`.
            Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads sparse :math:`\\mathbf{x}`.

    References:
        1. R. M. S. Farias, T. O. Maciel, G. Camilo, R. Lin, S. Ramos-Calderer, and L. Aolita,
        *Quantum encoder for fixed-Hamming-weight subspaces*,
        `Phys. Rev. Applied 23, 044014 (2025) <https://doi.org/10.1103/PhysRevApplied.23.044014>`_.

        2. `Hyperpherical coordinates <https://en.wikipedia.org/wiki/N-sphere>`_.
    """
    from qibo.quantum_info.utils import (  # pylint: disable=import-outside-toplevel
        hamming_distance,
        hamming_weight,
    )

    backend = _check_backend(backend)

    if isinstance(data, zip):
        data = list(data)

    # TODO: Fix this mess with qibo native dtypes
    try:
        type_test = bool("int" in str(data[0][0].dtype))
    except AttributeError:
        type_test = bool("int" in str(type(data[0][0])))

    if type_test and nqubits is None:
        raise_error(
            ValueError,
            "``nqubits`` must be specified when computational basis states are "
            + "indidated by integers.",
        )

    if isinstance(data[0][0], str) and nqubits is None:
        nqubits = len(data[0][0])

    _data_test = data[0][1]
    _data_test = (
        _data_test.dtype if "array" in str(type(_data_test)) else type(_data_test)
    )

    complex_data = bool("complex" in str(_data_test))

    # sort data by HW of the bitstrings
    data_sorted, bitstrings_sorted = _sort_data_sparse(data, nqubits, backend)
    # calculate phases
    _data_sorted = backend.np.abs(data_sorted) if complex_data else data_sorted
    thetas = _generate_rbs_angles(
        _data_sorted, architecture="diagonal", backend=backend
    )
    phis = backend.np.zeros(len(thetas) + 1)
    if complex_data:
        phis[0] = _angle_mod_two_pi(-backend.np.angle(data_sorted[0]))
        for k in range(1, len(phis)):
            phis[k] = _angle_mod_two_pi(
                -backend.np.angle(data_sorted[k]) + backend.np.sum(phis[:k])
            )
    phis = backend.cast(phis, dtype=phis[0].dtype)

    # marking qubits that have suffered the action of a gate
    initial_string = [int(bit) for bit in bitstrings_sorted[0]]
    circuit = comp_basis_encoder(initial_string, nqubits=nqubits, **kwargs)
    touched_qubits = list(np.nonzero(initial_string)[0])

    for b_1, b_0, theta, phi in zip(
        bitstrings_sorted[1:], bitstrings_sorted[:-1], thetas, phis
    ):
        hw_0, hw_1 = hamming_weight(b_0), hamming_weight(b_1)
        distance = hamming_distance(b_1, b_0)
        difference = b_1 - b_0

        ones, new_ones = (
            list(backend.np.argsort(b_0)[-hw_0:]),
            list(backend.np.argsort(b_1)[-hw_1:]),
        )
        ones, new_ones = {int(elem) for elem in ones}, {int(elem) for elem in new_ones}
        controls = (set(ones) & set(new_ones)) & set(touched_qubits)

        gate = _get_gate_sparse(
            distance,
            difference,
            touched_qubits,
            complex_data,
            controls,
            hw_0,
            hw_1,
            theta,
            phi,
        )
        circuit.add(gate)

    if complex_data:
        hw_0 = hamming_weight(bitstrings_sorted[-2])
        hw_1 = hamming_weight(bitstrings_sorted[-1])
        correction = _get_phase_gate_correction_sparse(
            bitstrings_sorted[-1],
            bitstrings_sorted[-2],
            nqubits,
            data_sorted[-1],
            data_sorted[-2],
            circuit,
            phis,
        )
        if hw_1 == nqubits and hw_0 == nqubits - 1:
            circuit.queue = correction
        else:
            circuit.add(correction)

    return circuit


def binary_encoder(
    data, parametrization: str = "hyperspherical", backend=None, **kwargs
):
    """Create circuit that encodes :math:`1`-dimensional data in all amplitudes
    of the computational basis.

    Given data vector :math:`\\mathbf{x} \\in \\mathbb{C}^{d}`, with :math:`d = 2^{n}`,
    this function generates a quantum circuit :math:`\\mathrm{Load}` that encodes
    :math:`\\mathbf{x}` in the amplitudes of an :math:`n`-qubit quantum state as

    .. math::
        \\mathrm{Load}(\\mathbf{x}) \\, \\ket{0}^{\\otimes \\, n} = \\sum_{j=0}^{d-1} \\,
            \\frac{x_{j}}{\\|\\mathbf{x}\\|_{F}} \\, \\ket{b_{j}} \\, ,

    where :math:`b_{j} \\in \\{0, \\, 1\\}^{\\otimes \\, n}` is the :math:`n`-bit representation
    of the integer :math:`j`, :math:`\\|\\cdot\\|_{F}` is the Frobenius norm.

    Resulting circuit parametrizes ``data`` in either ``hyperspherical`` or ``Hopf`` coordinates
    in the :math:`(2^{n} - 1)`-unit sphere.

    Args:
        data (ndarray): :math:`1`-dimensional array or length :math:`d = 2^{n}`
            to be loaded in the amplitudes of a :math:`n`-qubit quantum state.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads ``data`` in binary encoding.

    References:
        1. R. M. S. Farias, T. O. Maciel, G. Camilo, R. Lin, S. Ramos-Calderer, and L. Aolita,
        *Quantum encoder for fixed-Hamming-weight subspaces*
        `Phys. Rev. Applied 23, 044014 (2025) <https://doi.org/10.1103/PhysRevApplied.23.044014>`_.

        2. `Hyperpherical coordinates <https://en.wikipedia.org/wiki/N-sphere>`_.

        3. H. S. Cohl, *Fourier, Gegenbauer and Jacobi expansions for a power-law fundamental
        solution of the polyharmonic equation and polyspherical addition theorems*, `Symmetry,
        Integrability and Geometry: Methods and Applications 10.3842/sigma.2013.042 (2013)
        <https://arxiv.org/abs/1209.6047>`_.
    """
    backend = _check_backend(backend)

    dims = len(data)
    nqubits = float(np.log2(dims))
    if not nqubits.is_integer():
        raise_error(ValueError, "`data` size must be a power of 2.")
    nqubits = int(nqubits)

    complex_data = bool(
        "complex" in str(data.dtype)
    )  # backend-agnostic way of checking the dtype

    if parametrization == "hopf":
        return _binary_encoder_hopf(
            data, nqubits, complex_data=complex_data, backend=backend, **kwargs
        )

    return _binary_encoder_hyperspherical(
        data, nqubits, complex_data=complex_data, backend=backend, **kwargs
    )


def unary_encoder(data, architecture: str = "tree", backend=None, **kwargs):
    """Create circuit that performs the (deterministic) unary encoding of ``data``.

    Args:
        data (ndarray): :math:`1`-dimensional array of data to be loaded.
        architecture(str, optional): circuit architecture used for the unary loader.
            If ``diagonal``, uses a ladder-like structure.
            If ``tree``, uses a binary-tree-based structure.
            Defaults to ``tree``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads ``data`` in unary representation.
    """
    backend = _check_backend(backend)

    if isinstance(data, list):
        data = backend.cast(data, dtype=type(data[0]))

    if not isinstance(architecture, str):
        raise_error(
            TypeError,
            f"``architecture`` must be type str, but it is type {type(architecture)}.",
        )

    if architecture not in ["diagonal", "tree"]:
        raise_error(ValueError, f"``architecture`` {architecture} not found.")

    if architecture == "tree" and not math.log2(data.shape[0]).is_integer():
        raise_error(
            ValueError,
            "When ``architecture = 'tree'``, len(data) must be a power of 2. "
            + f"However, it is {len(data)}.",
        )

    nqubits = len(data)

    circuit = Circuit(nqubits, **kwargs)
    circuit.add(gates.X(nqubits - 1))
    circuit_rbs, _ = _generate_rbs_pairs(nqubits, architecture=architecture, **kwargs)
    circuit += circuit_rbs

    # calculating phases and setting circuit parameters
    phases = _generate_rbs_angles(data, architecture, nqubits, backend=backend)
    circuit.set_parameters(phases)

    return circuit


def unary_encoder_random_gaussian(
    nqubits: int, architecture: str = "tree", seed=None, backend=None, **kwargs
):
    """Create a circuit that performs the unary encoding of a random Gaussian state.

    At depth :math:`h` of the tree architecture, the angles :math:`\\theta_{k}
    \\in [0, 2\\pi]` of the the gates :math:`RBS(\\theta_{k})` are sampled from
    the following probability density function:

    .. math::
        p_{h}(\\theta) = \\frac{1}{2} \\, \\frac{\\Gamma(2^{h-1})}{\\Gamma^{2}(2^{h-2})} \\,
            \\left|\\sin(\\theta) \\, \\cos(\\theta)\\right|^{2^{h-1} - 1} \\, ,

    where :math:`\\Gamma(\\cdot)` is the
    `Gamma function <https://en.wikipedia.org/wiki/Gamma_function>`_.

    Args:
        nqubits (int): number of qubits.
        architecture(str, optional): circuit architecture used for the unary loader.
            If ``tree``, uses a binary-tree-based structure.
            Defaults to ``tree``.
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of
            random numbers or a fixed seed to initialize a generator. If ``None``,
            initializes a generator with a random seed. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads a random Gaussian
        array in unary representation.

    References:
        1. A. Bouland, A. Dandapani, and A. Prakash, *A quantum spectral method for simulating
        stochastic processes, with applications to Monte Carlo*.
        `arXiv:2303.06719v1 [quant-ph] <https://arxiv.org/abs/2303.06719>`_
    """
    backend = _check_backend(backend)

    if not isinstance(nqubits, int):
        raise_error(
            TypeError, f"nqubits must be type int, but it is type {type(nqubits)}."
        )

    if nqubits <= 0.0:
        raise_error(
            ValueError, f"nqubits must be a positive integer, but it is {nqubits}."
        )

    if not isinstance(architecture, str):
        raise_error(
            TypeError,
            f"``architecture`` must be type str, but it is type {type(architecture)}.",
        )

    if architecture != "tree":
        raise_error(
            NotImplementedError,
            "Currently, this function only accepts ``architecture=='tree'``.",
        )

    if not math.log2(nqubits).is_integer():
        raise_error(ValueError, f"nqubits must be a power of 2, but it is {nqubits}.")

    if (
        seed is not None
        and not isinstance(seed, int)
        and not isinstance(seed, np.random.Generator)
    ):
        raise_error(
            TypeError, "seed must be either type int or numpy.random.Generator."
        )

    from qibo.quantum_info.random_ensembles import (  # pylint: disable=C0415
        _ProbabilityDistributionGaussianLoader,
    )

    local_state = (
        np.random.default_rng(seed) if seed is None or isinstance(seed, int) else seed
    )

    sampler = _ProbabilityDistributionGaussianLoader(
        a=0, b=2 * math.pi, seed=local_state
    )

    circuit = Circuit(nqubits, **kwargs)
    circuit.add(gates.X(nqubits - 1))
    circuit_rbs, pairs_rbs = _generate_rbs_pairs(nqubits, architecture, **kwargs)
    circuit += circuit_rbs

    phases = []
    for depth, row in enumerate(pairs_rbs, 1):
        phases.extend(sampler.rvs(depth=depth, size=len(row)))

    phases = backend.cast(phases, dtype=type(phases[0]))

    circuit.set_parameters(phases)

    return circuit


def hamming_weight_encoder(
    data,
    nqubits: int,
    weight: int,
    full_hwp: bool = False,
    optimize_controls: bool = True,
    phase_correction: bool = True,
    initial_string=None,
    backend=None,
    **kwargs,
):
    """Create circuit that encodes ``data`` in the Hamming-weight-:math:`k` basis of ``nqubits``.

    Let :math:`\\mathbf{x}` be a :math:`1`-dimensional array of size :math:`d = \\binom{n}{k}` and
    :math:`B_{k} \\equiv \\{ \\ket{b_{j}} : b_{j} \\in \\{0, 1\\}^{\\otimes n} \\,\\, \\text{and}
    \\,\\, |b_{j}| = k \\}` be a set of :math:`d` computational basis states of :math:`n` qubits
    that are represented by bitstrings of Hamming weight :math:`k`. Then, an amplitude encoder
    in the basis :math:`B_{k}` is an :math:`n`-qubit parameterized quantum circuit
    :math:`\\operatorname{Load}_{B_{k}}` such that

    .. math::
        \\operatorname{Load}(\\mathbf{x}) \\, \\ket{0}^{\\otimes n} = \\frac{1}{\\|\\mathbf{x}\\|}
            \\, \\sum_{j = 1}^{d} \\, x_{j} \\, \\ket{b_{j}}

    Args:
        data (ndarray): :math:`1`-dimensional array of data to be loaded.
        nqubits (int): number of qubits.
        weight (int): Hamming weight that defines the subspace in which ``data`` will be encoded.
        full_hwp (bool, optional): if ``False``, includes Pauli-:math:`X` gates that prepare the
            first bitstring of Hamming weight ``k = weight``. If ``True``, circuit is full
            Hamming weight preserving. Defaults to ```False``.
        optimize_controls (bool, optional): if ``True``, removes unnecessary controlled operations.
            Defaults to ``True``.
        phase_correction (bool, optional): To be used when ``data`` is complex-valued.
            If ``True``, adds a controlled-$\\mathrm{RZ}$ gate to the end of the circuit,
            adding a final phase correction. If ``False``, gate is not added. Defaults to ``True``.
        initial_string (ndarray, optional): Array containing the desired initial bitstring of
            Hamming ``weight`` $k$. If ``None``, defaults to $\\ket{1^{k}0^{n-k}}$.
            Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads ``data`` in
        Hamming-weight-:math:`k` representation.

    References:
        1. R. M. S. Farias, T. O. Maciel, G. Camilo, R. Lin, S. Ramos-Calderer, and L. Aolita,
        *Quantum encoder for fixed-Hamming-weight subspaces*
        `Phys. Rev. Applied 23, 044014 (2025) <https://doi.org/10.1103/PhysRevApplied.23.044014>`_.
    """
    backend = _check_backend(backend)

    complex_data = bool("complex" in str(data.dtype))

    if initial_string is None:
        initial_string = np.array([1] * weight + [0] * (nqubits - weight))
    bitstrings, targets_and_controls = _ehrlich_algorithm(initial_string)

    # sort data such that the encoding is performed in lexicographical order
    lex_order = [int(string, 2) for string in bitstrings]
    lex_order_sorted = np.sort(np.copy(lex_order))
    lex_order = [np.where(lex_order_sorted == num)[0][0] for num in lex_order]
    data = data[lex_order]
    del lex_order, lex_order_sorted

    # Calculate all gate phases necessary to encode the amplitudes.
    _data = backend.np.abs(data) if complex_data else data
    thetas = _generate_rbs_angles(_data, architecture="diagonal", backend=backend)
    phis = backend.np.zeros(len(thetas) + 1)
    if complex_data:
        phis[0] = _angle_mod_two_pi(-backend.np.angle(data[0]))
        for k in range(1, len(phis)):
            phis[k] = _angle_mod_two_pi(
                -backend.np.angle(data[k]) + backend.np.sum(phis[:k])
            )

    last_qubit = nqubits - 1

    circuit = Circuit(nqubits, **kwargs)
    if not full_hwp:
        circuit.add(
            gates.X(qubit) for qubit in range(last_qubit, last_qubit - weight, -1)
        )

    if optimize_controls:
        indices = [
            int(binom(nqubits - j, weight - j)) - 1 for j in range(weight - 1, 0, -1)
        ]

    for k, ((targets, controls), theta, phi) in enumerate(
        zip(targets_and_controls, thetas, phis)
    ):
        targets = list(last_qubit - np.asarray(targets))
        controls = list(last_qubit - np.asarray(controls))
        controls.sort()

        if optimize_controls:
            controls = list(np.asarray(controls)[k >= np.asarray(indices)])

        gate = _get_gate(
            [targets[0]],
            [targets[1]],
            controls,
            theta,
            phi,
            complex_data,
        )
        circuit.add(gate)

    if complex_data and phase_correction:
        circuit.add(_get_phase_gate_correction(bitstrings[-1], phis[-1]))

    return circuit


def entangling_layer(
    nqubits: int,
    architecture: str = "diagonal",
    entangling_gate: Union[str, gates.Gate] = "CNOT",
    closed_boundary: bool = False,
    **kwargs,
):
    """Create a layer of two-qubit entangling gates.

    If the chosen gate is a parametrized gate, all phases are set to :math:`0.0`.

    Args:
        nqubits (int): Total number of qubits in the circuit.
        architecture (str, optional): Architecture of the entangling layer.
            In alphabetical order, options are ``"diagonal"``, ``"even_layer"``,
            ``"next_nearest"``, ``"odd_layer"``, ``"pyramid"``, ``"shifted"``,
            ``"v"``, and ``"x"``. The ``"x"`` architecture is only defined for an even number
            of qubits. Defaults to ``"diagonal"``.
        entangling_gate (str or :class:`qibo.gates.Gate`, optional): Two-qubit gate to be used
            in the entangling layer. If ``entangling_gate`` is a parametrized gate,
            all phases are initialized as :math:`0.0`. Defaults to  ``"CNOT"``.
        closed_boundary (bool, optional): If ``True`` and ``architecture not in
            ["pyramid", "v", "x"]``, adds a closed-boundary condition to the entangling layer.
            Defaults to ``False``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit containing layer of two-qubit gates.
    """

    if not isinstance(nqubits, int):
        raise_error(
            TypeError, f"nqubits must be type int, but it is type {type(nqubits)}."
        )

    if nqubits <= 0.0:
        raise_error(
            ValueError, f"nqubits must be a positive integer, but it is {nqubits}."
        )

    if not isinstance(architecture, str):
        raise_error(
            TypeError,
            f"``architecture`` must be type str, but it is type {type(architecture)}.",
        )

    if architecture not in [
        "diagonal",
        "even_layer",
        "next_nearest",
        "odd_layer",
        "pyramid",
        "shifted",
        "v",
        "x",
    ]:
        raise_error(
            NotImplementedError,
            f"``architecture`` {architecture} not found.",
        )

    if architecture == "x" and nqubits % 2 != 0.0:
        raise_error(
            ValueError, "``x`` architecture only defined for an even number of qubits."
        )

    if not isinstance(closed_boundary, bool):
        raise_error(
            TypeError,
            f"``closed_boundary`` must be type bool, but it is type {type(closed_boundary)}.",
        )

    gate = (
        getattr(gates, entangling_gate)
        if isinstance(entangling_gate, str)
        else entangling_gate
    )

    if gate.__name__ == "GeneralizedfSim":
        raise_error(
            NotImplementedError,
            "This function does not support the ``GeneralizedfSim`` gate.",
        )

    if architecture in ["next_nearest", "pyramid", "v", "x"]:
        circuit = _non_trivial_layers(
            nqubits,
            architecture=architecture,
            entangling_gate=entangling_gate,
            closed_boundary=closed_boundary,
            **kwargs,
        )
    else:
        # Finds the correct number of parameters to initialize the gate class.
        parameters = list(signature(gate).parameters)

        if "q2" in parameters:
            raise_error(
                NotImplementedError, "This function does not accept three-qubit gates."
            )

        # If gate is parametrized, sets all angles to 0.0
        parameters = (0.0,) * (len(parameters) - 3) if len(parameters) > 2 else None

        circuit = Circuit(nqubits, **kwargs)

        if architecture == "diagonal":
            qubits = range(nqubits - 1)
        elif architecture == "even_layer":
            qubits = range(0, nqubits - 1, 2)
        elif architecture == "odd_layer":
            qubits = range(1, nqubits - 1, 2)
        else:
            qubits = tuple(range(0, nqubits - 1, 2)) + tuple(range(1, nqubits - 1, 2))

        circuit.add(
            _parametrized_two_qubit_gate(gate, qubit, qubit + 1, parameters)
            for qubit in qubits
        )

        if closed_boundary:
            circuit.add(_parametrized_two_qubit_gate(gate, nqubits - 1, 0, parameters))

    return circuit


def ghz_state(nqubits: int, **kwargs):
    """Generate an :math:`n`-qubit Greenberger-Horne-Zeilinger (GHZ) state that takes the form

    .. math::
        \\ket{\\text{GHZ}} = \\frac{\\ket{0}^{\\otimes n} + \\ket{1}^{\\otimes n}}{\\sqrt{2}}

    where :math:`n` is the number of qubits.

    Args:
        nqubits (int): number of qubits :math:`n >= 2`.
        kwargs (dict, optional): additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that prepares the GHZ state.
    """
    if nqubits < 2:
        raise_error(
            ValueError,
            f"nqubits given as {nqubits}. nqubits needs to be >= 2.",
        )

    circuit = Circuit(nqubits, **kwargs)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(qubit, qubit + 1) for qubit in range(nqubits - 1))

    return circuit


def graph_state(matrix, backend=None, **kwargs):
    """Create circuit encoding an undirected graph state given its adjacency matrix.

    Given a graph :math:`G = (V, E)` with :math:`V` being the set of vertices and :math:`E`
    being the set of edges, an $n$-qubit graph state is defined as

    .. math::
        \\ket{G} = \\prod_{(j, k) \\in E} \\, CZ_{j, k} \\, \\ket{+}^{\\otimes V} \\, ,

    where :math:`CZ_{a,b}` is the :class:`qibo.gates.CZ` gate acting on qubits :math:`j`
    and :math:`k`, and :math:`\\ket{+} = H \\, \\ket{0}$, with :class:`qibo.gates.H`
    being the Hadamard gate.

    Args:
        matrix (ndarray or list): Adjacency matrix of the graph.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit of the graph state with the given
        adjacency matrix.
    """
    backend = _check_backend(backend)

    if isinstance(matrix, list):
        matrix = backend.cast(matrix, dtype=int)

    if not backend.np.allclose(matrix, matrix.T):
        raise_error(
            ValueError,
            f"``matrix`` is not symmetric, not representing an undirected graph",
        )

    nqubits = len(matrix)

    circuit = Circuit(nqubits, **kwargs)
    circuit.add(gates.H(qubit) for qubit in range(nqubits))

    # since the matrix is symmetric, we only need the upper triangular part
    rows, columns = backend.np.nonzero(backend.np.triu(matrix))
    circuit.add(gates.CZ(int(ind_r), int(ind_c)) for ind_r, ind_c in zip(rows, columns))

    return circuit


def dicke_state(nqubits: int, weight: int, all_to_all: bool = False, **kwargs):
    """Create a circuit that prepares the Dicke state :math:`\\ket{D_{k}^{n}}`.

    The Dicke state :math:`\\ket{D_{k}^{n}}` is the equal superposition of all :math:`n`-qubit
    computational basis states with fixed-Hamming-weight :math:`k`.
    The circuit prepares the state deterministically with :math:`O(k \\, n)` gates and :math:`O(n)`
    depth, or :math:`O(k \\log\\frac{n}{k})` depth under the assumption of ``all-to-all``
    connectivity.

    Args:
        nqubits (int): number of qubits :math:`n`.
        weight (int): Hamming weight :math:`k` of the Dicke state.
        all_to_all (bool, optional): If ``False``, uses implementation from Ref. [1].
            If ``True``, uses shorter-depth implementation from Ref. [2].
            Defaults to ``False``.
        kwargs (dict, optional): additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit` : Circuit that prepares :math:`\\ket{D_{k}^{n}}`.

    References:
        1. Andreas Bärtschi and Stephan Eidenbenz, *Deterministic preparation of Dicke states*,
        `22nd International Symposium on Fundamentals of Computation Theory, FCT'19, 126-139  (2019)
        <https://doi.org/10.1007/978-3-030-25027-0_9>`_.

        2. Andreas Bärtschi and Stephan Eidenbenz, *Short-Depth Circuits for Dicke State Preparation*,
        `IEEE International Conference on Quantum Computing & Engineering (QCE), 87--96 (2022)
        <https://doi.org/10.1109/QCE53715.2022.00027>`_.
    """
    if weight < 0 or weight > nqubits:
        raise_error(
            ValueError, f"weight must be between 0 and {nqubits}, but got {weight}."
        )

    circuit = Circuit(nqubits, **kwargs)

    if weight == 0:
        return circuit

    if not all_to_all:
        # Start with |0⟩^(n-k) |1⟩^k
        circuit.add(gates.X(qubit) for qubit in range(nqubits - weight, nqubits))

        _add_dicke_unitary_gate(circuit, range(nqubits), weight)

        return circuit

    # We prepare disjoint sets of qubits
    disjoint_sets = [
        {
            "qubits": list(range(weight * it, weight * (it + 1))),
            "tier": weight,
            "children": [],
        }
        for it in range(nqubits // weight)
    ]
    nmodk = nqubits % weight
    if nmodk != 0:
        disjoint_sets.append(
            {
                "qubits": list(range(nqubits - nmodk, nqubits)),
                "tier": nmodk,
                "children": [],
            }
        )
    # reverse to have in ascending order of tier
    disjoint_sets = list(reversed(disjoint_sets))

    trees = disjoint_sets.copy()
    # combine lowest tier trees into one tree
    while len(trees) > 1:
        first_smallest = trees.pop(0)
        second_smallest = trees.pop(0)
        second_smallest["tier"] += first_smallest["tier"]
        second_smallest["children"].append(first_smallest)
        new = second_smallest
        # put new combined tree in list mantaining ordering
        trees.insert(sum(x["tier"] < new["tier"] for x in trees), new)

    root = trees[0]
    # We initialize |0>^(n-k)|1>^k  bitstring at root qubits
    circuit.add(gates.X(q) for q in root["qubits"][-weight:])

    # undo the union-by-tier algorithm:
    # split each root's (x) highest tier child y, and add WBD acting on both
    while len(trees) < len(disjoint_sets):
        cut_nodes = []
        for node in trees:
            if len(node["children"]) > 0:
                # find highest tier child
                y = max(node["children"], key=lambda x: x["tier"])
                # add WBD acting on both sets of qubits
                _add_wbd_gate(
                    circuit,
                    node["qubits"],
                    y["qubits"],
                    node["tier"],
                    y["tier"],
                    weight,
                )
                # cut tree splitting x and y
                node["tier"] -= y["tier"]
                node["children"].remove(y)
                cut_nodes.append(y)
        trees += cut_nodes

    for node in disjoint_sets:
        _add_dicke_unitary_gate(
            circuit, node["qubits"], min(weight, len(node["qubits"]))
        )

    return circuit


def _generate_rbs_pairs(nqubits: int, architecture: str, **kwargs):
    """Generate list of indexes representing the RBS connections.

    Creates circuit with all RBS initialised with 0.0 phase.

    Args:
        nqubits (int): number of qubits.
        architecture(str, optional): circuit architecture used for the unary loader.
            If ``diagonal``, uses a ladder-like structure.
            If ``tree``, uses a binary-tree-based structure.
            Defaults to ``tree``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        (:class:`qibo.models.circuit.Circuit`, list): Circuit composed of
        :class:`qibo.gates.gates.RBS` and list of indexes of target qubits per depth.
    """

    if architecture == "diagonal":
        pairs_rbs = np.arange(nqubits)
        pairs_rbs = [[pair] for pair in zip(pairs_rbs[:-1], pairs_rbs[1:])]

    if architecture == "tree":
        pairs_rbs = [[(0, int(nqubits / 2))]]
        indexes = list(pairs_rbs[0][0])
        for depth in range(2, int(math.log2(nqubits)) + 1):
            pairs_rbs_per_depth = [
                [(index, index + int(nqubits / 2**depth)) for index in indexes]
            ]
            pairs_rbs += pairs_rbs_per_depth
            indexes = list(np.array(pairs_rbs_per_depth).flatten())

    pairs_rbs = [
        [(nqubits - 1 - a, nqubits - 1 - b) for a, b in row] for row in pairs_rbs
    ]

    circuit = Circuit(nqubits, **kwargs)
    for row in pairs_rbs:
        for pair in row:
            circuit.add(gates.RBS(*pair, 0.0, trainable=True))

    return circuit, pairs_rbs


def _generate_rbs_angles(data, architecture: str, nqubits: int = None, backend=None):
    """Generate list of angles for RBS gates based on ``architecture``.

    Args:
        data (ndarray, optional): :math:`1`-dimensional array of data to be loaded.
        architecture(str, optional): circuit architecture used for the unary loader.
            If ``diagonal``, uses a ladder-like structure.
            If ``tree``, uses a binary-tree-based structure.
            Defaults to ``tree``.
        nqubits (int): Number of qubits. To be used then ``architecture="tree"``.

    Returns:
        list: List of phases for RBS gates.
    """
    backend = _check_backend(backend)

    if architecture == "diagonal":
        phases = [
            backend.np.arctan2(backend.calculate_vector_norm(data[k + 1 :]), data[k])
            for k in range(len(data) - 2)
        ]
        phases.append(backend.np.arctan2(data[-1], data[-2]))

    if architecture == "tree":
        if nqubits is None:  # pragma: no cover
            raise_error(
                TypeError,
                '``nqubits`` must be specified when ``architecture=="tree"``.',
            )

        j_max = int(nqubits / 2)

        r_array = np.zeros(nqubits - 1, dtype=float)
        phases = np.zeros(nqubits - 1, dtype=float)
        for j in range(1, j_max + 1):
            r_array[j_max + j - 2] = math.sqrt(
                data[2 * j - 1] ** 2 + data[2 * j - 2] ** 2
            )
            theta = math.acos(data[2 * j - 2] / r_array[j_max + j - 2])
            if data[2 * j - 1] < 0.0:
                theta = 2 * math.pi - theta
            phases[j_max + j - 2] = theta

        for j in range(j_max - 1, 0, -1):
            r_array[j - 1] = math.sqrt(r_array[2 * j] ** 2 + r_array[2 * j - 1] ** 2)
            phases[j - 1] = math.acos(r_array[2 * j - 1] / r_array[j - 1])

    phases = backend.cast(phases, dtype=phases[0].dtype)

    return phases


def _parametrized_two_qubit_gate(gate, q0, q1, params=None):
    """Return two-qubit gate initialized with or without phases."""
    if params is not None:
        return gate(q0, q1, *params)

    return gate(q0, q1)


def _next_nearest_layer(
    nqubits: int, gate, parameters, closed_boundary: bool, **kwargs
):
    """Create entangling layer with next-nearest-neighbour connectivity."""
    circuit = Circuit(nqubits, **kwargs)
    circuit.add(
        _parametrized_two_qubit_gate(gate, qubit, qubit + 2, parameters)
        for qubit in range(nqubits - 2)
    )

    if closed_boundary:
        circuit.add(_parametrized_two_qubit_gate(gate, nqubits - 1, 0, parameters))

    return circuit


def _pyramid_layer(nqubits: int, gate, parameters, **kwargs):
    """Create entangling layer in triangular shape."""
    _, pairs_gates = _generate_rbs_pairs(nqubits, architecture="diagonal")
    pairs_gates = pairs_gates[::-1]

    circuit = Circuit(nqubits, **kwargs)
    circuit.add(
        _parametrized_two_qubit_gate(gate, pair[0][1], pair[0][0], parameters)
        for pair in pairs_gates
    )
    circuit.add(
        _parametrized_two_qubit_gate(gate, pair[0][1], pair[0][0], parameters)
        for k in range(1, len(pairs_gates))
        for pair in pairs_gates[:-k]
    )

    return circuit


def _v_layer(nqubits: int, gate, parameters, **kwargs):
    """Create entangling layer in V shape."""
    _, pairs_gates = _generate_rbs_pairs(nqubits, architecture="diagonal")
    pairs_gates = pairs_gates[::-1]

    circuit = Circuit(nqubits, **kwargs)
    circuit.add(
        _parametrized_two_qubit_gate(gate, pair[0][1], pair[0][0], parameters)
        for pair in pairs_gates
    )
    circuit.add(
        _parametrized_two_qubit_gate(gate, pair[0][1], pair[0][0], parameters)
        for pair in pairs_gates[::-1][1:]
    )

    return circuit


def _x_layer(nqubits, gate, parameters, **kwargs):
    """Create entangling layer in X shape."""
    _, pairs_gates = _generate_rbs_pairs(nqubits, architecture="diagonal")
    pairs_gates = pairs_gates[::-1]

    middle = int(np.floor(len(pairs_gates) / 2))
    pairs_1 = pairs_gates[:middle]
    pairs_2 = pairs_gates[-middle:]

    circuit = Circuit(nqubits, **kwargs)

    for first, second in zip(pairs_1, pairs_2[::-1]):
        circuit.add(
            _parametrized_two_qubit_gate(gate, first[0][1], first[0][0], parameters)
        )
        circuit.add(
            _parametrized_two_qubit_gate(gate, second[0][1], second[0][0], parameters)
        )

    circuit.add(
        _parametrized_two_qubit_gate(
            gate,
            pairs_gates[middle][0][1],
            pairs_gates[middle][0][0],
            parameters,
        )
    )

    for first, second in zip(pairs_1[::-1], pairs_2):
        circuit.add(
            _parametrized_two_qubit_gate(gate, first[0][1], first[0][0], parameters)
        )
        circuit.add(
            _parametrized_two_qubit_gate(gate, second[0][1], second[0][0], parameters)
        )

    return circuit


def _non_trivial_layers(
    nqubits: int,
    architecture: str = "pyramid",
    entangling_gate: Union[str, gates.Gate] = "RBS",
    closed_boundary: bool = False,
    **kwargs,
):
    """Create more intricate entangling layers of different shapes.

    Args:
        nqubits (int): number of qubits.
        architecture (str, optional): Architecture of the entangling layer.
            In alphabetical order, options are ``"next_nearest"``, ``"pyramid"``,
            ``"v"``, and ``"x"``. The ``"x"`` architecture is only defined for
            an even number of qubits. Defaults to ``"pyramid"``.
        entangling_gate (str or :class:`qibo.gates.Gate`, optional): Two-qubit gate to be used
            in the entangling layer. If ``entangling_gate`` is a parametrized gate,
            all phases are initialized as :math:`0.0`. Defaults to  ``"CNOT"``.
        closed_boundary (bool, optional): If ``True`` and ``architecture="next_nearest"``,
            adds a closed-boundary condition to the entangling layer. Defaults to ``False``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit containing layer of two-qubit gates.
    """

    gate = (
        getattr(gates, entangling_gate)
        if isinstance(entangling_gate, str)
        else entangling_gate
    )

    parameters = list(signature(gate).parameters)
    parameters = (0.0,) * (len(parameters) - 3) if len(parameters) > 2 else None

    if architecture == "next_nearest":
        return _next_nearest_layer(nqubits, gate, parameters, closed_boundary, **kwargs)

    if architecture == "v":
        return _v_layer(nqubits, gate, parameters, **kwargs)

    if architecture == "x":
        return _x_layer(nqubits, gate, parameters, **kwargs)

    return _pyramid_layer(nqubits, gate, parameters, **kwargs)


def _angle_mod_two_pi(angle):
    """Return angle mod 2pi."""
    return angle % (2 * np.pi)


def _get_markers(bitstring, last_run: bool = False):
    """Subroutine of the Ehrlich algorithm."""
    nqubits = len(bitstring)
    markers = [len(bitstring) - 1]
    for ind, value in zip(range(nqubits - 2, -1, -1), bitstring[::-1][1:]):
        if value == bitstring[-1]:
            markers.append(ind)
        else:
            break

    markers = set(markers)

    if not last_run:
        markers = set(range(nqubits)) - markers

    return markers


def _get_next_bistring(bitstring, markers, hamming_weight):
    """Subroutine of the Ehrlich algorithm."""
    if len(markers) == 0:  # pragma: no cover
        return bitstring

    new_bitstring = np.copy(bitstring)

    nqubits = len(new_bitstring)

    indexes = np.argsort(bitstring)
    zeros, ones = np.sort(indexes[:-hamming_weight]), np.sort(indexes[-hamming_weight:])

    max_index = max(markers)
    nearest_one = ones[ones > max_index]
    nearest_one = None if len(nearest_one) == 0 else nearest_one[0]
    if new_bitstring[max_index] == 0 and nearest_one is not None:
        new_bitstring[max_index] = 1
        new_bitstring[nearest_one] = 0
    else:
        farthest_zero = zeros[zeros > max_index]
        if nearest_one is not None:
            farthest_zero = farthest_zero[farthest_zero < nearest_one]
        farthest_zero = farthest_zero[-1]
        new_bitstring[max_index] = 0
        new_bitstring[farthest_zero] = 1

    markers.remove(max_index)
    last_run = _get_markers(new_bitstring, last_run=True)
    markers = markers | (set(range(max_index + 1, nqubits)) - last_run)

    new_ones = np.argsort(new_bitstring)[-hamming_weight:]
    controls = list(set(ones) & set(new_ones))
    difference = new_bitstring - bitstring
    qubits = [np.where(difference == -1)[0][0], np.where(difference == 1)[0][0]]

    return new_bitstring, markers, [qubits, controls]


def _ehrlich_algorithm(initial_string, return_indices: bool = True):
    """Return list of bitstrings with mininal Hamming distance between consecutive strings.

    Based on the Gray code called Ehrlich algorithm. For more details, please see Ref. [1].

    Args:
        initial_string (ndarray): initial bitstring as an :math:`1`-dimensional array
            of size :math:`n`. All ones in the bitstring need to be consecutive.
            For instance, for :math:`n = 6` and :math:`k = 2`, the bistrings
            :math:`000011` and :math:`001100` are examples of acceptable inputs.
            In contrast, :math:`001001` is not an acceptable input.
        return_indices (bool, optional): if ``True``, returns the list of indices of
            qubits that act like controls and targets of the circuit to be created.
            Defaults to ``True``.

    Returns:
        list or tuple(list, list): If ``return_indices=False``, returns list containing
        sequence of bistrings in the order generated by the Gray code.
        If ``return_indices=True`` returns tuple with the aforementioned list and the list
        of control anf target qubits of gates to be implemented based on the sequence of
        bitstrings.

    References:
        1. R. M. S. Farias, T. O. Maciel, G. Camilo, R. Lin, S. Ramos-Calderer, and L. Aolita,
        *Quantum encoder for fixed-Hamming-weight subspaces*
        `Phys. Rev. Applied 23, 044014 (2025) <https://doi.org/10.1103/PhysRevApplied.23.044014>`_.
    """
    k = np.unique(initial_string, return_counts=True)
    if len(k[1]) == 1:  # pragma: no cover
        return ["".join([str(item) for item in initial_string])]

    k = k[1][1]
    n = len(initial_string)
    n_choose_k = int(binom(n, k))

    markers = _get_markers(initial_string, last_run=False)
    string = initial_string
    strings = ["".join(string[::-1].astype(str))]
    controls_and_targets = []
    for _ in range(n_choose_k - 1):
        string, markers, c_and_t = _get_next_bistring(string, markers, k)
        strings.append("".join(string[::-1].astype(str)))
        controls_and_targets.append(c_and_t)

    if return_indices:
        return strings, controls_and_targets

    return strings


def _get_gate(
    qubits_in: List[int],
    qubits_out: List[int],
    controls: List[int],
    theta: float,
    phi: float,
    complex_data: bool = False,
):
    """Return gate(s) necessary to encode a complex amplitude in a given computational basis state,
    given the computational basis state used to encode the previous amplitude.

    Information about computational basis states in question are contained
    in the indexing of ``qubits_in``, ``qubits_out``, and ``controls``.

    Args:
        qubits_in (list): list of qubits with ``in`` label.
        qubits_out (list): list of qubits with ``out`` label.
        controls (list): list of qubits that control the resulting gate.
        theta (float): first phase, used to encode the ``abs`` of amplitude.
        phi (float): second phase, used to encode the complex phase of amplitude.
        complex_data (bool): if ``True``, uses :class:`qibo.gates.U3` to as basis gate.
            If ``False``, uses :class:`qibo.gates.RY` as basis gate. Defaults to ``False``.

    Returns:
        List[:class:`qibo.gates.Gate`]: gate(s) to be added to circuit.

    References:
        1. R. M. S. Farias, T. O. Maciel, G. Camilo, R. Lin, S. Ramos-Calderer, and L. Aolita,
        *Quantum encoder for fixed-Hamming-weight subspaces*
        `Phys. Rev. Applied 23, 044014 (2025) <https://doi.org/10.1103/PhysRevApplied.23.044014>`_.
    """
    if len(qubits_in) == 0 and len(qubits_out) == 1:  # pragma: no cover
        gate_list = (
            gates.U3(*qubits_out, 2 * theta, 2 * phi, 0.0).controlled_by(*controls)
            if complex_data
            else gates.RY(*qubits_out, 2 * theta).controlled_by(*controls)
        )
        gate_list = [gate_list]
    elif len(qubits_in) == 1 and len(qubits_out) == 1:
        ## chooses best combination of complex RBS gate
        ## given number of controls and if data is real or complex
        gate_list = []
        gate = gates.RBS(*qubits_in, *qubits_out, theta)
        if len(controls) > 0:
            gate = gate.controlled_by(*controls)
        gate_list.append(gate)

        if complex_data:
            gate = [gates.RZ(*qubits_in, -phi), gates.RZ(*qubits_out, phi)]
            if len(controls) > 0:
                gate = [g.controlled_by(*controls) for g in gate]
            gate_list.extend(gate)

    else:  # pragma: no cover
        # Important for future sparse encoder
        gate_list = [
            gates.GeneralizedRBS(
                list(qubits_in), list(qubits_out), theta, phi
            ).controlled_by(*controls)
        ]

    return gate_list


def _get_phase_gate_correction(last_string, phase: float):
    """Return final gate of HW-k circuits that encode complex data."""

    # to avoid circular import error
    from qibo.quantum_info.utils import (  # pylint: disable=import-outside-toplevel
        hamming_weight,
    )

    if isinstance(last_string, str):
        last_string = np.asarray(list(last_string), dtype=int)

    last_weight = hamming_weight(last_string)
    last_ones = np.argsort(last_string)
    last_zero = int(last_ones[0])
    last_controls = [int(qubit) for qubit in last_ones[-last_weight:]]

    # adding an RZ gate to correct the phase of the last amplitude encoded
    return gates.RZ(last_zero, 2 * phase).controlled_by(*last_controls)


def _binary_encoder_hopf(
    data, nqubits, complex_data, backend=None, **kwargs
):  # pylint: disable=unused-argument
    # TODO: generalize to complex-valued data
    backend = _check_backend(backend)

    dims = 2**nqubits

    base_strings = [f"{elem:0{nqubits}b}" for elem in range(dims)]
    base_strings = backend.np.reshape(base_strings, (-1, 2))
    strings = [base_strings]
    for _ in range(nqubits - 1):
        base_strings = backend.np.reshape(base_strings[:, 0], (-1, 2))
        strings.append(base_strings)
    strings = strings[::-1]

    targets_and_controls = []
    for pairs in strings:
        for pair in pairs:
            targets, controls, anticontrols = [], [], []
            for k, bits in enumerate(zip(pair[0], pair[1])):
                if bits == ("0", "0"):
                    anticontrols.append(k)
                elif bits == ("1", "1"):
                    controls.append(k)
                elif bits == ("0", "1"):
                    targets.append(k)
            targets_and_controls.append([targets, controls, anticontrols])

    circuit = Circuit(nqubits, **kwargs)
    for targets, controls, anticontrols in targets_and_controls:
        gate_list = []
        if len(anticontrols) > 0:
            gate_list.append(gates.X(qubit) for qubit in anticontrols)
        gate_list.append(
            gates.RY(targets[0], 0.0).controlled_by(*(controls + anticontrols))
        )
        if len(anticontrols) > 0:
            gate_list.append(gates.X(qubit) for qubit in anticontrols)
        circuit.add(gate_list)

    angles = _generate_rbs_angles(data, "tree", dims, backend=backend)
    circuit.set_parameters(2 * angles)

    return circuit


def _binary_encoder_hyperspherical(
    data, nqubits, complex_data: bool, backend=None, **kwargs
):
    backend = _check_backend(backend)

    dims = 2**nqubits
    last_qubit = nqubits - 1

    indexes_to_double, lex_order_global = [0], [0]

    circuit = Circuit(nqubits, **kwargs)
    if complex_data:
        circuit.add(gates.U3(last_qubit, 0.0, 0.0, 0.0))
    else:
        circuit.add(gates.RY(last_qubit, 0.0))

    cummul_n_k = 0
    initial_string = np.array([1] + [0] * (nqubits - 1))
    for weight in range(1, nqubits):
        n_choose_k = int(binom(nqubits, weight))
        cummul_n_k += n_choose_k
        placeholder = np.random.rand(n_choose_k)
        if complex_data:
            placeholder = placeholder.astype(complex) + 1j * np.random.rand(n_choose_k)
        placeholder = backend.cast(placeholder, dtype=placeholder[0].dtype)

        circuit += hamming_weight_encoder(
            placeholder,
            nqubits,
            weight,
            full_hwp=True,
            optimize_controls=False,
            phase_correction=False,
            initial_string=initial_string,
            backend=backend,
            **kwargs,
        )

        # add gate to be place between blocks of Hamming-weight encoders
        gate, lex_order, initial_string, phase_index = _intermediate_gate(
            initial_string,
            weight,
            last_qubit,
            cummul_n_k,
            complex_data,
        )
        circuit.add(gate)
        lex_order_global.extend(lex_order)
        indexes_to_double.append(phase_index)

    # sort data such that the encoding is performed in lexicographical order
    lex_order_global.append(dims - 1)
    lex_order_sorted = np.sort(np.copy(lex_order_global))
    lex_order_global = [
        np.where(lex_order_sorted == num)[0][0] for num in lex_order_global
    ]
    data = data[lex_order_global]
    del lex_order_global, lex_order_sorted

    _data = backend.np.abs(data) if complex_data else data

    thetas = _generate_rbs_angles(_data, architecture="diagonal", backend=backend)

    phis = backend.np.zeros(len(thetas) + 1)
    if complex_data:
        phis[0] = _angle_mod_two_pi(-backend.np.angle(data[0]))
        for k in range(1, len(phis)):
            phis[k] = _angle_mod_two_pi(
                -backend.np.angle(data[k]) + backend.np.sum(phis[:k])
            )
    phis = backend.cast(phis, dtype=phis[0].dtype)

    zero_casted = backend.cast(0.0, dtype=backend.np.float64)  # because of GPU backends

    angles = []
    for k, (theta, phi) in enumerate(zip(thetas, phis)):
        if k in indexes_to_double:
            angle = [2 * theta, 2 * phi, zero_casted] if complex_data else [2 * theta]
        else:
            angle = [theta, -phi, phi] if complex_data else [theta]

        angles.extend(angle)

    if complex_data:
        angles[-2] = 2 * _angle_mod_two_pi(
            (backend.np.angle(data[-1]) - backend.np.angle(data[-2])) / 2
        )
        angles[-1] = 2 * _angle_mod_two_pi(
            (-0.5) * (backend.np.angle(data[-2]) + backend.np.angle(data[-1]))
            + backend.np.sum(phis[:-2])
        )

    circuit.set_parameters(angles)

    return circuit


def _intermediate_gate(
    initial_string,
    weight,
    last_qubit,
    cummul_n_k,
    complex_data,
):
    """Calculate where to place the intermediate gate by finding the last string
    of the previous Hamming-weight block that was encoded"""

    # sort data such that the encoding is performed in lexicographical order
    bitstrings = _ehrlich_algorithm(initial_string, False)
    initial_string = bitstrings[-1]
    lex_order = [int(string, 2) for string in bitstrings]

    controls = [item.start() for item in finditer("1", initial_string)]
    index = (
        initial_string.find("0")
        if weight % 2 == 0
        else last_qubit - initial_string[::-1].find("0")
    )
    initial_string = np.array(list(initial_string), dtype=int)
    initial_string[index] = 1
    initial_string = initial_string[::-1]

    phase_index = cummul_n_k
    gate = (
        gates.U3(index, 0.0, 0.0, 0.0).controlled_by(*controls)
        if complex_data
        else gates.RY(index, 0.0).controlled_by(*controls)
    )

    return gate, lex_order, initial_string, phase_index


def _sort_data_sparse(data, nqubits, backend):
    from qibo.quantum_info.utils import (  # pylint: disable=import-outside-toplevel
        hamming_weight,
    )

    # TODO: Fix this mess with qibo native data types
    try:
        test_dtype = bool("int" in str(data[0][0].dtype))
    except AttributeError:
        test_dtype = bool("int" in str(type(data[0][0])))

    _data = [(f"{row[0]:0{nqubits}b}", row[1]) for row in data] if test_dtype else data

    _data = sorted(_data, key=lambda x: hamming_weight(x[0]))

    bitstrings_sorted, data_sorted = zip(*_data)
    bitstrings_sorted = [
        np.array(list(string)).astype(int) for string in bitstrings_sorted
    ]

    bitstrings_sorted = backend.cast(bitstrings_sorted, dtype=backend.np.int8)
    data_sorted = backend.cast(data_sorted, dtype=data_sorted[0].dtype)

    return data_sorted, bitstrings_sorted


def _get_gate_sparse(
    distance,
    difference,
    touched_qubits,
    complex_data,
    controls,
    hw_0,
    hw_1,
    theta,
    phi,
    backend=None,
):
    backend = _check_backend(backend)
    if distance == 1:
        qubit = int(backend.np.where(difference == 1)[0][0])
        if qubit not in touched_qubits:
            touched_qubits.append(qubit)
        gate = (
            gates.U3(qubit, 2 * theta, 2 * phi, 0.0).controlled_by(*controls)
            if complex_data
            else gates.RY(qubit, 2 * theta).controlled_by(*controls)
        )
    elif distance == 2 and hw_0 == hw_1:
        qubits = [
            int(np.where(difference == -1)[0][0]),
            int(np.where(difference == 1)[0][0]),
        ]
        touched_qubits += list(set(qubits) - set(touched_qubits))
        qubits_in = [int(qubits[0])]
        qubits_out = [int(qubits[1])]
        qubits = [np.where(difference == -1)[0][0], np.where(difference == 1)[0][0]]

        gate = _get_gate(
            qubits_in,
            qubits_out,
            controls,
            theta,
            phi,
            complex_data,
        )
    else:
        qubits = [np.where(difference == -1)[0], np.where(difference == 1)[0]]
        for row in qubits:
            row = [int(elem) for elem in row]
            touched_qubits += list(set(row) - set(touched_qubits))
        qubits_in = [int(qubit) for qubit in qubits[0]]
        qubits_out = [int(qubit) for qubit in qubits[1]]
        gate = gates.GeneralizedRBS(qubits_in, qubits_out, theta, -phi).controlled_by(
            *controls
        )

    return gate


def _get_phase_gate_correction_sparse(
    last_string,
    second_to_last_string,
    nqubits,
    last_data,
    second_to_last_data,
    circuit,
    phis,
):
    from qibo.quantum_info.utils import (  # pylint: disable=import-outside-toplevel
        hamming_weight,
    )

    hw_0 = hamming_weight(second_to_last_string)
    hw_1 = hamming_weight(last_string)
    if hw_1 == nqubits and hw_0 == nqubits - 1:
        phi = _angle_mod_two_pi(np.angle(last_data) - np.angle(second_to_last_data))
        lamb = _angle_mod_two_pi(
            -(np.angle(second_to_last_data) + np.angle(last_data))
            + 2 * np.sum(phis[:-2])
        )
        _gate = circuit.queue[-1]
        gate = gates.U3(
            *_gate.target_qubits, _gate.init_kwargs["theta"], phi, lamb
        ).controlled_by(*_gate.control_qubits)

        new_queue = circuit.queue[:-1] + [gate]

        return new_queue

    if hw_1 == nqubits:  # pragma: no cover
        first_one = np.argsort(second_to_last_string)[0]
        other_ones = list(set(range(nqubits)) ^ {first_one})
        gate = gates.RZ(first_one, 2 * phis[-1]).controlled_by(*other_ones)
    else:
        gate = _get_phase_gate_correction(last_string, phis[-1])

    return gate


def _add_scs_gate(circuit: Circuit, qubits: List[int], weight: int):
    """In-place addition of a Split & Cyclic Shift (SCS) gate to ``circuit``.
    Implements the SCS_{n,k} unitary from Definition 3 of the paper [1].
    Acts on the last weight+1 qubits of the nqubits passed qubits.
    """
    if weight == 0:
        return  # SCS_{n,0} is identity

    nqubits = len(qubits)
    last_qubit = qubits[-1]  # qubits[nqubits - 1]
    target_qubit = qubits[-2]
    # first_qubit = qubits[nqubits - 1 - weight]

    # Gate (i) - acts on last two qubits
    theta = 2 * np.arccos(np.sqrt(1 / nqubits))
    circuit.add(gates.CNOT(target_qubit, last_qubit))
    circuit.add(gates.RY(target_qubit, theta).controlled_by(last_qubit))
    circuit.add(gates.CNOT(target_qubit, last_qubit))

    # Gates (ii)_ℓ for ℓ from 2 to k
    for l in range(2, weight + 1):
        theta = 2 * np.arccos(np.sqrt(l / nqubits))
        target_qubit = qubits[-(l + 1)]
        control_qubit = qubits[-l]

        # Implement the three-qubit gate (ii)_ℓ
        circuit.add(gates.CNOT(target_qubit, last_qubit))
        circuit.add(
            gates.RY(target_qubit, theta).controlled_by(control_qubit, last_qubit)
        )
        circuit.add(gates.CNOT(target_qubit, last_qubit))


def _add_dicke_unitary_gate(circuit: Circuit, qubits: List[int], weight: int):
    """In-place addition to ``circuit`` of a U_{n,k} gate from Definition 2 of the paper [1]."""
    nqubits = len(qubits)
    for m in range(nqubits, weight, -1):
        # Add SCS_{m,k} acting on last k+1 qubits
        _add_scs_gate(circuit, qubits[:m], weight)

    # Recursively build the unitary U_n,k
    for m in range(weight, 0, -1):
        # Add SCS_{m,m-1} acting on last m qubits
        _add_scs_gate(circuit, qubits[:m], m - 1)


def _add_wbd_gate(
    circuit: Circuit,
    first_register: List[int],
    second_register: List[int],
    nqubits: int,
    mqubits: int,
    weight: int,
):
    """In-place addition of a Weight Distribution Block (WBD) to ``circuit``.
    Implements the :math:`WBD^{n,m}_k` unitary from Definition 2 of the paper [2].
    Only acts on first_register and second_register, last k qubits
    Our circuit is mirrored, as paper [2] uses a top-bottom circuit <-> right-left bitstring convention
    """

    if mqubits > nqubits / 2:
        raise_error(ValueError, "``m`` must not be greater than ``n - m``.")

    # only acts on last k qubits
    first_register = first_register[-weight:]
    second_register = second_register[-weight:]

    # if m>k, m is truncated. Operations involving the most significant k-m digits can be removed

    # (1) Switching from unary encoding to one hot encoding
    circuit.add(gates.CNOT(q, q + 1) for q in first_register[-2::-1])

    # (2) Adding a supperposition of hamming weights into the second register
    # this can be optimized
    # We follow Figure 4, but adjust definition of xi and si (suffix sum) to match
    theta_gate = lambda qubit, theta: gates.RY(qubit, 2 * math.acos(theta))
    for l in range(weight, 0, -1):
        x = [
            math.comb(mqubits, i) * math.comb(nqubits - mqubits, l - i)
            for i in range(l)
        ]
        s = math.comb(nqubits, l)
        circuit.add(
            theta_gate(second_register[-1], math.sqrt(x[0] / s)).controlled_by(
                first_register[-l]
            )
        )
        s -= x[0]
        for qubit in range(1, min(l, mqubits)):
            circuit.add(
                theta_gate(
                    second_register[-qubit - 1], math.sqrt(x[qubit] / s)
                ).controlled_by(first_register[-l], second_register[-qubit])
            )
            s -= x[qubit]

    # (3) Go back to unary encoding, undoing one hot encoding
    circuit.add(gates.CNOT(q, q + 1) for q in first_register[:-1])

    # (4) Substracting weight i in the first register from weight l in the second register.
    # fredkin, controlled swaps (decomposed into CNOT and Toffoli)
    fredkin = lambda control, s1, s2: (
        gates.CNOT(s2, s1),
        gates.TOFFOLI(control, s1, s2),
        gates.CNOT(s2, s1),
    )

    dif = max(0, weight - mqubits)
    for control in range(dif, weight):
        for q in range(control):
            circuit.add(
                fredkin(
                    second_register[control - dif],
                    first_register[control - q],
                    first_register[control - q - 1],
                )
            )
        circuit.add(gates.CNOT(second_register[control - dif], first_register[0]))


def _perm_column_ops(
    indices: list[int],
    n: int,
    backend=None,
):
    """Return (ell, gate_list) performing duplicate‑col removal + compaction."""
    backend = _check_backend(backend)
    # flatten the (x_0, x_1),...(x_{2m-2}, x_{2m-1})
    # We construct a matrix composed of
    # xj to track the changes in xj where xj,k represents the k-th bit and the
    # bits are arranged from the least significant bit to the most significant bit
    indices = list(sum(indices, ()))
    A = []
    for x in indices:
        bits = []
        for k in range(n):
            bits.append((x >> k) & 1)
        A.append(bits)
    A = backend.cast(A, dtype=backend.np.int8)
    ncols = A.shape[1]
    # initialize the list of gates
    qgates = []

    # number of non-zero columns
    ell = 0
    flag = backend.np.zeros(n, dtype=int)
    for idxj in range(ncols):
        if any(elem != 0 for elem in A[:, idxj]):
            ell += 1
            flag[idxj] = 1

            # look for columns that are equal to A[:,idxj]
            for idxk in range(idxj + 1, ncols):
                if backend.np.array_equal(A[:, idxj], A[:, idxk]):
                    qgates.append(gates.CNOT(n - idxj - 1, n - idxk - 1))
                    # this should transform the k-th column into an all-zero column
                    A[:, idxk] = 0

    # Now, we need to swap the ell non-zero columns to the first ell columns
    for idxk in range(ell, ncols):
        if not backend.np.array_equal(A[:, idxk], backend.np.zeros_like(A[:, idxk])):
            for k in range(len(flag)):
                if flag[k] == 0:
                    flag[k] = 1
                    flag[idxk] = 0

                    qgates.append(gates.SWAP(n - idxk - 1, n - k - 1))

                    bits = A[:, idxk].copy()
                    A[:, idxk] = A[:, k]
                    A[:, k] = bits
                    break

    return ell, qgates, A


def _perm_row_ops(A, ell: int, m: int, n: int, backend=None):
    """Return gates that reduce all rows after row0 to target form."""
    backend = _check_backend(backend)

    log2m = int(backend.np.log2(2 * m))
    atilde = backend.np.array(
        [[(x >> k) & 1 for k in range(n)] for x in range(2 * m)], dtype=int
    )

    qgates = []
    nrows = A.shape[0]
    ncols = A.shape[1]
    # Start with the first row (indexed as row 0)
    for k in range(ncols):
        # If we find a0,k = 1 for any 0 <= k <= n−1
        if A[0, k] == 1:
            qgates.append(gates.X(n - k - 1))
            A[:, k] = (A[:, k] + 1) % 2

    for j in range(1, nrows):
        flag = False
        for k in range(log2m, ncols):
            if A[j, k] != 0:
                flag = True
                break

        if not flag:
            # There is no element b_{j},k != 0 for k > {log2m-1}"
            ctrls = [n - l - 1 for l in range(ncols) if A[j, l] != 0]
            qgates.append(gates.X(n - log2m - 1).controlled_by(*ctrls))
            ctrls = [l for l in range(ncols) if A[j, l] == 1]

            # check whether the gate is applied on other rows
            for l in range(j, nrows):
                if all(elem == 1 for elem in A[l, ctrls]):
                    A[l, log2m] = (A[l, log2m] + 1) % 2

        # There is always an element b_{j},k != 0 for k > {log2m-1}
        for k in range(log2m, ncols):
            if A[j, k] != 0:
                # Element b_{j},{k} != 0, {k} > {log2m-1}"
                for kprime in range(ell):
                    # There is a typo in the paper
                    # b_{j},{kprime} should be different from Ã_{j},{kprime} not Ã_{j},{k}
                    if kprime != k and A[j, kprime] != atilde[j, kprime]:
                        qgates.append(gates.CNOT(n - k - 1, n - kprime - 1))
                        # check whether the gate is applied on other rows
                        for l in range(nrows):
                            if A[l, k] == 1:
                                A[l, kprime] = (A[l, kprime] + 1) % 2

                # Let us clean the element b_{j},{k}

                # There is another typo in the paper
                # the control qubits for this gate correspond to the non-zero elements in row j of matrix A, not Ã
                ctrls = [n - l - 1 for l in range(k) if A[j, l] != 0]
                qgates.append(gates.X(n - k - 1).controlled_by(*ctrls))
                ctrls = [l for l in range(k) if A[j, l] != 0]

                # check whether the gate is applied on other rows
                for l in range(nrows):
                    if all(elem == 1 for elem in A[l, ctrls]):
                        A[l, k] = (A[l, k] + 1) % 2

    return qgates


def _perm_pair_flip_ops(n: int, m: int, backend=None):
    """Implement σ_{i,2} as X fan‑in + MCX + X fan‑out."""
    backend = _check_backend(backend)
    # let us flip the first qubit when the last {int(n-math.log2(2*m))} qubits are all in the state |0⟩
    prefix = int(backend.np.ceil(backend.np.log2(2 * m)))
    x_qubits, controls = range(prefix, n), range(n - prefix)
    qgates = [gates.X(n - q - 1) for q in x_qubits]
    qgates.append(gates.X(n - 1).controlled_by(*controls))  # flip qubit 0
    qgates.extend(gates.X(n - q - 1) for q in x_qubits)

    return qgates


def permutation_synthesis(
    sigma: Union[List[int], tuple[int, ...]], m: int = 2, backend=None, **kwargs
):
    """Return circuit that implements a given permutation.

    Given permutation ``sigma`` on :math:`\\{0, \\, 1, \\, \\dots, \\, d-1\\}`
    and a power‑of‑two budget ``m``, this function factors ``sigma``
    into the fewest layers :math:`\\sigma_{1}, \\, \\sigma_{2}, \\, \\cdots, \\, \\sigma_{t}`
    such that:
        - each layer has at most :math:`m` disjoint transpositions;
        - each layer moves a power‑of‑two number of indices.

    The function returns a circuit synthesis of ``sigma``.

    Args:
        sigma (list or tuple): permutation description on :math:`\\{0, \\, 1, \\, \\dots, \\, d-1\\}`.
        m (int): power‑of‑two budget. Defauls to :math:`2`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that implements the permutation ``sigma``.

    References:
        1. L. Li, and J. Luo,
        *Nearly Optimal Circuit Size for Sparse Quantum State Preparation*
        `arXiv:2406.16142 (2024) <https://doi.org/10.48550/arXiv.2406.16142>`_.
    """
    backend = _check_backend(backend)

    if isinstance(sigma, tuple):
        sigma = list(sigma)

    if not isinstance(sigma, (list, tuple)):
        raise_error(
            TypeError,
            f"Permutation ``sigma`` must be either a ``list`` or a ``tuple`` of ``int``s.",
        )

    nqubits = int(backend.np.ceil(backend.np.log2(len(sigma))))
    if sum([abs(s - i) for s, i in zip(sorted(sigma), range(2**nqubits))]) != 0:
        raise_error(
            ValueError, "Permutation sigma must contain all indices {0,...,n-1}"
        )

    if m > 0 and (m & (m - 1)) != 0:
        raise_error(ValueError, f"budget m must be a power‑of‑two")

    from qibo.quantum_info.utils import (  # pylint: disable=import-outside-toplevel
        decompose_permutation,
    )

    # factor sigma into the fewest layers such that
    # each layer has at most m disjoint transpositions, and
    # each layer moves a power‑of‑two number of indices
    layers = decompose_permutation(sigma, m)

    circuit = Circuit(nqubits)
    # in case we have more than one permutation to do, do it in layers
    for layer in layers:
        m = len(layer)
        ell, col_gates, A = _perm_column_ops(layer, nqubits, backend)
        row_gates = _perm_row_ops(A, ell, m, nqubits, backend)
        flip_gates = _perm_pair_flip_ops(nqubits, m, backend)
        col_row = col_gates + row_gates
        circuit.add(col_row)
        circuit.add(flip_gates)
        circuit.add(col_row[::-1])

    return circuit
