"""Module with functions that encode classical data into quantum circuits."""

import math
from inspect import signature
from typing import List, Optional, Union

import numpy as np
from scipy.special import binom

from qibo import gates
from qibo.config import raise_error
from qibo.gates.gates import _check_engine
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
            TypeError, f"nqubits must be type int, but it is type {type(nqubits)}."
        )

    if nqubits is None:
        if isinstance(basis_element, int):
            raise_error(
                ValueError, f"nqubits must be specified when basis_element is type int."
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


def phase_encoder(data, rotation: str = "RY", **kwargs):
    """Create circuit that performs the phase encoding of ``data``.

    Args:
        data (ndarray or list): :math:`1`-dimensional array of phases to be loaded.
        rotation (str, optional): If ``"RX"``, uses :class:`qibo.gates.gates.RX` as rotation.
            If ``"RY"``, uses :class:`qibo.gates.gates.RY` as rotation.
            If ``"RZ"``, uses :class:`qibo.gates.gates.RZ` as rotation.
            Defaults to ``"RY"``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads ``data`` in phase encoding.
    """
    if isinstance(data, list):
        data = np.array(data)

    if len(data.shape) != 1:
        raise_error(
            TypeError,
            f"``data`` must be a 1-dimensional array, but it has dimensions {data.shape}.",
        )

    if not isinstance(rotation, str):
        raise_error(
            TypeError,
            f"``rotation`` must be type str, but it is type {type(rotation)}.",
        )

    if rotation not in ["RX", "RY", "RZ"]:
        raise_error(ValueError, f"``rotation`` {rotation} not found.")

    nqubits = len(data)
    gate = getattr(gates, rotation.upper())

    circuit = Circuit(nqubits, **kwargs)
    circuit.add(gate(qubit, 0.0) for qubit in range(nqubits))
    circuit.set_parameters(data)

    return circuit


def binary_encoder(data, **kwargs):
    """Create circuit that encodes real-valued ``data`` in all amplitudes of the computational basis.

    ``data`` has to be normalized with respect to the Hilbert-Schmidt norm.
    Resulting circuit parametrizes ``data`` in Hopf coordinates in the
    :math:`(2^{n} - 1)`-unit sphere.

    Args:
        data (ndarray): :math:`1`-dimensional array or length :math:`2^{n}`
            to be loaded in the amplitudes of a :math:`n`-qubit quantum state.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads ``data`` in binary encoding.
    """
    dims = len(data)
    nqubits = float(np.log2(dims))
    if not nqubits.is_integer():
        raise_error(ValueError, "`data` size must be a power of 2.")
    nqubits = int(nqubits)

    base_strings = [f"{elem:0{nqubits}b}" for elem in range(dims)]
    base_strings = np.reshape(base_strings, (-1, 2))
    strings = [base_strings]
    for _ in range(nqubits - 1):
        base_strings = np.reshape(base_strings[:, 0], (-1, 2))
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

    angles = _generate_rbs_angles(data, dims, "tree")
    circuit.set_parameters(2 * angles)

    return circuit


def unary_encoder(data, architecture: str = "tree", **kwargs):
    """Create circuit that performs the (deterministic) unary encoding of ``data``.

    Args:
        data (ndarray): :math:`1`-dimensional array of data to be loaded.
        architecture(str, optional): circuit architecture used for the unary loader.
            If ``diagonal``, uses a ladder-like structure.
            If ``tree``, uses a binary-tree-based structure.
            Defaults to ``tree``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads ``data`` in unary representation.
    """
    if isinstance(data, list):
        data = np.array(data)

    if len(data.shape) != 1:
        raise_error(
            TypeError,
            f"``data`` must be a 1-dimensional array, but it has dimensions {data.shape}.",
        )

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
    phases = _generate_rbs_angles(data, nqubits, architecture)
    circuit.set_parameters(phases)

    return circuit


def unary_encoder_random_gaussian(
    nqubits: int, architecture: str = "tree", seed=None, **kwargs
):
    """Create a circuit that performs the unary encoding of a random Gaussian state.

    At depth :math:`h` of the tree architecture, the angles :math:`\\theta_{k} \\in [0, 2\\pi]` of the the
    gates :math:`RBS(\\theta_{k})` are sampled from the following probability density function:

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
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads a random Gaussian array in unary representation.

    References:
        1. A. Bouland, A. Dandapani, and A. Prakash, *A quantum spectral method for simulating
        stochastic processes, with applications to Monte Carlo*.
        `arXiv:2303.06719v1 [quant-ph] <https://arxiv.org/abs/2303.06719>`_
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

    if architecture != "tree":
        raise_error(
            NotImplementedError,
            f"Currently, this function only accepts ``architecture=='tree'``.",
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

    circuit.set_parameters(phases)

    return circuit


def hamming_weight_encoder(
    data,
    nqubits: int,
    weight: int,
    full_hwp: bool = False,
    optimize_controls: bool = True,
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
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads ``data``in
        Hamming-weight-:math:`k` representation.

    References:
        1. R. M. S. Farias, T. O. Maciel, G. Camilo, R. Lin, S. Ramos-Calderer, and L. Aolita,
        *Quantum encoder for fixed Hamming-weight subspaces*
        `arXiv:2405.20408 [quant-ph] <https://arxiv.org/abs/2405.20408>`_.
    """
    complex_data = bool(data.dtype in [complex, np.dtype("complex128")])

    initial_string = np.array([1] * weight + [0] * (nqubits - weight))
    bitstrings, targets_and_controls = _ehrlich_algorithm(initial_string)

    # sort data such that the encoding is performed in lexicographical order
    lex_order = [int(string, 2) for string in bitstrings]
    lex_order_sorted = np.sort(np.copy(lex_order))
    lex_order = [np.where(lex_order_sorted == num)[0][0] for num in lex_order]
    data = data[lex_order]
    del lex_order, lex_order_sorted

    # Calculate all gate phases necessary to encode the amplitudes.
    _data = np.abs(data) if complex_data else data
    thetas = _generate_rbs_angles(_data, nqubits, architecture="diagonal")
    thetas = np.asarray(thetas, dtype=type(thetas[0]))
    phis = np.zeros(len(thetas) + 1)
    if complex_data:
        phis[0] = _angle_mod_two_pi(-np.angle(data[0]))
        for k in range(1, len(phis)):
            phis[k] = _angle_mod_two_pi(-np.angle(data[k]) + np.sum(phis[:k]))

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

    if complex_data:
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
                NotImplementedError, f"This function does not accept three-qubit gates."
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
        (:class:`qibo.models.circuit.Circuit`, list): Circuit composed of :class:`qibo.gates.gates.RBS`
        and list of indexes of target qubits per depth.
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


def _generate_rbs_angles(data, nqubits: int, architecture: str):
    """Generate list of angles for RBS gates based on ``architecture``.

    Args:
        data (ndarray, optional): :math:`1`-dimensional array of data to be loaded.
        nqubits (int): number of qubits.
        architecture(str, optional): circuit architecture used for the unary loader.
            If ``diagonal``, uses a ladder-like structure.
            If ``tree``, uses a binary-tree-based structure.
            Defaults to ``tree``.

    Returns:
        list: List of phases for RBS gates.
    """
    if architecture == "diagonal":
        engine = _check_engine(data)
        phases = [
            math.atan2(engine.linalg.norm(data[k + 1 :]), data[k])
            for k in range(len(data) - 2)
        ]
        phases.append(math.atan2(data[-1], data[-2]))

    if architecture == "tree":
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
        *Quantum encoder for fixed Hamming-weight subspaces*
        `arXiv:2405.20408 [quant-ph] <https://arxiv.org/abs/2405.20408>`_.
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
        *Quantum encoder for fixed Hamming-weight subspaces*
        `arXiv:2405.20408 [quant-ph] <https://arxiv.org/abs/2405.20408>`_.
    """
    if len(qubits_in) == 0 and len(qubits_out) == 1:  # pragma: no cover
        # Important for future binary encoder
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
    from qibo.quantum_info.utils import hamming_weight

    if isinstance(last_string, str):
        last_string = np.asarray(list(last_string), dtype=int)

    last_weight = hamming_weight(last_string)
    last_ones = np.argsort(last_string)
    last_zero = last_ones[0]
    last_controls = last_ones[-last_weight:]

    # adding an RZ gate to correct the phase of the last amplitude encoded
    return gates.RZ(last_zero, 2 * phase).controlled_by(*last_controls)
