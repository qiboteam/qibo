"""Module with functions that encode classical data into quantum circuits."""

import math
from inspect import signature
from typing import Optional, Union

import numpy as np

from qibo import gates
from qibo.config import raise_error
from qibo.gates.gates import _check_engine
from qibo.models.circuit import Circuit


def comp_basis_encoder(
    basis_element: Union[int, str, list, tuple], nqubits: Optional[int] = None, **kwargs
):
    """Creates circuit that performs encoding of bitstrings into computational basis states.

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
       :class:`qibo.models.circuit.Circuit`: circuit encoding computational basis element.
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
    """Creates circuit that performs the phase encoding of ``data``.

    Args:
        data (ndarray or list): :math:`1`-dimensional array of phases to be loaded.
        rotation (str, optional): If ``"RX"``, uses :class:`qibo.gates.gates.RX` as rotation.
            If ``"RY"``, uses :class:`qibo.gates.gates.RY` as rotation.
            If ``"RZ"``, uses :class:`qibo.gates.gates.RZ` as rotation.
            Defaults to ``"RY"``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: circuit that loads ``data`` in phase encoding.
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


def unary_encoder(data, architecture: str = "tree", **kwargs):
    """Creates circuit that performs the (deterministic) unary encoding of ``data``.

    Args:
        data (ndarray): :math:`1`-dimensional array of data to be loaded.
        architecture(str, optional): circuit architecture used for the unary loader.
            If ``diagonal``, uses a ladder-like structure.
            If ``tree``, uses a binary-tree-based structure.
            Defaults to ``tree``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: circuit that loads ``data`` in unary representation.
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
    """Creates a circuit that performs the unary encoding of a random Gaussian state.

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
        :class:`qibo.models.circuit.Circuit`: circuit that loads a random Gaussian array in unary representation.

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


def entangling_layer(
    nqubits: int,
    architecture: str = "diagonal",
    entangling_gate: Union[str, gates.Gate] = "CNOT",
    closed_boundary: bool = False,
    **kwargs,
):
    """Creates a layer of two-qubit, entangling gates.

    If the chosen gate is a parametrized gate, all phases are set to :math:`0.0`.

    Args:
        nqubits (int): Total number of qubits in the circuit.
        architecture (str, optional): Architecture of the entangling layer.
            Options are ``diagonal``, ``shifted``, ``even-layer``, and ``odd-layer``.
            Defaults to ``"diagonal"``.
        entangling_gate (str or :class:`qibo.gates.Gate`, optional): Two-qubit gate to be used
            in the entangling layer. If ``entangling_gate`` is a parametrized gate,
            all phases are initialized as :math:`0.0`. Defaults to  ``"CNOT"``.
        closed_boundary (bool, optional): If ``True`` adds a closed-boundary condition
            to the entangling layer. Defaults to ``False``.
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

    if architecture not in ["diagonal", "shifted", "even-layer", "odd-layer"]:
        raise_error(
            NotImplementedError,
            f"``architecture`` {architecture} not found.",
        )

    if not isinstance(closed_boundary, bool):
        raise_error(
            TypeError,
            f"closed_boundary must be type bool, but it is type {type(closed_boundary)}.",
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

    # Finds the number of correct number of parameters to initialize the gate class.
    parameters = list(signature(gate).parameters)

    if "q2" in parameters:
        raise_error(
            NotImplementedError, f"This function does not accept three-qubit gates."
        )

    # If gate is parametrized, sets all angles to 0.0
    parameters = (0.0,) * (len(parameters) - 3) if len(parameters) > 2 else None

    circuit = Circuit(nqubits, **kwargs)

    if architecture == "diagonal":
        circuit.add(
            _parametrized_two_qubit_gate(gate, qubit, qubit + 1, parameters)
            for qubit in range(nqubits - 1)
        )
    elif architecture == "even-layer":
        circuit.add(
            _parametrized_two_qubit_gate(gate, qubit, qubit + 1, parameters)
            for qubit in range(0, nqubits - 1, 2)
        )
    elif architecture == "odd-layer":
        circuit.add(
            _parametrized_two_qubit_gate(gate, qubit, qubit + 1, parameters)
            for qubit in range(1, nqubits - 1, 2)
        )
    else:
        circuit.add(
            _parametrized_two_qubit_gate(gate, qubit, qubit + 1, parameters)
            for qubit in range(0, nqubits - 1, 2)
        )
        circuit.add(
            _parametrized_two_qubit_gate(gate, qubit, qubit + 1, parameters)
            for qubit in range(1, nqubits - 1, 2)
        )

    if closed_boundary:
        circuit.add(_parametrized_two_qubit_gate(gate, nqubits - 1, 0, parameters))

    return circuit


def _generate_rbs_pairs(nqubits: int, architecture: str, **kwargs):
    """Generating list of indexes representing the RBS connections

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
        (:class:`qibo.models.circuit.Circuit`, list): circuit composed of :class:`qibo.gates.gates.RBS`
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
    """Generating list of angles for RBS gates based on ``architecture``.

    Args:
        data (ndarray, optional): :math:`1`-dimensional array of data to be loaded.
        nqubits (int): number of qubits.
        architecture(str, optional): circuit architecture used for the unary loader.
            If ``diagonal``, uses a ladder-like structure.
            If ``tree``, uses a binary-tree-based structure.
            Defaults to ``tree``.

    Returns:
        list: list of phases for RBS gates.
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
    """Returns two-qubit gate initialized with or without phases."""
    if params is not None:
        return gate(q0, q1, *params)

    return gate(q0, q1)
