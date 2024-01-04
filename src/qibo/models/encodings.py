"""Module with functions that encode classical data into quantum circuits."""

import math

import numpy as np
from scipy.stats import rv_continuous

from qibo import gates
from qibo.config import raise_error
from qibo.models.circuit import Circuit


def unary_encoder(data, architecture: str = "tree"):
    """Creates circuit that performs the unary encoding of ``data``.

    Given a classical ``data`` array :math:`\\mathbf{x} \\in \\mathbb{R}^{d}` such that

    .. math::
        \\mathbf{x} = (x_{1}, x_{2}, \\dots, x_{d}) \\, ,

    this function generate the circuit that prepares the following quantum state
    :math:`\\ket{\\psi} \\in \\mathcal{H}`:

    .. math::
        \\ket{\\psi} = \\frac{1}{\\|\\mathbf{x}\\|_{\\textup{HS}}} \\,
            \\sum_{k=1}^{d} \\, x_{k} \\, \\ket{k} \\, ,

    with :math:`\\mathcal{H} \\cong \\mathbb{C}^{d}` being a :math:`d`-qubit Hilbert space,
    and :math:`\\|\\cdot\\|_{\\textup{HS}}` being the Hilbert-Schmidt norm.
    Here, :math:`\\ket{k}` is a unary representation of the number :math:`1` through
    :math:`d`.

    Args:
        data (ndarray, optional): :math:`1`-dimensional array of data to be loaded.
        architecture(str, optional): circuit architecture used for the unary loader.
            If ``diagonal``, uses a ladder-like structure.
            If ``tree``, uses a binary-tree-based structure.
            Defaults to ``tree``.

    Returns:
        :class:`qibo.models.circuit.Circuit`: circuit that loads ``data`` in unary representation.

    References:
        1. S. Johri *et al.*, *Nearest Centroid Classiﬁcation on a Trapped Ion Quantum Computer*.
        `arXiv:2012.04145v2 [quant-ph] <https://arxiv.org/abs/2012.04145>`_.
    """
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

    circuit = Circuit(nqubits)
    circuit.add(gates.X(nqubits - 1))
    circuit_rbs, _ = _generate_rbs_pairs(nqubits, architecture=architecture)
    circuit += circuit_rbs

    # calculating phases and setting circuit parameters
    phases = _generate_rbs_angles(data, nqubits, architecture)
    circuit.set_parameters(phases)

    return circuit


def unary_encoder_random_gaussian(nqubits: int, architecture: str = "tree", seed=None):
    """Creates a circuit that performs the unary encoding of a random Gaussian state.

    Given :math:`d` qubits, encodes the quantum state
    :math:`\\ket{\\psi} \\in \\mathcal{H}` such that


    .. math::
        \\ket{\\psi} = \\frac{1}{\\|\\mathbf{x}\\|_{\\textup{HS}}} \\,
            \\sum_{k=1}^{d} \\, x_{k} \\, \\ket{k}

    where :math:`x_{k}` are independent Gaussian random variables,
    :math:`\\mathcal{H} \\cong \\mathbb{C}^{d}` is a :math:`d`-qubit Hilbert space,
    and :math:`\\|\\cdot\\|_{\\textup{HS}}` being the Hilbert-Schmidt norm.
    Here, :math:`\\ket{k}` is a unary representation of the number :math:`1` through
    :math:`d`.

    At depth :math:`h`, the angles :math:`\\theta_{k} \\in [0, 2\\pi]` of the the
    gates :math:`RBS(\\theta_{k})` are sampled from the following probability density function:

    .. math::
        p_{h}(\\theta) = \\frac{1}{2} \\, \\frac{\\Gamma(2^{h-1})}{\\Gamma^{2}(2^{h-2})}
            \\abs{\\sin(\\theta) \\, \\cos(\\theta)}^{2^{h-1} - 1} \\, ,

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

    local_state = (
        np.random.default_rng(seed) if seed is None or isinstance(seed, int) else seed
    )

    sampler = _ProbabilityDistributionGaussianLoader(
        a=0, b=2 * math.pi, seed=local_state
    )

    circuit = Circuit(nqubits)
    circuit.add(gates.X(nqubits - 1))
    circuit_rbs, pairs_rbs = _generate_rbs_pairs(nqubits, architecture)
    circuit += circuit_rbs

    phases = []
    for depth, row in enumerate(pairs_rbs, 1):
        phases.extend(sampler.rvs(depth=depth, size=len(row)))

    circuit.set_parameters(phases)

    return circuit


def _generate_rbs_pairs(nqubits: int, architecture: str):
    """Generating list of indexes representing the RBS connections

    Creates circuit with all RBS initialised with 0.0 phase.

    Args:
        nqubits (int): number of qubits.
        architecture(str, optional): circuit architecture used for the unary loader.
            If ``diagonal``, uses a ladder-like structure.
            If ``tree``, uses a binary-tree-based structure.
            Defaults to ``tree``.

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

    circuit = Circuit(nqubits)
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
        phases = [
            math.atan2(np.linalg.norm(data[k + 1 :]), data[k])
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


class _ProbabilityDistributionGaussianLoader(rv_continuous):
    """Probability density function for sampling phases of
    the RBS gates as a function of circuit depth."""

    def _pdf(self, theta: float, depth: int):
        amplitude = 2 * math.gamma(2 ** (depth - 1)) / math.gamma(2 ** (depth - 2)) ** 2

        probability = abs(math.sin(theta) * math.cos(theta)) ** (2 ** (depth - 1) - 1)

        return amplitude * probability / 4
