"""Module with functions that encode classical data into quantum circuits."""

import math
from inspect import signature
from typing import Optional, Union

import numpy as np

from qibo import gates
from qibo.config import log, raise_error
from qibo.gates.gates import _check_engine
from qibo.models.circuit import Circuit


class CompBasisEncoder(Circuit):
    def __init__(
        self,
        basis_element: Union[int, str, list, tuple],
        nqubits: Optional[int] = None,
        **kwargs,
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
        if isinstance(basis_element, int):
            basis_element = f"{basis_element:0{nqubits}b}"

        if isinstance(basis_element, (str, tuple)):
            basis_element = list(basis_element)

        basis_element = list(map(int, basis_element))

        super().__init__(nqubits, **kwargs)
        for qubit, elem in enumerate(basis_element):
            if elem == 1:
                super().add(gates.X(qubit))


class PhaseEncoder(Circuit):
    def __init__(self, data, rotation: str = "RY", **kwargs):
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

        super().__init__(nqubits, **kwargs)
        super().add(gate(qubit, 0.0) for qubit in range(nqubits))
        super().set_parameters(data)


class UnaryEncoder(Circuit):
    def __init__(self, data, architecture: str = "tree", **kwargs):
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

        super().__init__(nqubits, **kwargs)
        super().add(gates.X(nqubits - 1))
        self._generate_rbs_pairs(nqubits, architecture=architecture, **kwargs)

        # calculating phases and setting circuit parameters
        phases = self._generate_rbs_angles(data, nqubits, architecture)
        super().set_parameters(phases)

    def _generate_rbs_pairs(self, nqubits: int, architecture: str, **kwargs):
        """Generating list of indexes representing the RBS connections

        Creates circuit with all RBS gates initialised with :math:`0.0` phase.

        Args:
            nqubits (int): number of qubits.
            architecture(str, optional): circuit architecture used for the unary loader.
                If ``diagonal``, uses a ladder-like structure.
                If ``tree``, uses a binary-tree-based structure.
                Defaults to ``tree``.
            kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
                For details, see the documentation of :class:`qibo.models.circuit.Circuit`.
        """

        if architecture == "diagonal":
            self._pairs_rbs = np.arange(nqubits)
            self._pairs_rbs = [
                [pair] for pair in zip(self._pairs_rbs[:-1], self._pairs_rbs[1:])
            ]

        if architecture == "tree":
            self._pairs_rbs = [[(0, int(nqubits / 2))]]
            indexes = list(self._pairs_rbs[0][0])
            for depth in range(2, int(math.log2(nqubits)) + 1):
                pairs_rbs_per_depth = [
                    [(index, index + int(nqubits / 2**depth)) for index in indexes]
                ]
                self._pairs_rbs += pairs_rbs_per_depth
                indexes = list(np.array(pairs_rbs_per_depth).flatten())

        self._pairs_rbs = [
            [(nqubits - 1 - a, nqubits - 1 - b) for a, b in row]
            for row in self._pairs_rbs
        ]

        super().add(
            gates.RBS(*pair, 0.0, trainable=True)
            for row in self._pairs_rbs
            for pair in row
        )

    @staticmethod
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
                r_array[j - 1] = math.sqrt(
                    r_array[2 * j] ** 2 + r_array[2 * j - 1] ** 2
                )
                phases[j - 1] = math.acos(r_array[2 * j - 1] / r_array[j - 1])

        return phases


class UnaryEncoderRandomGaussian(UnaryEncoder):
    def __init__(self, nqubits: int, architecture: str = "tree", seed=None, **kwargs):
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
            raise_error(
                ValueError, f"nqubits must be a power of 2, but it is {nqubits}."
            )

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
            np.random.default_rng(seed)
            if seed is None or isinstance(seed, int)
            else seed
        )

        sampler = _ProbabilityDistributionGaussianLoader(
            a=0, b=2 * math.pi, seed=local_state
        )

        Circuit.__init__(self, nqubits, **kwargs)
        super().add(gates.X(nqubits - 1))
        self._generate_rbs_pairs(nqubits, architecture, **kwargs)

        phases = []
        for depth, row in enumerate(self._pairs_rbs, 1):
            phases.extend(sampler.rvs(depth=depth, size=len(row)))

        super().set_parameters(phases)


class EntanglingLayer(Circuit):
    def __init__(
        self,
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

        super().__init__(nqubits, **kwargs)

        if architecture == "diagonal":
            super().add(
                self._parametrized_two_qubit_gate(gate, qubit, qubit + 1, parameters)
                for qubit in range(nqubits - 1)
            )
        elif architecture == "even-layer":
            super().add(
                self._parametrized_two_qubit_gate(gate, qubit, qubit + 1, parameters)
                for qubit in range(0, nqubits - 1, 2)
            )
        elif architecture == "odd-layer":
            super().add(
                self._parametrized_two_qubit_gate(gate, qubit, qubit + 1, parameters)
                for qubit in range(1, nqubits - 1, 2)
            )
        else:
            super().add(
                self._parametrized_two_qubit_gate(gate, qubit, qubit + 1, parameters)
                for qubit in range(0, nqubits - 1, 2)
            )
            super().add(
                self._parametrized_two_qubit_gate(gate, qubit, qubit + 1, parameters)
                for qubit in range(1, nqubits - 1, 2)
            )

        if closed_boundary:
            super().add(
                self._parametrized_two_qubit_gate(gate, nqubits - 1, 0, parameters)
            )

    def _parametrized_two_qubit_gate(self, gate, q0, q1, params=None):
        """Returns two-qubit gate initialized with or without phases."""
        if params is not None:
            return gate(q0, q1, *params)

        return gate(q0, q1)


class GHZState(Circuit):
    def __init__(self, nqubits: int, **kwargs):
        """Generates an :math:`n`-qubit Greenberger-Horne-Zeilinger (GHZ) state that takes the form

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

        super().__init__(nqubits, **kwargs)
        super().add(gates.H(0))
        super().add(gates.CNOT(qubit, qubit + 1) for qubit in range(nqubits - 1))


def _log_message(old_name: str, new_name: str):
    log.warning(
        f"Function {old_name} is deprecated and will be removed on release 0.2.15. "
        + "For the new implementation of the same functionality, "
        f"please see `qibo.models.encodings.{new_name}`."
    )


def comp_basis_encoder(
    basis_element: Union[int, str, list, tuple], nqubits: Optional[int] = None, **kwargs
):
    _log_message("comp_basis_encoder", CompBasisEncoder.__name__)
    return CompBasisEncoder(basis_element, nqubits, **kwargs)


def phase_encoder(data, rotation: str = "RY", **kwargs):
    _log_message("phase_encoder", PhaseEncoder.__name__)
    return PhaseEncoder(data, rotation, **kwargs)


def unary_encoder(data, architecture: str = "tree", **kwargs):
    _log_message("unary_encoder", UnaryEncoder.__name__)
    return UnaryEncoder(data, architecture, **kwargs)


def unary_encoder_random_gaussian(
    nqubits: int, architecture: str = "tree", seed=None, **kwargs
):
    _log_message("unary_encoder_random_gaussian", UnaryEncoderRandomGaussian.__name__)
    return UnaryEncoderRandomGaussian(nqubits, architecture, seed, **kwargs)


def entangling_layer(
    nqubits: int,
    architecture: str = "diagonal",
    entangling_gate: Union[str, gates.Gate] = "CNOT",
    closed_boundary: bool = False,
    **kwargs,
):
    _log_message("entangling_layer", EntanglingLayer.__name__)
    return EntanglingLayer(
        nqubits,
        architecture,
        entangling_gate,
        closed_boundary,
        **kwargs,
    )


def ghz_state(nqubits: int, **kwargs):
    _log_message("ghz_state", GHZState.__name__)
    return GHZState(nqubits, **kwargs)
