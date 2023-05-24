"""Define quantum channels."""
import warnings
from itertools import product
from math import exp, sqrt
from typing import Tuple

import numpy as np

from qibo.config import PRECISION_TOL, raise_error
from qibo.gates.abstract import Gate
from qibo.gates.gates import I, Unitary, X, Y, Z
from qibo.gates.special import FusedGate


class Channel(Gate):
    """Abstract class for channels."""

    def __init__(self):
        super().__init__()
        self.coefficients = tuple()
        self.gates = tuple()

    def controlled_by(self, *q):
        raise_error(ValueError, f"Noise channel cannot be controlled on qubits {q}.")

    def on_qubits(self, qubit_map):  # pragma: no cover
        # future TODO
        raise_error(
            NotImplementedError,
            "`on_qubits` method is not available for the `Channel` gate.",
        )

    def apply(self, backend, state, nqubits):  # pragma: no cover
        raise_error(
            NotImplementedError,
            f"{self.__class__.__name__} cannot be applied to state vector.",
        )

    def apply_density_matrix(self, backend, state, nqubits):
        return backend.apply_channel_density_matrix(self, state, nqubits)

    def to_choi(self, nqubits: int = None, order: str = "row", backend=None):
        """Returns the Choi representation :math:`\\mathcal{E}`
        of the Kraus channel :math:`\\{K_{\\alpha}\\}_{\\alpha}`.

        .. math::
            \\mathcal{E} = \\sum_{\\alpha} \\, |K_{\\alpha}\\rangle\\rangle
                \\langle\\langle K_{\\alpha}|

        Args:
            nqubits (int, optional): total number of qubits to be considered
                in a channel. Must be equal or greater than ``target_qubits``.
                If ``None``, defaults to the number of target qubits in the
                channel. Default is ``None``.
            order (str, optional): if ``"row"``, vectorization of
                Kraus operators is performed row-wise. If ``"column"``,
                vectorization is done column-wise. If ``"system"``,
                vectorization is done block-wise. Defaut is ``"row"``.
            backend (``qibo.backends.abstract.Backend``, optional):
                backend to be used in the execution. If ``None``,
                it uses ``GlobalBackend()``. Defaults to ``None``.

        Returns:
            Choi representation of the channel.
        """

        if nqubits is not None and nqubits < 1 + max(self.target_qubits):
            raise_error(
                ValueError,
                f"nqubits={nqubits}, but channel acts on qubit with index {max(self.target_qubits)}.",
            )

        from qibo.quantum_info.superoperator_transformations import vectorization

        if backend is None:  # pragma: no cover
            from qibo.backends import GlobalBackend

            backend = GlobalBackend()

        self.nqubits = 1 + max(self.target_qubits) if nqubits is None else nqubits

        if type(self) not in [KrausChannel, ReadoutErrorChannel]:
            p0 = 1 - sum(self.coefficients)
            if p0 > PRECISION_TOL:
                self.coefficients += (p0,)
                self.gates += (I(*self.target_qubits),)

        super_op = np.zeros((4**self.nqubits, 4**self.nqubits), dtype=complex)
        super_op = backend.cast(super_op, dtype=super_op.dtype)
        for coeff, gate in zip(self.coefficients, self.gates):
            kraus_op = FusedGate(*range(self.nqubits))
            kraus_op.append(gate)
            kraus_op = kraus_op.asmatrix(backend)
            kraus_op = vectorization(kraus_op, order=order, backend=backend)
            super_op += coeff * np.outer(kraus_op, np.conj(kraus_op))
            del kraus_op

        return super_op

    def to_liouville(self, nqubits: int = None, order: str = "row", backend=None):
        """Returns the Liouville representation of the channel.

        Args:
            nqubits (int, optional): total number of qubits to be considered
                in a channel. Must be equal or greater than ``target_qubits``.
                If ``None``, defaults to the number of target qubits in the
                channel. Default is ``None``.
            order (str, optional): If ``"row"``, vectorization of
                Kraus operators is performed row-wise. If ``"column"``,
                vectorization is done column-wise. If ``"system"``,
                it raises ``NotImplementedError``. Defaut is ``"row"``.
            backend (``qibo.backends.abstract.Backend``, optional):
                backend to be used in the execution. If ``None``,
                it uses ``GlobalBackend()``. Defaults to ``None``.

        Returns:
            Liouville representation of the channel.
        """

        from qibo.quantum_info.superoperator_transformations import choi_to_liouville

        if backend is None:  # pragma: no cover
            from qibo.backends import GlobalBackend

            backend = GlobalBackend()

        super_op = self.to_choi(nqubits=nqubits, order=order, backend=backend)
        super_op = choi_to_liouville(super_op, order=order, backend=backend)

        return super_op

    def to_pauli_liouville(
        self,
        nqubits: int = None,
        normalize: bool = False,
        pauli_order: str = "IXYZ",
        backend=None,
    ):
        """Returns the Liouville representation of the channel
        in the Pauli basis.

        Args:
            nqubits (int, optional): total number of qubits to be considered
                in a channel. Must be equal or greater than ``target_qubits``.
                If ``None``, defaults to the number of target qubits in the
                channel. Default is ``None``.
            normalize (bool, optional): If ``True``, normalized basis is returned.
                Defaults to False.
            pauli_order (str, optional): corresponds to the order of 4 single-qubit
                Pauli elements in the basis. Default is "IXYZ".
            backend (``qibo.backends.abstract.Backend``, optional): backend
                to be used in the execution. If ``None``, it uses
                ``GlobalBackend()``. Defaults to ``None``.

        Returns:
            Pauli-Liouville representation of the channel.
        """

        from qibo.quantum_info.basis import comp_basis_to_pauli

        if backend is None:  # pragma: no cover
            from qibo.backends import GlobalBackend

            backend = GlobalBackend()

        super_op = self.to_liouville(nqubits=nqubits, backend=backend)

        # unitary that transforms from comp basis to pauli basis
        unitary = comp_basis_to_pauli(
            self.nqubits, normalize, pauli_order=pauli_order, backend=backend
        )
        unitary = backend.cast(unitary, dtype=unitary.dtype)

        super_op = unitary @ super_op @ np.transpose(np.conj(unitary))

        return super_op


class KrausChannel(Channel):
    """General channel defined by arbitrary Kraus operators.

    Implements the following transformation:

    .. math::
        \\mathcal{E}(\\rho ) = \\sum _k A_k \\rho A_k^\\dagger

    where A are arbitrary Kraus operators given by the user. Note that Kraus
    operators set should be trace preserving, however this is not checked.
    Simulation of this gate requires the use of density matrices.
    For more information on channels and Kraus operators please check
    `J. Preskill's notes <http://theory.caltech.edu/~preskill/ph219/chap3_15.pdf>`_.

    Args:
        qubits (int or list or tuple or None): Qubits that the Kraus operators act on.
            Type ``int`` and ``tuple`` will be considered as the same qubit ids for
            all operators. A ``list`` should contain tuples of qubits corresponding
            to each operator. Can be an empty ``list`` if ``ops`` are of type
            ``qibo.gates.Gate``.
        ops (list): List of Kraus operators ``Ak`` as matrices of type
            ``np.ndarray | tf.Tensor`` or gates ``qibo.gates.Gate``.

    Example:
        .. testcode::

            import numpy as np
            from qibo.models import Circuit
            from qibo import gates
            # initialize circuit with 3 qubits
            c = Circuit(3, density_matrix=True)
            # define a sqrt(0.4) * X gate
            a1 = np.sqrt(0.4) * np.array([[0, 1], [1, 0]])
            # define a sqrt(0.6) * CNOT gate
            a2 = np.sqrt(0.6) * np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                          [0, 0, 0, 1], [0, 0, 1, 0]])
            # define the channel rho -> 0.4 X{1} rho X{1} + 0.6 CNOT{0, 2} rho CNOT{0, 2}
            channel1 = gates.KrausChannel([(1,), (0, 2)], [a1, a2])
            # add channel to the circuit
            c.add(channel1)

            # define the same channel using qibo.gates.Unitary
            a1 = gates.Unitary(a1, 1)
            a2 = gates.Unitary(a2, 0, 2)
            channel2 = gates.KrausChannel([], [a1, a2])
            # add channel to the circuit
            c.add(channel2)

            # define the channel rho -> 0.4 X{0} rho X{0} + 0.6 CNOT{1, 2} rho CNOT{1, 2}
            channel3 = gates.KrausChannel([(0,), (1, 2)], [a1, a2])
            # add channel to the circuit
            c.add(channel3)
    """

    def __init__(self, *ops):
        super().__init__()
        self.name = "KrausChannel"
        self.draw_label = "K"

        if len(ops) == 1:
            warnings.warn(
                f"{self.__class__.__name__} initialisation has changed. "
                + "Please check the latest documentation. Previous initialisation "
                + "will be removed in Release 1.15.",
                DeprecationWarning,
                stacklevel=2,
            )

            qubits = [row[0] for row in ops[0]]
            ops = [row[1] for row in ops[0]]
        else:
            qubits = ops[0]
            ops = ops[1]

        # Check qubits type
        if isinstance(qubits, int) is True:
            qubits = [(qubits,)] * len(ops)
        elif isinstance(qubits, tuple) is True:
            qubits = [qubits] * len(ops)
        elif isinstance(qubits, list) is False:
            raise_error(
                TypeError,
                "``qubits`` must be of type int or tuple or int. "
                + f"Got {type(qubits)} instead",
            )
        elif not all(isinstance(q, (tuple)) for q in qubits):
            raise_error(TypeError, "All elements of ``qubits`` list must be tuples.")

        if isinstance(ops[0], Gate) is True:
            if qubits:
                ops = [
                    ops[k].on_qubits(
                        {
                            ops[k].qubits[i]: qubits[k][i]
                            for i in range(len(ops[k].qubits))
                        }
                    )
                    for k in range(len(ops))
                ]
            self.gates = tuple(ops)
            self.target_qubits = tuple(
                sorted({q for gate in ops for q in gate.target_qubits})
            )
        elif len(qubits) != len(ops):
            raise_error(
                ValueError,
                f"``qubits`` list has length {len(qubits)} while "
                + f"{len(ops)} operators were given.",
            )
        else:
            gates, qubitset = [], set()

            for qubit_tuple, matrix in zip(qubits, ops):
                rank = 2 ** len(qubit_tuple)
                shape = tuple(matrix.shape)
                if shape != (rank, rank):
                    raise_error(
                        ValueError,
                        f"Invalid Kraus operator shape {shape} for "
                        + f"acting on {len(qubit_tuple)} qubits.",
                    )
                qubitset.update(qubit_tuple)
                gates.append(Unitary(matrix, *list(qubit_tuple)))
            self.gates = tuple(gates)
            self.target_qubits = tuple(sorted(qubitset))
        self.init_args = [self.gates]
        self.coefficients = len(self.gates) * (1,)
        self.coefficient_sum = 1


class UnitaryChannel(KrausChannel):
    """Channel that is a probabilistic sum of unitary operations.

    Implements the following transformation:

    .. math::
        \\mathcal{E}(\\rho ) = \\left (1 - \\sum _k p_k \\right )\\rho +
                                \\sum _k p_k U_k \\rho U_k^\\dagger

    where U are arbitrary unitary operators and p are floats between 0 and 1.
    Note that unlike :class:`qibo.gates.KrausChannel` which requires
    density matrices, it is possible to simulate the unitary channel using
    state vectors and probabilistic sampling. For more information on this
    approach we refer to :ref:`Using repeated execution <repeatedexec-example>`.

    Args:
        qubits (int or list or tuple or None): Qubits that the unitary operators
            act on. Types ``int`` and ``tuple`` will be considered as the same
            qubit(s) for all unitaries. A ``list`` should contain tuples of
            qubits corresponding to each operator. Can be [] if ``ops`` are of
            type ``qibo.gates.Gate``.
        ops (list): List of  operators as pairs ``(pk, Uk)`` where
            ``pk`` is float probability corresponding to a unitary ``Uk``
            of type ``np.ndarray``/``tf.Tensor`` or gates ``qibo.gates.Gate``.
    """

    def __init__(self, qubits, ops):
        if isinstance(qubits, list) is True and any(
            isinstance(q, float) is True for q in qubits
        ):
            warnings.warn(
                f"{self.__class__.__name__} initialisation has changed. "
                + "Please check the latest documentation. Previous initialisation "
                + "will be removed in Release 1.15.",
                DeprecationWarning,
                stacklevel=2,
            )
            probabilities = np.copy(qubits)
            qubits = [row[0] for row in ops]

            ops = [row[1] for row in ops]
            ops = list(zip(probabilities, ops))

        if not all(isinstance(pair, (tuple)) for pair in ops):
            raise_error(TypeError, "``ops`` must be a list of tuples ``(pk, Uk)``.")

        probabilities = [pair[0] for pair in ops]
        ops = [pair[1] for pair in ops]
        if any((p < 0 or p > 1) for p in probabilities):
            raise_error(
                ValueError,
                "Probabilities should be between 0 and 1.",
            )
        super().__init__(qubits, ops)
        self.name = "UnitaryChannel"
        self.draw_label = "U"
        self.coefficients = tuple(probabilities)
        self.coefficient_sum = sum(probabilities)
        if self.coefficient_sum > 1 + PRECISION_TOL or self.coefficient_sum <= 0:
            raise_error(
                ValueError,
                "UnitaryChannel probability sum should be "
                + f"between 0 and 1 but is {self.coefficient_sum}.",
            )

        self.init_args = [probabilities, self.gates]

    def apply(self, backend, state, nqubits):
        return backend.apply_channel(self, state, nqubits)


class PauliNoiseChannel(UnitaryChannel):
    """Multi-qubit noise channel that applies Pauli operators with given probabilities.

    Implements the following transformation:

    .. math::
        \\mathcal{E}(\\rho ) = \\left (1 - \\sum _{k} p_{k} \\right ) \\, \\rho +
                                \\sum_{k} \\, p_{k} \\, P_{k} \\, \\rho \\, P_{k}


    where :math:`P_{k}` is the :math:`k`-th Pauli ``string`` and :math:`p_{k}` is
    the probability associated to :math:`P_{k}`.

    Example:
        .. testcode::

            from itertools import product

            import numpy as np

            from qibo.gates.channels import PauliNoiseChannel

            qubits = (0, 2)
            nqubits = len(qubits)

            # excluding the Identity operator
            paulis = list(product(["I", "X"], repeat=nqubits))[1:]
            # this next line is optional
            paulis = [''.join(pauli) for pauli in paulis]

            probabilities = np.random.rand(len(paulis) + 1)
            probabilities /= np.sum(probabilities)
            #Excluding probability of Identity operator
            probabilities = probabilities[1:]

            channel = PauliNoiseChannel(
                qubits, list(zip(paulis, probabilities))
            )

    This channel can be simulated using either density matrices or state vectors
    and sampling with repeated execution.
    See :ref:`How to perform noisy simulation? <noisy-example>` for more
    information.

    Args:
        qubits (int or list or tuple): Qubits that the noise acts on.
        operators (list): list of operators as pairs :math:`(P_{k}, p_{k})`.
    """

    def __init__(self, qubits: Tuple[int, list, tuple], operators: list):
        if isinstance(qubits, int) is True:
            qubits = (qubits,)

        probabilities, paulis = [], []
        for pauli, probability in operators:
            probabilities.append(probability)
            paulis.append(pauli)

        single_paulis = {"I": I, "X": X, "Y": Y, "Z": Z}

        gates = []
        for pauli in paulis:
            fgate = FusedGate(*qubits)
            for q, p in zip(qubits, pauli):
                fgate.append(single_paulis[p](q))
            gates.append(fgate)
        self.gates = tuple(gates)
        self.coefficients = tuple(probabilities)

        super().__init__([], list(zip(probabilities, gates)))
        self.name = "PauliNoiseChannel"
        self.draw_label = "PN"
        self.init_args = qubits
        self.init_kwargs = dict(operators)


class DepolarizingChannel(PauliNoiseChannel):
    """:math:`n`-qubit Depolarizing quantum error channel,

    .. math::
        \\mathcal{E}(\\rho ) = (1 - \\lambda) \\rho +
            \\lambda \\text{Tr}_q[\\rho]\\otimes \\frac{I}{2^n}

    where :math:`\\lambda` is the depolarizing error parameter
    and :math:`0 \\le \\lambda \\le 4^n / (4^n - 1)`.

    * If :math:`\\lambda = 1` this is a completely depolarizing channel
      :math:`E(\\rho) = I / 2^n`
    * If :math:`\\lambda = 4^n / (4^n - 1)` this is a uniform Pauli
      error channel: :math:`E(\\rho) = \\sum_j P_j \\rho P_j / (4^n - 1)` for
      all :math:`P_j \\neq I`.

    Args:
        qubits (tuple): Qubit ids that the noise acts on.
        lam (float): Depolarizing error parameter.
    """

    def __init__(self, qubits, lam: float = 0):
        if isinstance(qubits, int) is True:
            qubits = (qubits,)

        num_qubits = len(qubits)
        num_terms = 4**num_qubits
        max_param = num_terms / (num_terms - 1)
        if lam < 0 or lam > max_param:
            raise_error(
                ValueError,
                f"Depolarizing parameter must be in between 0 and {max_param}.",
            )

        pauli_noise_params = list(product(["I", "X", "Y", "Z"], repeat=num_qubits))[1::]
        pauli_noise_params = zip(
            pauli_noise_params, [lam / num_terms] * (num_terms - 1)
        )
        super().__init__(qubits, pauli_noise_params)

        self.name = "DepolarizingChannel"
        self.draw_label = "D"
        self.target_qubits = qubits

        self.init_args = [qubits]
        self.init_kwargs = {"lam": lam}

    def apply_density_matrix(self, backend, state, nqubits):
        return backend.depolarizing_error_density_matrix(self, state, nqubits)


class ThermalRelaxationChannel(KrausChannel):
    """Single-qubit thermal relaxation error channel.

    Implements the following transformation:

    If :math:`T_1 \\geq T_2`:

    .. math::
        \\mathcal{E} (\\rho ) = (1 - p_z - p0 - p_1)\\rho + p_zZ\\rho Z
            + \\mathrm{Tr}_q[\\rho] \\otimes (p0|0\\rangle \\langle 0|
            + p_1|1\\rangle \\langle 1|)


    while if :math:`T_1 < T_2`:

    .. math::
        \\mathcal{E}(\\rho ) = \\mathrm{Tr}_\\mathcal{X}
            \\left[\\Lambda_{\\mathcal{X}\\mathcal{Y}} (\\rho_\\mathcal{X}^T
            \\otimes I_{\\mathcal{Y}}) \\right]

    with

    .. math::
        \\Lambda = \\begin{pmatrix}
        1 - p_1 & 0 & 0 & e^{-t / T_2} \\\\
        0 & p_1 & 0 & 0 \\\\
        0 & 0 & p0 & 0 \\\\
        e^{-t / T_2} & 0 & 0 & 1 - p0
        \\end{pmatrix}

    where :math:`p0 = (1 - e^{-t / T_1})(1 - \\eta )`,
    :math:`p_1 = (1 - e^{-t / T_1})\\eta`, and
    :math:`p_z = (e^{-t / T_1} - e^{-t / T_2})/2`.
    Here :math:`\\eta` is the ``excited_population``
    and :math:`t` is the ``time``, both controlled by the user.
    This gate is based on `Qiskit's thermal relaxation error channel
    <https://qiskit.org/documentation/stubs/qiskit.providers.aer.noise.thermal_relaxation_error.html#qiskit.providers.aer.noise.thermal_relaxation_error>`_.

    Args:
        qubit (int): Qubit id that the noise channel acts on.
        params (list): list of 3 or 4 parameters
            (t_1, t_2, time, excited_population=0), where
            t_1 (float): T1 relaxation time. Should satisfy ``t_1 > 0``.
            t_2 (float): T2 dephasing time.
            Should satisfy ``t_1 > 0`` and ``t_2 < 2 * t_1``.
            time (float): the gate time for relaxation error.
            excited_population (float): the population of the excited state at
            equilibrium. Default is 0.
    """

    def __init__(self, *args):
        self.name = "ThermalRelaxationChannel"
        # check given parameters
        if len(args) in [4, 5]:
            warnings.warn(
                f"{self.__class__.__name__} initialisation has changed. "
                + "Please check the latest documentation. Previous initialisation "
                + "will be removed in Release 1.15.",
                DeprecationWarning,
                stacklevel=2,
            )
            qubit = args[0]
            params = args[1:]
        else:
            qubit = args[0]
            params = args[1]
        del args

        if len(params) not in [3, 4]:
            raise_error(
                ValueError,
                "``params`` list must have 3 or 4 elements "
                + f"while {len(params)} were given.",
            )

        t_1, t_2, time = params[:3]
        excited_population = params[-1] if len(params) == 4 else 0.0

        if excited_population < 0 or excited_population > 1:
            raise_error(
                ValueError, f"Invalid excited state population {excited_population}."
            )
        if time < 0:
            raise_error(ValueError, f"Invalid gate_time ({time} < 0).")
        if t_1 <= 0:
            raise_error(ValueError, "Invalid t_1 relaxation time parameter: t_1 <= 0.")
        if t_2 <= 0:
            raise_error(ValueError, "Invalid t_2 relaxation time parameter: t_2 <= 0.")
        if t_2 > 2 * t_1:
            raise_error(
                ValueError,
                "Invalid t_2 relaxation time parameter: t_2 > 2 * t_1.",
            )

        # calculate probabilities
        self.t_1, self.t_2 = t_1, t_2
        p_reset = 1 - exp(-time / t_1)
        p_0 = p_reset * (1 - excited_population)
        p_1 = p_reset * excited_population

        if t_1 < t_2:
            e_t2 = exp(-time / t_2)

            operators = [
                sqrt(p_0) * sqrt(p_0) * np.array([[0, 1], [0, 0]]),
                sqrt(p_1) * np.array([[0, 0], [1, 0]]),
            ]

            k_term = sqrt(4 * e_t2**2 + (p_0 - p_1) ** 2)
            kraus_coeff = sqrt(1 - (p_0 + p_1 + k_term) / 2)

            operators.append(
                kraus_coeff
                * np.array(
                    [
                        [(p_0 - p_1 - k_term) / (2 * e_t2), 0],
                        [0, 1],
                    ]
                )
            )

            operators.append(
                kraus_coeff
                * np.array(
                    [
                        [(p_0 - p_1 + k_term) / (2 * e_t2), 0],
                        [0, 1],
                    ]
                )
            )

            super().__init__([(qubit,)] * len(operators), operators)
            self.init_kwargs["e_t2"] = e_t2
        else:
            pz = (exp(-time / t_1) - exp(-time / t_2)) / 2
            operators = (
                sqrt(p_0) * np.array([[1, 0], [0, 0]]),
                sqrt(p_0) * np.array([[0, 1], [0, 0]]),
                sqrt(p_1) * np.array([[0, 0], [1, 0]]),
                sqrt(p_1) * np.array([[0, 0], [0, 1]]),
                sqrt(pz) * np.array([[1, 0], [0, -1]]),
                sqrt(1 - p_0 - p_1 - pz) * np.eye(2),
            )
            super().__init__([(qubit,)] * len(operators), operators)
            self.init_kwargs["pz"] = pz

        self.init_args = [qubit, t_1, t_2, time]
        self.t_1, self.t_2 = t_1, t_2
        self.init_kwargs["excited_population"] = excited_population
        self.init_kwargs["p0"] = p_0
        self.init_kwargs["p1"] = p_1

        self.name = "ThermalRelaxationChannel"
        self.draw_label = "TR"

    def apply_density_matrix(self, backend, state, nqubits):
        qubit = self.target_qubits[0]

        if self.t_1 < self.t_2:
            preset0, preset1, e_t2 = (
                self.init_kwargs["p0"],
                self.init_kwargs["p1"],
                self.init_kwargs["e_t2"],
            )
            matrix = [
                [1 - preset1, 0, 0, preset0],
                [0, e_t2, 0, 0],
                [0, 0, e_t2, 0],
                [preset1, 0, 0, 1 - preset0],
            ]

            qubits = (qubit, qubit + nqubits)
            gate = Unitary(matrix, *qubits)

            return backend.thermal_error_density_matrix(gate, state, nqubits)

        pz = self.init_kwargs["pz"]

        return (
            backend.reset_error_density_matrix(self, state, nqubits)
            - pz * backend.cast(state)
            + pz * backend.apply_gate_density_matrix(Z(0), state, nqubits)
        )


class ReadoutErrorChannel(KrausChannel):
    """Readout error channel implemented as a quantum-to-classical channel.

    Args:
        qubits (int or list or tuple): Qubit ids that the channel acts on.
        probabilities (array): row-stochastic matrix :math:`P` with all
            readout transition probabilities.

            Example:
                For 1 qubit, the transition matrix :math:`P` would be

                .. math::
                    P = \\begin{pmatrix}
                        p(0 \\, | \\, 0) & p(1 \\, | \\, 0) \\\\
                        p(0 \\, | \\, 1) & p(1 \\, | \\, 1)
                    \\end{pmatrix} \\, .
    """

    def __init__(self, qubits: Tuple[int, list, tuple], probabilities):
        if any(sum(row) < 1 - PRECISION_TOL for row in probabilities) or any(
            sum(row) > 1 + PRECISION_TOL for row in probabilities
        ):
            raise_error(ValueError, "all rows of probabilities must sum to 1.")

        if isinstance(qubits, int) is True:
            qubits = (qubits,)

        dim = len(probabilities)
        operators = []
        for j in range(dim):
            for k in range(dim):
                operator = np.zeros((dim, dim))
                operator[j, k] = sqrt(probabilities[k, j])
                operators.append(operator)

        super().__init__([qubits] * len(operators), operators)
        self.name = "ReadoutErrorChannel"
        self.draw_label = "RE"


class ResetChannel(KrausChannel):
    """Single-qubit reset channel.

    Implements the following transformation:

    .. math::
        \\mathcal{E}(\\rho ) = (1 - p0 - p_1) \\rho
        + \\mathrm{Tr}_q[\\rho] \\otimes (p0|0\\rangle \\langle 0|
        + p_1|1\\rangle \\langle 1|),

    Args:
        q (int): Qubit id that the channel acts on.
        probabilities (list or ndarray): list :math:`[p_{0}, p_{1}]`,
            where :math:`p_{0}` and `p_{1}` are the probabilities to
            reset to 0 and 1, respectively.
    """

    def __init__(self, *args):
        if isinstance(args[1], float) is True:
            warnings.warn(
                f"{self.__class__.__name__} initialisation has changed. "
                + "Please check the latest documentation. Previous initialisation "
                + "will be removed in Release 1.15.",
                DeprecationWarning,
                stacklevel=2,
            )
            qubit = args[0]
            probabilities = list(args[1:])
        else:
            qubit = args[0]
            probabilities = args[1]
        del args

        if len(probabilities) != 2:
            raise_error(
                ValueError,
                f"ResetChannel needs 2 probabilities, got {len(probabilities)} instead.",
            )
        p0, p1 = probabilities
        if p0 < 0:
            raise_error(ValueError, "Invalid p0 ({p0} < 0).")
        if p1 < 0:
            raise_error(ValueError, "Invalid p1 ({p1} < 0).")
        if p0 + p1 > 1 + PRECISION_TOL:
            raise_error(ValueError, f"Invalid probabilities (p0 + p1 = {p0 + p1} > 1).")

        operators = [
            sqrt(p0) * np.array([[1, 0], [0, 0]]),
            sqrt(p0) * np.array([[0, 1], [0, 0]]),
            sqrt(p1) * np.array([[0, 0], [1, 0]]),
            sqrt(p1) * np.array([[0, 0], [0, 1]]),
        ]

        if p0 + p1 < 1:
            operators.append(sqrt(np.abs(1 - p0 - p1)) * np.eye(2))

        super().__init__([(qubit,)] * len(operators), operators)
        self.init_kwargs = {"p0": p0, "p1": p1}
        self.name = "ResetChannel"
        self.draw_label = "R"

    def apply_density_matrix(self, backend, state, nqubits):
        return backend.reset_error_density_matrix(self, state, nqubits)
