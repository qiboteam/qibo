import warnings
from itertools import product
from math import exp, sqrt
from typing import Tuple

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
        """"""
        raise_error(ValueError, "Noise channel cannot be controlled on qubits.")

    def on_qubits(self, qubit_map):  # pragma: no cover
        # future TODO
        raise_error(
            NotImplementedError,
            "`on_qubits` method is not available " "for the `Channel` gate.",
        )

    def apply(self, backend, state, nqubits):  # pragma: no cover
        raise_error(
            NotImplementedError, f"{self.name} cannot be applied to state vector."
        )

    def apply_density_matrix(self, backend, state, nqubits):
        return backend.apply_channel_density_matrix(self, state, nqubits)

    def to_choi(self, order: str = "row", backend=None):
        """Returns the Choi representation :math:`\\mathcal{E}`
        of the Kraus channel :math:`\\{K_{\\alpha}\\}_{\\alpha}`.

        .. math::
            \\mathcal{E} = \\sum_{\\alpha} \\, |K_{\\alpha}\\rangle\\rangle \\langle\\langle K_{\\alpha}|

        Args:
            order (str, optional): If ``"row"``, vectorization of
                Kraus operators is performed row-wise. If ``"column"``,
                vectorization is done column-wise. If ``"system"``,
                vectorization is done block-wise. Defaut is ``"row"``.
            backend (``qibo.backends.abstract.Backend``, optional):
                backend to be used in the execution. If ``None``,
                it uses ``GlobalBackend()``. Defaults to ``None``.

        Returns:
            Choi representation of the channel.
        """
        import numpy as np

        from qibo.quantum_info.superoperator_transformations import vectorization

        if backend is None:  # pragma: no cover
            from qibo.backends import GlobalBackend

            backend = GlobalBackend()

        self.nqubits = 1 + max(self.target_qubits)

        if isinstance(self, DepolarizingChannel) is True:
            num_qubits = len(self.target_qubits)
            num_terms = 4**num_qubits
            prob_pauli = self.init_kwargs["lam"] / num_terms
            probs = (num_terms - 1) * [prob_pauli]
            gates = []
            for pauli_list in list(product([I, X, Y, Z], repeat=num_qubits))[1::]:
                fgate = FusedGate(*self.target_qubits)
                for j, pauli in enumerate(pauli_list):
                    fgate.append(pauli(j))
                gates.append(fgate)
            self.gates = tuple(gates)
            self.coefficients = tuple(probs)

        if type(self) not in [KrausChannel, ReadoutErrorChannel]:
            p0 = 1 - sum(self.coefficients)
            if p0 > PRECISION_TOL:
                self.coefficients += (p0,)
                self.gates += (I(*self.target_qubits),)

        super_op = np.zeros((4**self.nqubits, 4**self.nqubits), dtype="complex")
        for coeff, gate in zip(self.coefficients, self.gates):
            kraus_op = FusedGate(*range(self.nqubits))
            kraus_op.append(gate)
            kraus_op = kraus_op.asmatrix(backend)
            kraus_op = vectorization(kraus_op, order=order)
            super_op += coeff * np.outer(kraus_op, np.conj(kraus_op))
            del kraus_op

        super_op = backend.cast(super_op, dtype=super_op.dtype)

        return super_op

    def to_liouville(self, order: str = "row", backend=None):
        """Returns the Liouville representation of the channel.

        Args:
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
        import numpy as np

        from qibo.quantum_info.superoperator_transformations import choi_to_liouville

        if backend is None:  # pragma: no cover
            from qibo.backends import GlobalBackend

            backend = GlobalBackend()

        super_op = self.to_choi(order=order, backend=backend)
        super_op = choi_to_liouville(super_op, order=order)
        super_op = backend.cast(super_op, dtype=super_op.dtype)

        return super_op

    def to_pauli_liouville(self, normalize: bool = False, backend=None):
        """Returns the Liouville representation of the channel
        in the Pauli basis.

        Args:
            normalize (bool, optional): If ``True``, normalized basis is returned.
                Defaults to False.
            backend (``qibo.backends.abstract.Backend``, optional): backend
                to be used in the execution. If ``None``, it uses
                ``GlobalBackend()``. Defaults to ``None``.

        Returns:
            Pauli-Liouville representation of the channel.
        """
        import numpy as np

        from qibo.quantum_info.basis import comp_basis_to_pauli

        if backend is None:  # pragma: no cover
            from qibo.backends import GlobalBackend

            backend = GlobalBackend()

        super_op = self.to_liouville(backend=backend)

        # unitary that transforms from comp basis to pauli basis
        U = comp_basis_to_pauli(self.nqubits, normalize)
        U = backend.cast(U, dtype=U.dtype)

        super_op = U @ super_op @ np.transpose(np.conj(U))
        super_op = backend.cast(super_op, dtype=super_op.dtype)

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
        ops (list): List of Kraus operators as pairs ``(qubits, Ak)`` where
          ``qubits`` refers the qubit ids that ``Ak`` acts on and ``Ak`` is
          the corresponding matrix as a ``np.ndarray`` or ``tf.Tensor``.

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
            channel = gates.KrausChannel([((1,), a1), ((0, 2), a2)])
            # add the channel to the circuit
            c.add(channel)
    """

    def __init__(self, ops):
        super().__init__()
        self.name = "KrausChannel"
        if isinstance(ops[0], Gate):
            self.gates = tuple(ops)
            self.target_qubits = tuple(
                sorted({q for gate in ops for q in gate.target_qubits})
            )
        else:
            gates, qubitset = [], set()
            for qubits, matrix in ops:
                rank = 2 ** len(qubits)
                shape = tuple(matrix.shape)
                if shape != (rank, rank):
                    raise_error(
                        ValueError,
                        f"Invalid Kraus operator shape {shape} for "
                        + f"acting on {len(qubits)} qubits.",
                    )
                qubitset.update(qubits)
                gates.append(Unitary(matrix, *list(qubits)))
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
        probabilities (list): List of floats that correspond to the probability
            that each unitary Uk is applied.
        ops (list): List of  operators as pairs ``(qubits, Uk)`` where
            ``qubits`` refers the qubit ids that ``Uk`` acts on and ``Uk`` is
            the corresponding matrix as a ``np.ndarray``/``tf.Tensor``.
            Must have the same length as the given probabilities ``p``.
    """

    def __init__(self, probabilities, ops):
        if len(probabilities) != len(ops):
            raise_error(
                ValueError,
                f"Probabilities list has length {len(probabilities)} while "
                + f"{len(ops)} gates were given.",
            )
        for p in probabilities:
            if p < 0 or p > 1:
                raise_error(
                    ValueError,
                    f"Probabilities should be between 0 and 1 but {p} was given.",
                )
        super().__init__(ops)
        self.name = "UnitaryChannel"
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
    """Noise channel that applies Pauli operators with given probabilities.

    Implements the following transformation:

    .. math::
        \\mathcal{E}(\\rho ) = (1 - p_x - p_y - p_z) \\rho + p_x X\\rho X + p_y Y\\rho Y + p_z Z\\rho Z

    which can be used to simulate phase flip and bit flip errors.
    This channel can be simulated using either density matrices or state vectors
    and sampling with repeated execution.
    See :ref:`How to perform noisy simulation? <noisy-example>` for more
    information.

    Args:
        q (int): Qubit id that the noise acts on.
        px (float): Bit flip (X) error probability.
        py (float): Y-error probability.
        pz (float): Phase flip (Z) error probability.
    """

    def __init__(self, q, px=0, py=0, pz=0):
        warnings.warn(
            "This channel will be removed in a later release. "
            + "Use GeneralizedPauliNoiseChannel instead.",
            DeprecationWarning,
        )

        probs, gates = [], []
        for p, gate in [(px, X), (py, Y), (pz, Z)]:
            if p > 0:
                probs.append(p)
                gates.append(gate(q))

        super().__init__(probs, gates)
        self.name = "PauliNoiseChannel"
        assert self.target_qubits == (q,)

        self.init_args = [q]
        self.init_kwargs = {"px": px, "py": py, "pz": pz}


class GeneralizedPauliNoiseChannel(UnitaryChannel):
    """Multi-qubit noise channel that applies Pauli operators with given probabilities.

    Implements the following transformation:

    .. math::
        \\mathcal{E}(\\rho ) = \\left (1 - \\sum _{k} p_{k} \\right ) \\, \\rho +
                                \\sum_{k} \\, p_{k} \\, P_{k} \\, \\rho \\, P_{k}


    where :math:`P_{k}` is the :math:`k`-th Pauli ``string`` and :math:`p_{k}` is
    the probability associated to :math:`P_{k}`.

    Example:
        .. testcode::

            import numpy as np

            from itertools import product

            from qibo.gates.channels import GeneralizedPauliNoiseChannel

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

            channel = GeneralizedPauliNoiseChannel(
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
        warnings.warn(
            "The class GeneralizedPauliNoiseChannel will be renamed "
            + "PauliNoiseChannel in a later release."
        )

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

        super().__init__(probabilities, gates)
        self.name = "GeneralizedPauliNoiseChannel"
        self.init_args = qubits
        self.init_kwargs = dict(operators)


class DepolarizingChannel(Channel):
    """:math:`n`-qubit Depolarizing quantum error channel,

    .. math::
        \\mathcal{E}(\\rho ) = (1 - \\lambda) \\rho +\\lambda \\text{Tr}_q[\\rho]\\otimes \\frac{I}{2^n}

    where :math:`\\lambda` is the depolarizing error parameter
    and :math:`0 \\le \\lambda \\le 4^n / (4^n - 1)`.

    * If :math:`\\lambda = 1` this is a completely depolarizing channel
      :math:`E(\\rho) = I / 2^n`
    * If :math:`\\lambda = 4^n / (4^n - 1)` this is a uniform Pauli
      error channel: :math:`E(\\rho) = \\sum_j P_j \\rho P_j / (4^n - 1)` for
      all :math:`P_j \\neq I`.

    Args:
        q (tuple): Qubit ids that the noise acts on.
        lam (float): Depolarizing error parameter.
    """

    def __init__(self, q, lam: str = 0):
        if isinstance(q, int) is True:
            q = (q,)

        super().__init__()
        num_qubits = len(q)
        num_terms = 4**num_qubits
        max_param = num_terms / (num_terms - 1)
        if lam < 0 or lam > max_param:
            raise_error(
                ValueError,
                f"Depolarizing parameter must be in between 0 and {max_param}.",
            )

        self.name = "DepolarizingChannel"
        self.target_qubits = q

        self.init_args = [q]
        self.init_kwargs = {"lam": lam}

    def apply_density_matrix(self, backend, state, nqubits):
        return backend.depolarizing_error_density_matrix(self, state, nqubits)

    def apply(self, backend, state, nqubits):
        num_qubits = len(self.target_qubits)
        num_terms = 4**num_qubits
        prob_pauli = self.init_kwargs["lam"] / num_terms
        probs = (num_terms - 1) * [prob_pauli]
        gates = []
        for pauli_list in list(product([I, X, Y, Z], repeat=num_qubits))[1::]:
            fgate = FusedGate(*self.target_qubits)
            for j, pauli in enumerate(pauli_list):
                fgate.append(pauli(j))
            gates.append(fgate)
        self.gates = tuple(gates)
        self.coefficients = tuple(probs)

        return backend.apply_channel(self, state, nqubits)


class ThermalRelaxationChannel(KrausChannel):
    """Single-qubit thermal relaxation error channel.

    Implements the following transformation:

    If :math:`T_1 \\geq T_2`:

    .. math::
        \\mathcal{E} (\\rho ) = (1 - p_z - p_0 - p_1)\\rho + p_zZ\\rho Z
        +  \\mathrm{Tr}_q[\\rho] \\otimes (p_0|0\\rangle \\langle 0| + p_1|1\\rangle \\langle 1|)


    while if :math:`T_1 < T_2`:

    .. math::
        \\mathcal{E}(\\rho ) = \\mathrm{Tr} _\\mathcal{X}\\left [\\Lambda _{\\mathcal{X}\\mathcal{Y}}(\\rho _\\mathcal{X} ^T \\otimes I_\\mathcal{Y})\\right ]

    with

    .. math::
        \\Lambda = \\begin{pmatrix}
        1 - p_1 & 0 & 0 & e^{-t / T_2} \\\\
        0 & p_1 & 0 & 0 \\\\
        0 & 0 & p_0 & 0 \\\\
        e^{-t / T_2} & 0 & 0 & 1 - p_0
        \\end{pmatrix}

    where :math:`p_0 = (1 - e^{-t / T_1})(1 - \\eta )` :math:`p_1 = (1 - e^{-t / T_1})\\eta`
    and :math:`p_z = (e^{-t / T_1} - e^{-t / T_2})/2`.
    Here :math:`\\eta` is the ``excited_population``
    and :math:`t` is the ``time``, both controlled by the user.
    This gate is based on
    `Qiskit's thermal relaxation error channel <https://qiskit.org/documentation/stubs/qiskit.providers.aer.noise.thermal_relaxation_error.html#qiskit.providers.aer.noise.thermal_relaxation_error>`_.

    Args:
        q (int): Qubit id that the noise channel acts on.
        t1 (float): T1 relaxation time. Should satisfy ``t1 > 0``.
        t2 (float): T2 dephasing time.
            Should satisfy ``t1 > 0`` and ``t2 < 2 * t1``.
        time (float): the gate time for relaxation error.
        excited_population (float): the population of the excited state at
            equilibrium. Default is 0.
    """

    def __init__(self, q, t1, t2, time, excited_population=0):
        self.name = "ThermalRelaxationChannel"

        # check given parameters
        if excited_population < 0 or excited_population > 1:
            raise_error(
                ValueError, f"Invalid excited state population {excited_population}."
            )
        if time < 0:
            raise_error(ValueError, "Invalid gate_time ({time} < 0).")
        if t1 <= 0:
            raise_error(
                ValueError, "Invalid T_1 relaxation time parameter: " "T_1 <= 0."
            )
        if t2 <= 0:
            raise_error(
                ValueError, "Invalid T_2 relaxation time parameter: " "T_2 <= 0."
            )
        if t2 > 2 * t1:
            raise_error(
                ValueError,
                "Invalid T_2 relaxation time parameter: " "T_2 greater than 2 * T_1.",
            )

        # calculate probabilities
        self.t1, self.t2 = t1, t2
        p_reset = 1 - exp(-time / t1)
        preset0 = p_reset * (1 - excited_population)
        preset1 = p_reset * excited_population

        import numpy as np

        if t1 < t2:
            exp_t2 = exp(-time / t2)

            from qibo.quantum_info import choi_to_kraus

            choi_matrix = np.array(
                [
                    [1 - preset1, 0, 0, exp_t2],
                    [0, preset1, 0, 0],
                    [0, 0, preset0, 0],
                    [exp_t2, 0, 0, 1 - preset0],
                ]
            )

            operators, _ = choi_to_kraus(choi_matrix)
            operators = list(zip([(q,)] * len(operators), operators))
            super().__init__(ops=operators)
            self.init_kwargs["exp_t2"] = exp_t2

        else:
            pz = (exp(-time / t1) - exp(-time / t2)) / 2
            operators = (
                sqrt(preset0) * np.array([[1, 0], [0, 0]]),
                sqrt(preset0) * np.array([[0, 1], [0, 0]]),
                sqrt(preset1) * np.array([[0, 0], [1, 0]]),
                sqrt(preset1) * np.array([[0, 0], [0, 1]]),
                sqrt(pz) * np.array([[1, 0], [0, -1]]),
                sqrt(1 - preset0 - preset1 - pz) * np.eye(2),
            )
            operators = list(zip([(q,)] * len(operators), operators))
            super().__init__(ops=operators)
            self.init_kwargs["pz"] = pz

        self.init_args = [q, t1, t2, time]
        self.t1, self.t2 = t1, t2
        self.init_kwargs["excited_population"] = excited_population
        self.init_kwargs["p0"] = preset0
        self.init_kwargs["p1"] = preset1

    def apply_density_matrix(self, backend, state, nqubits):
        q = self.target_qubits[0]

        if self.t1 < self.t2:
            preset0, preset1, exp_t2 = (
                self.init_kwargs["p0"],
                self.init_kwargs["p1"],
                self.init_kwargs["exp_t2"],
            )
            matrix = [
                [1 - preset1, 0, 0, preset0],
                [0, exp_t2, 0, 0],
                [0, 0, exp_t2, 0],
                [preset1, 0, 0, 1 - preset0],
            ]

            qubits = (q, q + nqubits)
            gate = Unitary(matrix, *qubits)

            return backend.thermal_error_density_matrix(gate, state, nqubits)

        else:
            pz = self.init_kwargs["pz"]

            return (
                backend.reset_error_density_matrix(self, state, nqubits)
                - pz * backend.cast(state)
                + pz * backend.apply_gate_density_matrix(Z(0), state, nqubits)
            )


class ReadoutErrorChannel(KrausChannel):
    """Readout error channel implemented as a quantum-to-classical channel.

    Args:
        q (int or list or tuple): Qubit ids that the channel acts on.
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

    def __init__(self, q: Tuple[int, list, tuple], probabilities):
        if any(sum(row) < 1 - PRECISION_TOL for row in probabilities) or any(
            sum(row) > 1 + PRECISION_TOL for row in probabilities
        ):
            raise_error(ValueError, "all rows of probabilities must sum to 1.")

        if isinstance(q, int) is True:
            q = (q,)

        import numpy as np

        d = len(probabilities)
        operators = []
        for j in range(d):
            for k in range(d):
                operator = np.zeros((d, d))
                operator[j, k] = sqrt(probabilities[k, j])
                operators.append(operator)

        operators = list(zip([q] * len(operators), operators))

        super().__init__(ops=operators)
        self.name = "ReadoutErrorChannel"


class ResetChannel(KrausChannel):
    """Single-qubit reset channel.

    Implements the following transformation:

    .. math::
        \\mathcal{E}(\\rho ) = (1 - p_0 - p_1) \\rho
        +  \\mathrm{Tr}_q[\\rho] \\otimes (p_0|0\\rangle \\langle 0| + p_1|1\\rangle \\langle 1|),

    Args:
        q (int): Qubit id that the channel acts on.
        p0 (float): Probability to reset to 0.
        p1 (float): Probability to reset to 1.
    """

    def __init__(self, q, p0=0.0, p1=0.0):
        import numpy as np

        if p0 < 0:
            raise_error(ValueError, "Invalid p0 ({p0} < 0).")
        if p1 < 0:
            raise_error(ValueError, "Invalid p1 ({p1} < 0).")
        if p0 + p1 > 1 + PRECISION_TOL:
            raise_error(ValueError, f"Invalid probabilities (p0 + p1 = {p0+p1} > 1).")

        operators = [
            sqrt(p0) * np.array([[1, 0], [0, 0]]),
            sqrt(p0) * np.array([[0, 1], [0, 0]]),
            sqrt(p1) * np.array([[0, 0], [1, 0]]),
            sqrt(p1) * np.array([[0, 0], [0, 1]]),
        ]

        if p0 + p1 < 1:
            operators.append(sqrt(np.abs(1 - p0 - p1)) * np.eye(2))

        operators = list(zip([(q,)] * len(operators), operators))
        super().__init__(ops=operators)
        self.init_kwargs = {"p0": p0, "p1": p1}
        self.name = "ResetChannel"

    def apply_density_matrix(self, backend, state, nqubits):
        return backend.reset_error_density_matrix(self, state, nqubits)
