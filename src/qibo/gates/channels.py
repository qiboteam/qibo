from abc import abstractmethod
from qibo.gates.abstract import Gate
from qibo.gates.gates import X, Y, Z, Unitary
from qibo.gates.measurements import M
from qibo.config import raise_error, PRECISION_TOL


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
        raise_error(NotImplementedError, "`on_qubits` method is not available "
                                         "for the `Channel` gate.")

    def apply(self, backend, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, f"{self.name} cannot be applied to state vector.")

    def apply_density_matrix(self, backend, state, nqubits):
        return backend.apply_channel_density_matrix(self, state, nqubits)


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
            self.target_qubits = tuple(sorted(set(
                q for gate in ops for q in gate.target_qubits)))
        else:
            gates, qubitset = [], set()
            for qubits, matrix in ops:
                rank = 2 ** len(qubits)
                shape = tuple(matrix.shape)
                if shape != (rank, rank):
                    raise_error(ValueError, "Invalid Krauss operator shape {} for "
                                            "acting on {} qubits."
                                            "".format(shape, len(qubits)))
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
            raise_error(ValueError, "Probabilities list has length {} while "
                                    "{} gates were given."
                                    "".format(len(probabilities), len(ops)))
        for p in probabilities:
            if p < 0 or p > 1:
                raise_error(ValueError, "Probabilities should be between 0 "
                                        "and 1 but {} was given.".format(p))
        super().__init__(ops)
        self.name = "UnitaryChannel"
        self.coefficients = tuple(probabilities)
        self.coefficient_sum = sum(probabilities)
        if self.coefficient_sum > 1 + PRECISION_TOL or self.coefficient_sum <= 0:
            raise_error(ValueError, "UnitaryChannel probability sum should be "
                                    "between 0 and 1 but is {}."
                                    "".format(self.coefficient_sum))

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


class ResetChannel(Channel):
    """Single-qubit reset channel.

    Implements the following transformation:

    .. math::
        \\mathcal{E}(\\rho ) = (1 - p_0 - p_1) \\rho
        +  \mathrm{Tr}\\rho \\otimes (p_0|0\\rangle \\langle 0| + p_1|1\\rangle \langle 1|)

    Args:
        q (int): Qubit id that the channel acts on.
        p0 (float): Probability to reset to 0.
        p1 (float): Probability to reset to 1.
    """

    def __init__(self, q, p0=0.0, p1=0.0):
        super().__init__()
        self.name = "ResetChannel"
        self.target_qubits = (q,)
        self.coefficients = (p0, p1)
        self.init_args = [q]
        self.init_kwargs = {"p0": p0, "p1": p1}

    def apply_density_matrix(self, backend, state, nqubits):
        return backend.reset_error_density_matrix(self, state, nqubits)


class ThermalRelaxationChannel(Channel):
    """Single-qubit thermal relaxation error channel.

    Implements the following transformation:

    If :math:`T_1 \\geq T_2`:

    .. math::
        \\mathcal{E} (\\rho ) = (1 - p_z - p_0 - p_1)\\rho + p_zZ\\rho Z
        +  \mathrm{Tr}\\rho \\otimes (p_0|0\\rangle \\langle 0| + p_1|1\\rangle \langle 1|)


    while if :math:`T_1 < T_2`:

    .. math::
        \\mathcal{E}(\\rho ) = \\mathrm{Tr} _\\mathcal{X}\\left [\\Lambda _{\\mathcal{X}\\mathcal{Y}}(\\rho _\\mathcal{X} ^T \\otimes \\mathbb{I}_\\mathcal{Y})\\right ]

    with

    .. math::
        \\Lambda = \\begin{pmatrix}
        1 - p_1 & 0 & 0 & e^{-t / T_2} \\\\
        0 & p_1 & 0 & 0 \\\\
        0 & 0 & p_0 & 0 \\\\
        e^{-t / T_2} & 0 & 0 & 1 - p_0
        \\end{pmatrix}

    where :math:`p_0 = (1 - e^{-t / T_1})(1 - \\eta )` :math:`p_1 = (1 - e^{-t / T_1})\\eta`
    and :math:`p_z = 1 - e^{-t / T_1} + e^{-t / T_2} - e^{t / T_1 - t / T_2}`.
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
        super().__init__()
        self.name = "ThermalRelaxationChannel"
        self.target_qubits = (q,)
        self.init_args = [q, t1, t2, time]
        self.init_kwargs = {"excited_population": excited_population}

        # check given parameters
        if excited_population < 0 or excited_population > 1:
            raise_error(ValueError, "Invalid excited state population {}."
                                    "".format(excited_population))
        if time < 0:
            raise_error(ValueError, "Invalid gate_time ({} < 0)".format(time))
        if t1 <= 0:
            raise_error(ValueError, "Invalid T_1 relaxation time parameter: "
                                    "T_1 <= 0.")
        if t2 <= 0:
            raise_error(ValueError, "Invalid T_2 relaxation time parameter: "
                                    "T_2 <= 0.")
        if t2 > 2 * t1:
            raise_error(ValueError, "Invalid T_2 relaxation time parameter: "
                                    "T_2 greater than 2 * T_1.")

        # calculate probabilities
        import numpy as np
        self.t1, self.t2 = t1, t2
        p_reset = 1 - np.exp(-time / t1)
        self.coefficients = [p_reset * (1 - excited_population),
                             p_reset * excited_population]
        if t1 < t2:
            self.coefficients.append(np.exp(-time / t2))
        else:
            pz = p_reset + np.exp(-time / t2) * (1 - np.exp(time / t1))
            self.coefficients.append(pz)

    def apply_density_matrix(self, backend, state, nqubits):
        q = self.target_qubits[0]
        if self.t1 < self.t2:
            from qibo.gates import Unitary
            preset0, preset1, exp_t2 = self.coefficients
            matrix = [[1 - preset1, 0, 0, preset1],
                      [0, exp_t2, 0, 0],
                      [0, 0, exp_t2, 0],
                      [preset0, 0, 0, 1 - preset0]]

            qubits = (q, q + nqubits)
            gate = Unitary(matrix, *qubits)
            return backend.thermal_error_density_matrix(gate, state, nqubits)

        else:
            from qibo.gates import Z
            pz = self.coefficients[-1]
            return (backend.reset_error_density_matrix(self, state, nqubits) -
                    pz * backend.cast(state) +
                    pz * backend.apply_gate_density_matrix(Z(0), state, nqubits))
