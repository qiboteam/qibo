import math
from fractions import Fraction
from numbers import Integral

import numpy as np

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.models.algorithms.qft import QFT
from qibo.models.circuit import Circuit


class Shor:
    """Shor's algorithm to factor an integer.

    Even numbers and prime powers are factored classically, as in Ref. [1]. Otherwise, a base
    :math:`x` is chosen at random, which factors :math:`n` if it shares a divisor with it, and
    the order :math:`r` of :math:`x` modulo :math:`n`, *i.e.* the smallest :math:`r` such that
    :math:`x^{r} \\equiv 1 \\pmod{n}`, is found. If :math:`r` is even and
    :math:`x^{r/2} \\not\\equiv -1 \\pmod{n}`, then :math:`\\gcd(x^{r/2} - 1, n)` is a
    nontrivial factor of :math:`n`. The order is found with a circuit that acts on two
    registers, with :math:`q = 2^{l}` such that :math:`n^{2} \\leq q < 2 n^{2}`:

    1. a Hadamard gate on each of the :math:`l` qubits of the first register, which prepares
       :math:`q^{-1/2} \\sum_{a = 0}^{q - 1} \\ket{a}`, and the second register in
       :math:`\\ket{1}`;
    2. the modular exponentiation :math:`\\ket{a} \\ket{1} \\mapsto \\ket{a} \\ket{x^{a} \\bmod n}`,
       which multiplies the second register by :math:`x^{2^{k}} \\bmod n` if the bit :math:`k` of
       :math:`a` is :math:`1`;
    3. the :func:`qibo.models.QFT` of the first register, and its measurement, with outcome
       :math:`c`.

    With a constant probability, :math:`c / q` is closer than :math:`1 / (2q)` to a fraction
    :math:`d / r`, which is found with the continued fraction expansion of :math:`c / q`, with
    denominators smaller than :math:`n`. The post-processing also tries the outcomes
    :math:`c \\pm 1` and :math:`c \\pm 2`, and the multiples of the denominators, which are the
    ones suggested in Ref. [1]. Each multiplication by :math:`x^{2^{k}} \\bmod n` is a permutation
    of the basis states of the second register, which is built from the classical value of
    :math:`x^{2^{k}} \\bmod n`, instead of the arithmetic circuit of Ref. [1]. Hence, the circuit
    is exact, but its size grows exponentially with the number of bits of :math:`n`, and it is
    only meant for small numbers.

    The circuit of the last trial is stored in the attribute ``circuit``, which is ``None``
    before :meth:`execute`, and if no circuit was needed. The first register has ``ncounting``
    qubits, and the second one has ``nbits`` qubits, which are also stored as attributes.

    Example:
        .. testcode::

            from qibo.models import Shor

            print(Shor(15).execute())

        .. testoutput::

            (3, 5)

    Args:
        nvalue (int): integer :math:`n \\geq 4` to be factored. It has to be composite, because
            no factor is found otherwise.
        base (int, optional): base :math:`x`, with :math:`1 < x < n`, used in all of the trials.
            If ``None``, a random base is drawn in each trial. Defaults to ``None``.
        seed (int or :class:`numpy.random.Generator`, optional): seed or generator used to draw the
            bases, with the ``default_rng`` of the backend in :meth:`execute`. An integer gives the
            same bases in each execution. The measurements are sampled by the backend, so they are
            reproducible only if the seed of the backend is set, *e.g.* with
            :meth:`qibo.backends.abstract.Backend.set_seed`. Defaults to ``None``.

    References:
        1. P. W. Shor, *Polynomial-time algorithms for prime factorization and discrete
        logarithms on a quantum computer*, `SIAM J. Comput. 26, 1484 (1997)
        <https://doi.org/10.1137/S0097539795293172>`_.
    """

    def __init__(
        self,
        nvalue: int,
        base: int | None = None,
        seed: int | np.random.Generator | None = None,
    ):
        if isinstance(nvalue, bool) or not isinstance(nvalue, Integral) or nvalue < 4:
            raise_error(
                ValueError,
                f"``nvalue`` must be an integer of at least 4, but it is {nvalue}.",
            )

        if base is not None and (
            isinstance(base, bool)
            or not isinstance(base, Integral)
            or not 1 < base < nvalue
        ):
            raise_error(
                ValueError,
                f"``base`` must be an integer in (1, {nvalue}), but it is {base}.",
            )

        self.nvalue = int(nvalue)
        self.base = None if base is None else int(base)
        self.seed = seed
        self.circuit = None

        # the first register has the qubits of q = 2^ncounting, and the second one has the
        # qubits of the values between 0 and n - 1
        self.ncounting = (self.nvalue**2 - 1).bit_length()
        self.nbits = (self.nvalue - 1).bit_length()

    def execute(
        self, ntrials: int = 10, backend: Backend | None = None
    ) -> tuple[int, int]:
        """Executes Shor's algorithm.

        Each of the at most ``ntrials`` trials runs the circuit once, and the attribute
        ``circuit`` is the one of the last trial. A ``RuntimeError`` is raised if no factor is
        found, *e.g.* if the number is prime.

        Args:
            ntrials (int, optional): maximum number of trials, each of which runs the circuit
                once. Defaults to :math:`10`.
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in
                the execution of the circuits. If ``None``, it uses the current backend.
                Defaults to ``None``.

        Returns:
            tuple: Two integers :math:`p \\leq p'`, with :math:`p p' = n` and :math:`1 < p`.
        """
        backend = _check_backend(backend)

        if (
            isinstance(ntrials, bool)
            or not isinstance(ntrials, Integral)
            or ntrials < 1
        ):
            raise_error(
                ValueError,
                f"``ntrials`` must be a positive integer, but it is {ntrials}.",
            )

        nvalue = self.nvalue
        self.circuit = None

        # even numbers and prime powers
        if nvalue % 2 == 0:
            return 2, nvalue // 2

        for exponent in range(2, nvalue.bit_length()):
            root = round(nvalue ** (1 / exponent))
            for candidate in (root - 1, root, root + 1):
                if candidate > 1 and candidate**exponent == nvalue:
                    return candidate, nvalue // candidate

        modulus = 2**self.ncounting

        # an integer seed starts the same sequence of bases in each execution, and a generator
        # continues its own
        generator = (
            backend.default_rng(self.seed)
            if self.seed is None or isinstance(self.seed, int)
            else self.seed
        )

        for _ in range(ntrials):
            integer = (
                self.base
                if self.base is not None
                else int(generator.integers(2, nvalue - 1))
            )

            divisor = math.gcd(integer, nvalue)
            if divisor > 1:
                return tuple(sorted((divisor, nvalue // divisor)))

            circuit = Circuit(self.ncounting + self.nbits)
            circuit.add(gates.H(qubit) for qubit in range(self.ncounting))
            circuit.add(
                _modular_exponentiation(integer, nvalue, self.nbits, self.ncounting)
            )
            circuit.add(QFT(self.ncounting).on_qubits(*range(self.ncounting)))
            circuit.add(gates.M(*range(self.ncounting)))
            self.circuit = circuit

            frequencies = backend.execute_circuit(circuit, nshots=1).frequencies(
                binary=True
            )
            outcome = int(next(iter(frequencies)), 2)

            # fractions close to the outcome, with denominators smaller than n, and their
            # multiples
            order = None
            for shift in (0, 1, -1, 2, -2):
                measured = (outcome + shift) % modulus
                denominator = (
                    Fraction(measured, modulus)
                    .limit_denominator(nvalue - 1)
                    .denominator
                )
                for multiple in range(1, self.nbits + 2):
                    if pow(integer, multiple * denominator, nvalue) == 1:
                        order = multiple * denominator
                        break

                if order is not None:
                    break

            if order is None:
                continue

            # the multiple of the order is replaced by the smallest divisor with the same
            # property
            order = min(
                divisor
                for divisor in range(1, order + 1)
                if order % divisor == 0 and pow(integer, divisor, nvalue) == 1
            )

            if order % 2 == 0 and pow(integer, order // 2, nvalue) != nvalue - 1:
                divisor = math.gcd(pow(integer, order // 2, nvalue) - 1, nvalue)
                return tuple(sorted((divisor, nvalue // divisor)))

        raise_error(
            RuntimeError,
            f"No factor of {nvalue} was found in {ntrials} trials. "
            + "Is it a prime number? Otherwise, increase ntrials.",
        )


def _modular_exponentiation(
    base: int, nvalue: int, nbits: int, ncounting: int
) -> Circuit:
    """Creates the circuit of the modular exponentiation of Ref. [1].

    It acts on :math:`l + m` qubits, where :math:`l` is ``ncounting`` and :math:`m` is
    ``nbits``, and it takes :math:`\\ket{a} \\ket{0 \\cdots 01}` to
    :math:`\\ket{a} \\ket{x^{a} \\bmod n}`, where :math:`x` is ``base`` and :math:`n` is
    ``nvalue``. The first :math:`l` qubits hold :math:`a`, with the most significant bit
    first, and the last :math:`m` qubits hold the value of the second register.

    Args:
        base (int): base :math:`x`, which is coprime to :math:`n`.
        nvalue (int): modulus :math:`n`.
        nbits (int): number of qubits :math:`m` of the second register, such that
            :math:`n \\leq 2^{m}`.
        ncounting (int): number of qubits :math:`l` of the first register.

    Returns:
        :class:`qibo.models.circuit.Circuit`: circuit on :math:`l + m` qubits.

    References:
        1. P. W. Shor, *Polynomial-time algorithms for prime factorization and discrete
        logarithms on a quantum computer*, `SIAM J. Comput. 26, 1484 (1997)
        <https://doi.org/10.1137/S0097539795293172>`_.
    """
    register = list(range(ncounting, ncounting + nbits))
    dimension = 2**nbits

    circuit = Circuit(ncounting + nbits)
    circuit.add(gates.X(register[-1]))

    for bit in range(ncounting):
        multiplier = pow(base, 2**bit, nvalue)
        if multiplier == 1:
            continue

        # multiplication by a number that is coprime to n permutes the values below n
        matrix = [[0.0] * dimension for _ in range(dimension)]
        for value in range(dimension):
            product = value * multiplier % nvalue if value < nvalue else value
            matrix[product][value] = 1.0

        # the bit 2^bit of a is in the qubit that has the same weight, with the first qubit
        # as the most significant one
        control = ncounting - 1 - bit
        circuit.add(gates.Unitary(matrix, *register).controlled_by(control))

    return circuit
