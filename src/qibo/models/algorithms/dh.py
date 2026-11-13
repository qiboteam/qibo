import math
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.models.algorithms.qaa import QAA
from qibo.models.circuit import Circuit


def DH(
    values: ArrayLike,
    timeout: float = 1.0,
    growth: float = 6 / 5,
    seed: int | np.random.Generator | None = None,
    backend: Backend | None = None,
) -> tuple[int, int]:
    """Finds the index of the minimum of a table with the Dürr-Høyer algorithm.

    Given a table :math:`T` of :math:`N = 2^{n}` values, a threshold index :math:`y` is chosen
    uniformly at random, and the following steps are repeated:

    1. the state :math:`N^{-1/2} \\sum_{j} \\ket{j}` of :math:`n` qubits is prepared, and the
       indices :math:`j` for which :math:`T[j] < T[y]` are marked, *i.e.* the oracle multiplies
       them by :math:`-1`;
    2. a marked index :math:`j` is searched for with the exponential search of Ref. [2]. It
       measures the circuit :math:`\\mathbf{Q}^{m} \\mathcal{A}`, with
       :math:`\\mathcal{A}` the Hadamard gates, :math:`\\mathbf{Q}` the operator of
       :func:`qibo.models.amplification_operator`, and :math:`m` drawn uniformly from
       :math:`\\{0, \\dots, \\lceil s \\rceil - 1\\}`, where :math:`s` is :math:`1` at
       first and is multiplied by ``growth`` after each failed measurement, up to
       :math:`\\sqrt{N}`;
    3. if the outcome :math:`j'` has :math:`T[j'] < T[y]`, the threshold becomes :math:`y = j'`.

    The algorithm is interrupted when the number of time steps is larger than
    ``timeout`` :math:`\\times (22.5 \\sqrt{N} + 1.4 \\lg^{2} N)`, where each preparation
    and marking takes :math:`\\lg N` steps, and each iteration of :math:`\\mathbf{Q}` takes one.
    The outcome of the measurement in progress is still used, and the index of the threshold is
    returned. With ``timeout`` :math:`= c`, it holds the minimum with probability at least
    :math:`1 - 2^{-c}` (Theorem 1 and the final remarks of Ref. [1]). If no index is marked, the
    threshold is the minimum, and the search in Ref. [1] would run until the time-out, but here
    the algorithm returns, since the table is known. The oracle is built from the table, with
    one multi-controlled gate for each of the marked indices, so that it is only efficient for
    a small :math:`N`.

    Example:
        .. testcode::

            from qibo.models import DH

            # the minimum is in the position 7
            table = [5, 3, 7, 1, 6, 2, 4, 0]
            index, napplications = DH(table, timeout=10, seed=0)
            print(index)

        .. testoutput::

            7

    Args:
        values (ArrayLike): :math:`1`-dimensional array with the :math:`N` values of the table,
            where :math:`N` is a power of :math:`2`. They do not need to be distinct.
        timeout (float, optional): factor :math:`c` that multiplies the number of time steps
            after which the algorithm is interrupted, for which the probability of finding the
            minimum is at least :math:`1 - 2^{-c}`. Defaults to :math:`1`.
        growth (float, optional): factor by which the maximum number of iterations grows after
            each failed measurement, which has to be in the interval :math:`(1, 2)`. Defaults to
            :math:`6 / 5`.
        seed (int or :class:`numpy.random.Generator`, optional): seed or generator used to draw
            the threshold index and the numbers of iterations. The measurements are sampled by
            the backend, so they are reproducible only if the seed of the backend is set, *e.g.*
            with :meth:`qibo.backends.abstract.Backend.set_seed`. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in the
            execution of the circuits. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        tuple: the index of the minimum value, and the number of applications of
        :math:`\\mathcal{A}` and :math:`\\mathcal{A}^{-1}` used, as in :func:`qibo.models.QSearch`.

    References:
        1. C. Dürr and P. Høyer, *A quantum algorithm for finding the minimum*,
        `arXiv:quant-ph/9607014 <https://arxiv.org/abs/quant-ph/9607014>`_.

        2. M. Boyer, G. Brassard, P. Høyer, and A. Tapp, *Tight bounds on quantum searching*,
        `Fortschr. Phys. 46, 493 (1998)
        <https://doi.org/10.1002/(SICI)1521-3978(199806)46:4/5%3C493::AID-PROP493%3E3.0.CO;2-P>`_.
    """
    backend = _check_backend(backend)

    values = backend.cast(values, dtype=backend.float64)
    nvalues = len(values)

    if len(values.shape) != 1 or nvalues < 2 or nvalues & (nvalues - 1) != 0:
        raise_error(
            ValueError,
            "values must be a 1-dimensional array with a power of 2 of entries, "
            + f"at least 2, but its shape is {tuple(values.shape)}.",
        )

    if isinstance(growth, bool) or not isinstance(growth, Real) or not 1 < growth < 2:
        raise_error(
            ValueError, f"growth must be in the interval (1, 2), but it is {growth}."
        )

    if isinstance(timeout, bool) or not isinstance(timeout, Real) or timeout <= 0:
        raise_error(
            ValueError, f"timeout must be a positive number, but it is {timeout}."
        )

    nqubits = int(math.log2(nvalues))
    generator = np.random.default_rng(seed)

    algorithm = Circuit(nqubits)
    algorithm.add(gates.H(qubit) for qubit in range(nqubits))

    budget = timeout * (22.5 * math.sqrt(nvalues) + 1.4 * nqubits**2)
    threshold = int(generator.integers(nvalues))

    time, napplications = 0, 0
    while True:
        marked = [
            state for state, smaller in enumerate(values < values[threshold]) if smaller
        ]
        if len(marked) == 0:
            break

        # oracle that multiplies the marked states by -1
        oracle = Circuit(nqubits)
        for state in marked:
            bits = format(state, f"0{nqubits}b")
            flipped = [qubit for qubit, bit in enumerate(bits) if bit == "0"]
            sign = gates.Z(nqubits - 1)
            if nqubits > 1:
                sign = sign.controlled_by(*range(nqubits - 1))

            oracle.add(gates.X(qubit) for qubit in flipped)
            oracle.add(sign)
            oracle.add(gates.X(qubit) for qubit in flipped)

        # exponential search of a marked state
        time += nqubits
        scale = 1.0
        while True:
            niterations = int(generator.integers(math.ceil(scale)))
            circuit = QAA(algorithm, oracle, niterations=niterations, backend=backend)
            outcome = backend.execute_circuit(circuit, nshots=1).frequencies()
            candidate = int(next(iter(outcome)), 2)

            time += niterations
            napplications += 2 * niterations + 1

            if values[candidate] < values[threshold]:
                threshold = candidate
                break

            if time > budget:
                break

            scale = min(growth * scale, math.sqrt(nvalues))

        if time > budget:
            break

    return threshold, napplications
