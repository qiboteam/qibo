"""Tests for the phase retrieval in `qibo/models/algorithms/phase_retrieval.py`."""

import pytest

from qibo.models import hybrid_input_output, vectorial_phase_retrieval


@pytest.mark.parametrize(
    "kwargs",
    [
        {"magnitudes": [1.0, 2.0]},
        {"phases": [0.0, 1.0]},
        {"beta": -0.1},
        {"beta": 1.5},
        {"beta": "0.9"},
        {"niterations": 0},
        {"niterations": 2.5},
        {"initial": [[1.0, 2.0], [3.0, 4.0]]},
    ],
)
def test_hybrid_input_output_errors(backend, kwargs):
    kwargs = {"magnitudes": backend.ones((5, 7)), "backend": backend, **kwargs}

    with pytest.raises(ValueError):
        hybrid_input_output(**kwargs)


@pytest.mark.parametrize("shape", [(15, 17), (16, 16)])
def test_hybrid_input_output_fixed_point(backend, shape):
    """The spectrum of the exact signal is a fixed point of the algorithm.

    The signal has lines at frequencies whose weights are positive, so its spectrum is real,
    and it is positive after multiplying by the triangular window.
    """
    signal, window = _signal(backend, shape)
    spectrum = backend.real(backend.fft(backend.fft(window * signal, axis=0), axis=1))

    result, estimate = hybrid_input_output(
        backend.abs(signal),
        phases=backend.angle(signal[0]),
        niterations=5,
        initial=spectrum,
        backend=backend,
    )

    assert float(backend.min(spectrum)) > 0
    backend.assert_allclose(result, spectrum, atol=1e-10)
    backend.assert_allclose(estimate, window * signal, atol=1e-10)


def test_hybrid_input_output_reproducible(backend):
    signal, _ = _signal(backend, (11, 13))
    kwargs = {"magnitudes": backend.abs(signal), "niterations": 20, "backend": backend}

    spectrum, estimate = hybrid_input_output(seed=3, **kwargs)
    spectrum_same, estimate_same = hybrid_input_output(seed=3, **kwargs)
    spectrum_other, _ = hybrid_input_output(seed=4, **kwargs)

    backend.assert_allclose(spectrum, spectrum_same)
    backend.assert_allclose(estimate, estimate_same)
    assert float(backend.max(backend.abs(spectrum - spectrum_other))) > 1e-6


def test_hybrid_input_output_signal_constraints(backend):
    """The estimate has the absolute values of the data, which include the window, and the
    known phases of the time zero."""
    signal, window = _signal(backend, (15, 17))

    spectrum, estimate = hybrid_input_output(
        backend.abs(signal),
        phases=backend.angle(signal[0]),
        niterations=50,
        seed=0,
        backend=backend,
    )

    assert spectrum.shape == signal.shape
    backend.assert_allclose(backend.abs(estimate), window * backend.abs(signal))
    backend.assert_allclose(estimate[0], window[0] * signal[0], atol=1e-12)


@pytest.mark.parametrize("orthogonal", [False, True])
@pytest.mark.parametrize(
    "nsamples, support, nreferences", [(16, 5, 1), (24, 8, 2), (40, 14, 3)]
)
def test_vectorial_phase_retrieval(backend, nsamples, support, nreferences, orthogonal):
    """If the support is exact, the phases of the signal are recovered up to a global phase.

    If the references are orthogonal to the target state, their signals vanish at the time zero,
    which makes the relative phase of this time undefined.
    """
    signal, target, references = _probabilities(
        backend, nsamples, support, nreferences, orthogonal=orthogonal
    )

    estimate, eigenvalues = vectorial_phase_retrieval(
        target, references, support, backend=backend
    )

    backend.assert_allclose(estimate, signal, atol=1e-8)
    assert abs(float(eigenvalues[0])) < 1e-10
    assert float(eigenvalues[1]) > 1e-4


def test_vectorial_phase_retrieval_eigenvalues(backend):
    """The smallest eigenvalue drops to zero at the size of the support."""
    _, target, references = _probabilities(backend, 24, 8, 2)

    smallest = [
        float(vectorial_phase_retrieval(target, references, s, backend=backend)[1][0])
        for s in (5, 6, 7, 8, 9)
    ]

    assert min(smallest[:3]) > 1e-3
    assert abs(smallest[3]) < 1e-10
    assert abs(smallest[4]) < 1e-10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target": [[1.0, 2.0]]},
        {"references": [[1.0, 2.0]]},
        {"references": [[[1.0] * 6] * 3]},
        {"references": [[[1.0] * 8] * 2]},
        {"references": []},
        {"support": -1},
        {"support": 8},
        {"support": 2.5},
        {"support": True},
    ],
)
def test_vectorial_phase_retrieval_errors(backend, kwargs):
    kwargs = {
        "target": backend.ones(8),
        "references": backend.ones((1, 3, 8)),
        "support": 3,
        "backend": backend,
        **kwargs,
    }

    with pytest.raises(ValueError):
        vectorial_phase_retrieval(**kwargs)


@pytest.mark.parametrize("nreferences", [2, 4])
def test_vectorial_phase_retrieval_noise(backend, nreferences):
    """The error is linear in the leakage of the spectra out of the support, and in the noise
    of the probabilities, which is the one of a finite number of shots."""
    nsamples, support = 32, 10

    for tail, noise in ((1e-3, 0.0), (0.0, 1e-6)):
        signal, target, references = _probabilities(
            backend, nsamples, support, nreferences, tail=tail, noise=noise
        )

        estimate, _ = vectorial_phase_retrieval(
            target, references, support, backend=backend
        )

        error = float(backend.max(backend.abs(estimate - signal)))
        assert 1e-8 < error < 1e-1 * float(backend.max(backend.abs(signal)))


def _probabilities(
    backend, nsamples, support, nreferences, noise=0.0, orthogonal=False, tail=0.0
):
    # probabilities that the circuits measure, which are the squared absolute values of
    # f1, f2, f1 + f2 and f1 + i f2, where the last two are divided by 2 as their states are
    # normalized, and the signals are the exact ones if there is no noise
    # the signals have spectra that are supported on the first frequencies, with a leakage out of
    # them if tail is not zero, and the global phase that makes the first value of the target
    # real. The value of a signal at the time zero is the mean of its spectrum, so the references
    # vanish there if they are orthogonal to the target state
    generator = backend.default_rng(nsamples)
    shape = (nreferences + 1, nsamples)
    spectra = generator.normal(size=shape) + 1j * generator.normal(size=shape)
    spectra[:, support:] *= tail
    if orthogonal:
        spectra[1:, :support] -= spectra[1:, :support].mean(axis=1, keepdims=True)
    signals = backend.ifft(backend.cast(spectra, dtype=backend.complex128), axis=1)
    signals = signals * backend.conj(signals[0, 0]) / backend.abs(signals[0, 0])
    target, references = signals[0], signals[1:]
    probabilities = backend.cast(
        [
            [
                backend.abs(reference) ** 2,
                backend.abs(target + reference) ** 2 / 2,
                backend.abs(target + 1j * reference) ** 2 / 2,
            ]
            for reference in references
        ],
        dtype=backend.float64,
    )
    generator = backend.default_rng(1)
    target_probabilities = backend.abs(target) ** 2
    if noise > 0:
        target_probabilities += noise * backend.cast(
            generator.normal(size=nsamples), dtype=backend.float64
        )
        probabilities += noise * backend.cast(
            generator.normal(size=probabilities.shape), dtype=backend.float64
        )

    return (
        target,
        backend.abs(target_probabilities),
        backend.abs(probabilities),
    )


def _signal(backend, shape):
    # signal with three lines, whose frequencies are the energies of H and of H_D, and the
    # triangular window of the algorithm, with the signed indices of the times
    nrows, ncols = shape
    rows = [j if j <= nrows - j else j - nrows for j in range(nrows)]
    cols = [m if m <= ncols - m else m - ncols for m in range(ncols)]
    times_rows = backend.cast(rows, dtype=backend.float64)
    times_cols = backend.cast(cols, dtype=backend.float64)
    signal = sum(
        weight
        * backend.exp(1j * energy * times_rows)[:, None]
        * backend.exp(1j * charge * times_cols)[None, :]
        for weight, energy, charge in zip(
            (0.5, 0.3, 0.2), (0.5, 1.0, 1.6), (0.2, 0.8, 1.4)
        )
    )
    window = backend.cast(
        [
            [(1 - 2 * abs(j) / nrows) * (1 - 2 * abs(m) / ncols) for m in cols]
            for j in rows
        ],
        dtype=backend.float64,
    )

    return signal, window
