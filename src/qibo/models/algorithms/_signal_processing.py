"""Private helpers shared by quantum signal processing (QSP) and quantum singular
value transformation (QSVT)."""

from numpy.typing import ArrayLike

from qibo.backends import Backend, _check_backend


def _spectral_factor(
    residual: ArrayLike, degree: int, backend: Backend = None
) -> tuple[ArrayLike, bool]:
    """Computes the spectral factor of a positive trigonometric polynomial.

    The cepstrum, i.e. the Fourier series of the logarithm, of the polynomial
    :math:`F(x) = |h(e^{i x})|^{2}` gives the factor :math:`h(z) = \\sum_{n = 0}^{q}
    h_{n} z^{n}` that is analytic and has no roots in the unit disk, which is stable
    for large degrees. A positive polynomial with real coefficients has a real factor.

    Args:
        residual (ArrayLike): values of :math:`F` on :math:`N` equispaced points of
            :math:`[0, 2 \\pi)`, with :math:`N` a power of two.
        degree (int): degree :math:`q` of the factor.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        tuple: coefficients :math:`(h_{0}, \\dots, h_{q})` and whether the factor
        has converged, i.e. its coefficients beyond :math:`q` are negligible.
    """
    backend = _check_backend(backend)

    nsamples = len(residual)
    cepstrum = backend.fft(backend.log(residual)) / nsamples
    causal = backend.zeros(nsamples, dtype="complex128")
    causal[0] = cepstrum[0] / 2
    causal[1 : nsamples // 2] = cepstrum[1 : nsamples // 2]
    causal[nsamples // 2] = cepstrum[nsamples // 2] / 2
    series = backend.fft(backend.exp(backend.ifft(causal) * nsamples)) / nsamples
    converged = backend.max(backend.abs(series[degree + 1 : nsamples // 2])) < 1e-12

    return series[: degree + 1], bool(converged)
