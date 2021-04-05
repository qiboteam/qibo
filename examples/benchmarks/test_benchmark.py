import pytest
import qibo
from qibo import models, gates


def test_qft_create(benchmark, nqubits, accelerators, backend, precision):
    qibo.set_backend(backend)
    qibo.set_precision(precision)
    circuit = benchmark(models.QFT, nqubits, accelerators)


def test_qft_execute(benchmark, nqubits, accelerators, backend, precision):
    qibo.set_backend(backend)
    qibo.set_precision(precision)
    circuit = models.QFT(nqubits, accelerators)
    result = benchmark(circuit)


def test_frequency_sampling(benchmark, nqubits, nshots, backend, precision):
    qibo.set_backend(backend)
    qibo.set_precision(precision)
    circuit = models.Circuit(nqubits)
    circuit.add((gates.H(i) for i in range(nqubits)))
    circuit.add(gates.M(*range(nqubits)))
    result = circuit(nshots=nshots)
    result = benchmark(result.frequencies)
