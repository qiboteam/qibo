import numpy as np
import pytest

from qibo import gates
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models import Circuit
from qibo.models.error_mitigation import CDR
from qibo.noise import DepolarizingError, NoiseModel, PauliError
from qibo.symbols import I, X, Y, Z


def get_noise_model(error):
    noise = NoiseModel()
    noise.add(error, gates.CNOT)
    return noise


@pytest.mark.parametrize("nqubits", [2, 3, 4])
@pytest.mark.parametrize(
    "noise",
    [get_noise_model(DepolarizingError(0.1)), get_noise_model(PauliError(pz=0.1))],
)
def test_cdr(nqubits, noise):
    """Test that CDR reduces the noise."""
    # Define the circuit
    nlayers = 3
    c = Circuit(nqubits, density_matrix=True)
    for l in range(nlayers):
        c.add(gates.RY(q, np.pi / 2) for q in range(nqubits))
        c.add(gates.RZ(q, theta=np.random.rand() * 2 * np.pi) for q in range(nqubits))
        c.add(gates.CNOT(q, q + 1) for q in range(0, nqubits - 1, 2))
        c.add(gates.RZ(q, theta=np.random.rand() * 2 * np.pi) for q in range(nqubits))
        c.add(gates.CNOT(q, q + 1) for q in range(1, nqubits - 2, 2))
        c.add(gates.CNOT(0, nqubits - 1))
        c.add(gates.RZ(q, theta=np.random.rand() * 2 * np.pi) for q in range(nqubits))
    c.add(gates.M(*range(nqubits)))

    # Define the observable
    obs = np.prod([Z(i) for i in range(nqubits)])
    obs = SymbolicHamiltonian(obs)
    # Define the noise model
    # Noise-free expected value
    exact = obs.expectation(c().state())
    noisy = obs.expectation(noise.apply(c)().state())
    estimate = CDR(circuit=c, observable=obs, noise_model=noise, nshots=10000)
    assert np.abs(exact - estimate) <= np.abs(exact - noisy)
