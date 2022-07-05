"""Test style-qGAN model defined in `qibo/models/qgan.py`."""
import pytest
import numpy as np
from qibo import gates, models


def generate_distribution(samples):
    mean = [0, 0, 0]
    cov = [[0.5, 0.1, 0.25], [0.1, 0.5, 0.1], [0.25, 0.1, 0.5]]
    x, y, z = np.random.multivariate_normal(mean, cov, samples).T / 4.0
    s1 = np.reshape(x, (samples,1))
    s2 = np.reshape(y, (samples,1))
    s3 = np.reshape(z, (samples,1))
    return np.hstack((s1, s2, s3))


def test_default_qgan():
    reference_distribution = generate_distribution(10)
    qgan = models.StyleQGAN(latent_dim=2, layers=1)
    qgan.fit(reference_distribution, n_epochs=1, save=False)
    assert qgan.layers == 1
    assert qgan.latent_dim == 2
    assert qgan.batch_samples == 128
    assert qgan.n_epochs == 1
    assert qgan.lr == 0.5


def test_custom_qgan():
    def set_params(circuit, params, x_input, i):
        """Set the parameters for the quantum generator circuit."""
        p = []
        index = 0
        noise = 0
        for l in range(1):
            for q in range(3):
                p.append(params[index]*x_input[noise][i] + params[index+1])
                index+=2
                noise=(noise+1)%2
                p.append(params[index]*x_input[noise][i] + params[index+1])
                index+=2
                noise=(noise+1)%2
        for q in range(3):
            p.append(params[index]*x_input[noise][i] + params[index+1])
            index+=2
            noise=(noise+1)%2
        circuit.set_parameters(p)

    nqubits = 3
    nlayers = 1
    reference_distribution = generate_distribution(10)
    circuit = models.Circuit(nqubits)
    for l in range(nlayers):
        for q in range(nqubits):
            circuit.add(gates.RY(q, 0))
            circuit.add(gates.RZ(q, 0))
        for i in range(0, nqubits - 1):
            circuit.add(gates.CZ(i, i + 1))
        circuit.add(gates.CZ(nqubits - 1, 0))
    for q in range(nqubits):
        circuit.add(gates.RY(q, 0))

    initial_params = np.random.uniform(-0.15, 0.15, 18)
    qgan = models.StyleQGAN(
            latent_dim=2,
            circuit=circuit,
            set_parameters=set_params
        )
    qgan.fit(reference_distribution, initial_params=initial_params, n_epochs=1, save=False)
    assert qgan.latent_dim == 2
    assert qgan.batch_samples == 128
    assert qgan.n_epochs == 1
    assert qgan.lr == 0.5


def test_qgan_errors():
    with pytest.raises(ValueError):
        qgan = models.StyleQGAN(latent_dim=2)
    circuit = models.Circuit(2)
    with pytest.raises(ValueError):
        qgan = models.StyleQGAN(latent_dim=2, layers=2, circuit=circuit)

    with pytest.raises(ValueError):
        qgan = models.StyleQGAN(latent_dim=2, circuit=circuit)
    with pytest.raises(ValueError):
        qgan = models.StyleQGAN(latent_dim=2, layers=2, set_parameters=lambda x: x)

    reference_distribution = generate_distribution(10)
    qgan = models.StyleQGAN(latent_dim=2, circuit=circuit, set_parameters=lambda x: x)
    with pytest.raises(ValueError):
        qgan.fit(reference_distribution, save=False)
    initial_params = np.random.uniform(-0.15, 0.15, 18)
    qgan = models.StyleQGAN(latent_dim=2, layers=2)
    with pytest.raises(ValueError):
        qgan.fit(reference_distribution, initial_params=initial_params, save=False)


def test_qgan_custom_discriminator():
    from tensorflow.keras.models import Sequential  # pylint: disable=E0611,E0401
    from tensorflow.keras.layers import Dense  # pylint: disable=E0611,E0401
    reference_distribution = generate_distribution(10)
    # use wrong number of qubits so that we capture the error
    nqubits = reference_distribution.shape[1] + 1
    discriminator = Sequential()
    discriminator.add(Dense(200, use_bias=False, input_dim=nqubits))
    discriminator.add(Dense(1, activation='sigmoid'))
    qgan = models.StyleQGAN(latent_dim=2, layers=1, discriminator=discriminator)
    with pytest.raises(ValueError):
        qgan.fit(reference_distribution, n_epochs=1, save=False)


def test_qgan_circuit_error():
    reference_distribution = generate_distribution(10)
    # use wrong number of qubits so that we capture the error
    nqubits = reference_distribution.shape[1] + 1
    nlayers = 1
    circuit = models.Circuit(nqubits)
    for l in range(nlayers):
        for q in range(nqubits):
            circuit.add(gates.RY(q, 0))
            circuit.add(gates.RZ(q, 0))
        for i in range(0, nqubits - 1):
            circuit.add(gates.CZ(i, i + 1))
        circuit.add(gates.CZ(nqubits - 1, 0))
    for q in range(nqubits):
        circuit.add(gates.RY(q, 0))

    initial_params = np.random.uniform(-0.15, 0.15, 18)
    qgan = models.StyleQGAN(
            latent_dim=2,
            circuit=circuit,
            set_parameters=lambda x: x
        )
    with pytest.raises(ValueError):
        qgan.fit(reference_distribution, initial_params=initial_params, n_epochs=1, save=False)
