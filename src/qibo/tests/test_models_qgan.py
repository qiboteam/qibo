"""Test style-qGAN model defined in `qibo/models/qgan.py`."""
import numpy as np
from qibo import gates, models
from qibo.models.qgan import StyleQGAN
import tensorflow as tf

 
def test_default_qgan(backend):
    layers = 2
    latent_dim = 2
    reference_distribution = []
    samples = 1000
    mean = [0, 0, 0]
    cov = [[0.5, 0.1, 0.25], [0.1, 0.5, 0.1], [0.25, 0.1, 0.5]]
    x, y, z = np.random.multivariate_normal(mean, cov, samples).T/4
    s1 = np.reshape(x, (samples,1))
    s2 = np.reshape(y, (samples,1))
    s3 = np.reshape(z, (samples,1))
    reference_distribution = np.hstack((s1,s2,s3))
    train_qGAN = StyleQGAN(latent_dim=latent_dim, layers=layers, n_epochs=10)
    train_qGAN.fit(reference_distribution)
    assert train_qGAN.layers == layers       
    assert train_qGAN.latent_dim == latent_dim
    assert train_qGAN.batch_samples == 128
    assert train_qGAN.n_epochs == 10
    assert train_qGAN.lr == 0.5
 
def test_custom_qgan(backend):
    
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
    layers = 1
    latent_dim = 2
    reference_distribution = []
    samples = 1000
    mean = [0, 0, 0]
    cov = [[0.5, 0.1, 0.25], [0.1, 0.5, 0.1], [0.25, 0.1, 0.5]]
    x, y, z = np.random.multivariate_normal(mean, cov, samples).T/4
    s1 = np.reshape(x, (samples,1))
    s2 = np.reshape(y, (samples,1))
    s3 = np.reshape(z, (samples,1))
    reference_distribution = np.hstack((s1,s2,s3))
    
    circuit = models.Circuit(nqubits)
    for l in range(layers):
        for q in range(nqubits):
            circuit.add(gates.RY(q, 0))
            circuit.add(gates.RZ(q, 0))
        for i in range(0, nqubits-1):
            circuit.add(gates.CZ(i, i+1))
        circuit.add(gates.CZ(nqubits-1, 0))
    for q in range(nqubits):
        circuit.add(gates.RY(q, 0))
        
    initial_params = tf.Variable(np.random.uniform(-0.15, 0.15, 18))    
    
    train_qGAN = StyleQGAN(latent_dim=latent_dim, circuit=circuit, set_parameters=set_params,
                           initial_params=initial_params, n_epochs=10)
    train_qGAN.fit(reference_distribution)
    assert train_qGAN.layers == layers        
    assert train_qGAN.latent_dim == latent_dim
    assert train_qGAN.batch_samples == 128
    assert train_qGAN.n_epochs == 10
    assert train_qGAN.lr == 0.5
