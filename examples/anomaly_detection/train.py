''' Training of the quantum autoencoder with 5000 standard samples (zero handwritten digits)'''

import numpy as np
import tensorflow as tf
import qibo
import math
from qibo import gates
from qibo.models import Circuit
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import schedules

qibo.set_backend("tensorflow")

dataset_np=np.load("data/standard_data.npy")
dataset=tf.convert_to_tensor(dataset_np)

n_qubits=6
n_layers=6
q_compression=3
batch_size=20
nepochs = 20
train_size=5000
filename="parameters/trained_params"

n_params=(n_layers*n_qubits+q_compression)*3


def make_encoder(n_qubits, n_layers, params, q_compression):
    index=0
    encoder = Circuit(n_qubits)
    for i in range(n_layers):
        for j in range(n_qubits):
            encoder.add(gates.RX(j, params[index]))
            encoder.add(gates.RY(j, params[index+1]))
            encoder.add(gates.RZ(j, params[index+2]))
            index+=3
        
        for j in range(n_qubits):
            encoder.add(gates.CNOT(j,(j+1)%n_qubits))
            
    for j in range(q_compression):
        encoder.add(gates.RX(j, params[index]))
        encoder.add(gates.RY(j, params[index+1]))
        encoder.add(gates.RZ(j, params[index+2])) 
        index+=3  
    return encoder

params = tf.Variable(tf.random.normal((n_params,)))
encoder=make_encoder(n_qubits, n_layers, params, q_compression)
print("Circuit model summary")
print(encoder.draw())

@tf.function
def compute_loss(encoder, params, vector):
    encoder.set_parameters(params)
    reconstructed=encoder(vector)
    #3 qubits compression
    loss=reconstructed.probabilities(qubits=[0])[0] + reconstructed.probabilities(qubits=[1])[0] + reconstructed.probabilities(qubits=[2])[0]
    return loss

@tf.function
def train_step(batch_size, encoder, params, dataset):
    loss=0.
    with tf.GradientTape() as tape:
        for sample in range(batch_size):
            loss=loss+compute_loss(encoder, params, dataset[sample])
        loss=loss/batch_size
        grads = tape.gradient(loss, params)
        optimizer.apply_gradients(zip([grads], [params]))
    return loss

steps_for_epoch=math.ceil(train_size/batch_size)
boundaries = [steps_for_epoch*3, steps_for_epoch*5, steps_for_epoch*8, steps_for_epoch*12, steps_for_epoch*15, steps_for_epoch*18]
values = [0.4, 0.2, 0.08, 0.04, 0.01, 0.005, 0.001]
learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

train=dataset[0:train_size]
trained_params=np.zeros((nepochs,n_params), dtype=float)

print("Trained parameters will be saved in: ", filename)
print("Start training")
for epoch in range(nepochs):
    tf.random.shuffle(train)
    for i in range(steps_for_epoch):
        loss=train_step(batch_size, encoder, params, train[i*batch_size: (i+1)*batch_size])
    trained_params[epoch]=params.numpy()
    print("Epoch: %d  Loss: %f" % (epoch+1,loss))

np.save(filename, trained_params[-1])