'''Test the anomaly detection algorithm (loss function distribution and ROC curve) on 2000 standard (zeros) and 2000 anomalous (ones) samples'''

import numpy as np
import tensorflow as tf
import qibo
from qibo import gates
from qibo.models import Circuit
import matplotlib.pyplot as plt

qibo.set_backend("tensorflow")

'''hyper-parameters must match the ones used for training'''
n_qubits=6
n_layers=6
q_compression=3

'''change to parameters/trained_params.npy for parameters trained with train.py'''
filename="parameters/best_parameters.npy"
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
    
    
def compute_loss_test(encoder, vector):
    reconstructed=encoder(vector)
    #3q compression
    loss=reconstructed.probabilities(qubits=[0])[0] + reconstructed.probabilities(qubits=[1])[0] + reconstructed.probabilities(qubits=[2])[0]
    return loss
    
'''Load data '''
train_size=5000
dataset_np_s=np.load("data/standard_data.npy")
dataset_np_s=dataset_np_s[train_size:]
dataset_s=tf.convert_to_tensor(dataset_np_s)
dataset_np_a=np.load("data/anomalous_data.npy")
dataset_np_a=dataset_np_a[train_size:]
dataset_a=tf.convert_to_tensor(dataset_np_a)

params_np = np.load(filename)
trained_params=tf.convert_to_tensor(params_np)
encoder=make_encoder(n_qubits, n_layers, trained_params, q_compression)
encoder.compile()
#print("Circuit model summary")
#print(encoder.draw())

print("Computing losses...")
loss_s=[]
for i in range(len(dataset_np_s)):
    loss_s.append(compute_loss_test(encoder,dataset_s[i]).numpy())

loss_a=[]
for i in range(len(dataset_np_a)):
    loss_a.append(compute_loss_test(encoder,dataset_a[i]).numpy())

#np.save("results/losses_standard_data",loss_s)
#np.save("results/losses_anomalous_data",loss_a)

'''Loss distribution graph '''
plt.hist(loss_a,bins=60,histtype="step",color="red",label="Anomalous data")
plt.hist(loss_s,bins=60,histtype="step",color="blue",label="Standard data")
plt.ylabel('Number of images')
plt.xlabel('Loss value')
plt.title("Loss function distribution (MNIST dataset)")
plt.legend()
plt.savefig("results/loss_distribution.png")
plt.close()

'''compute ROC curve'''
max1=np.amax(loss_s)
max2=np.amax(loss_a)
ma=max(max1,max2)
min1=np.amin(loss_s)
min2=np.amin(loss_a)
mi=min(min1,min2)

tot_neg=len(loss_s)
tot_pos=len(loss_a)

n_step=100.
n_step_int=100
step=(ma-mi)/n_step
fpr=[]
tpr=[]
for i in range(n_step_int):
  treshold=i*step+mi
  c=0
  for j in range(tot_neg):
    if loss_s[j]>treshold:
      c+=1
  false_positive=c/float(tot_neg)
  fpr.append(false_positive)
  c=0
  for j in range(tot_pos):
    if loss_a[j]>treshold:
      c+=1
  true_positive=c/float(tot_pos)
  tpr.append(true_positive)


plt.title("Receiver Operating Characteristic")
plt.plot(fpr, tpr)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("results/ROC.png")