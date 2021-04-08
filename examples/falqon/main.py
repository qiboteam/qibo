import numpy as np
from qibo import models, hamiltonians
# create XXZ Hamiltonian for four qubits
hamiltonian = hamiltonians.XXZ(4)
# create FALQON model for this Hamiltonian
falqon = models.FALQON(hamiltonian)

delta_t = .1
max_layers = 100
best_energy, final_parameters = falqon.minimize(delta_t, max_layers)[:2]

print('The optimal energy found is', best_energy)