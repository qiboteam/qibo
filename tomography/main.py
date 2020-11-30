# -*- coding: utf-8 -*-
import numpy as np
import copy
import utils
from scipy.linalg import sqrtm
from scipy.optimize import minimize
from scipy.linalg import solve


#%% Extract state data
beta = utils.find_beta("states_181120.json")
gate = np.array(utils.matrices.gate(beta))

# Extract tomography amplitudes
index = 3
filename = ["tomo_181120-00.json",
            "tomo_181120-01.json",
            "tomo_181120-10.json",
            "tomo_181120-11.json"]
            #"tomo_181120-bell.json"
            #"tomo_181120-hadamard-tunable.json",
            #"tomo_181120-hadamard-fixed.json",
            #"tomo_181120-bell_beta.json"
data = utils.extract_data(filename[index])
amp = np.sqrt([v[0] ** 2 + v[1] ** 2 for v in data.values()])


#%% Linear calculation method
dtype = complex
rho_linear = np.zeros((4,4), dtype=dtype)


#########################
# [(0, 0), (1, 1), (2, 2), (3, 3)]
# [(0, 0), (1, 1), (2, 2), (3, 3)]
set1 = [0, 1, 2, 15]
set1a = (0, 1, 2, 3)
set1b = (0, 1, 2, 3)

set1_A = gate[set1][:, set1a, set1b]
set1_B = amp[set1].astype(dtype) # shape (4,)
rho_linear[set1b, set1a] = solve(set1_A, set1_B) # shape (4,)


#########################
# [(1, 0), (0, 1), (3, 2), (2, 3)]
# [(0, 1), (1, 0), (2, 3), (3, 2)]
set2 = [11, 12, 13, 14]
set2a = (1, 0, 3, 2)
set2b = (0, 1, 2, 3)

set2_A = gate[set2][:, set2a, set2b]
set2_B = (amp[set2].astype(dtype) -
          np.einsum("ik,k->i", gate[set2][:, set1a, set1b], rho_linear[set1b, set1a]))
rho_linear[set2b, set2a] = solve(set2_A, set2_B)

#########################
# [(2, 0), (3, 1), (0, 2), (1, 3)]
# [(0, 2), (1, 3), (2, 0), (3, 1)]
set3 = [3, 6, 7, 10]
set3a = (2, 3, 0, 1)
set3b = (0, 1, 2, 3)

set3_A = gate[set3][:, set3a, set3b]
set3_B = (amp[set3].astype(dtype)
          - np.einsum("ik,k->i", gate[set3][:, set1a, set1b], rho_linear[set1b, set1a])
          - np.einsum("ik,k->i", gate[set3][:, set2a, set2b], rho_linear[set2b, set2a]))
rho_linear[set3b, set3a] = solve(set3_A, set3_B)


#########################
set4 = [4, 5, 8, 9]
set4a = (3, 2, 1, 0)
set4b = (0, 1, 2, 3)
set4_A = gate[set4][:, set4a, set4b]
set4_B = (amp[set4].astype(dtype)
          - np.einsum("ik,k->i", gate[set4][:, set1a, set1b], rho_linear[set1b, set1a])
          - np.einsum("ik,k->i", gate[set4][:, set2a, set2b], rho_linear[set2b, set2a])
          - np.einsum("ik,k->i", gate[set4][:, set3a, set3b], rho_linear[set3b, set3a]))
rho_linear[set4b, set4a] = solve(set4_A, set4_B)


#%% Maximum likelihood estimation
T_linear = utils.cholesky(rho_linear)
t_guess = utils.to_vector(T_linear)
res = minimize(utils.MLE, t_guess , args = (amp, gate), tol = 1e-9)
print("Convergence:", res.success)
T = utils.from_vector(res.x)
rho_fit = np.array(T.H*T/np.trace(T.H*T))


r = sqrtm(utils.matrices.rho_th_plot(index))
fidelity = abs(np.trace(sqrtm(r@rho_fit@r)))*100

np.testing.assert_allclose(rho_fit, np.load("rho_fit.npy"))
print("Check passed")
