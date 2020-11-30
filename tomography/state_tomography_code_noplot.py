# -*- coding: utf-8 -*-
import numpy as np
import json
import copy
from scipy.linalg import sqrtm
from scipy.linalg import solve
from scipy.optimize import minimize

def extract_data(filename):
    with open(filename,"r") as r:
        r = r.read()
        raw = json.loads(r)
    return raw #np.array(raw) #Get x data (time)


def cholesky(inmatrix):
    T_linear = np.zeros((4,4), dtype = complex)
    det_linear = np.linalg.det(inmatrix)
    M11 = copy.copy(inmatrix)
    M11 = np.delete(M11, 0, 0)
    M11 = np.delete(M11, 0, 1)
    M11 = np.linalg.det(M11)
    M12 = copy.copy(inmatrix)
    M12 = np.delete(M12, 0, 0)
    M12 = np.delete(M12, 1, 1)
    M12 = np.linalg.det(M12)
    M1122 = copy.copy(inmatrix)
    M1122 = np.delete(M1122, [0,1] , 0)
    M1122 = np.delete(M1122, [0,1] , 1)
    M1122 = np.linalg.det(M1122)
    M1223 = copy.copy(inmatrix)
    M1223 = np.delete(M1223, [0,1] , 0)
    M1223 = np.delete(M1223, [1,2] , 1)
    M1223 = np.linalg.det(M1223)
    M1123 = copy.copy(inmatrix)
    M1123 = np.delete(M1123, [0,1] , 0)
    M1123 = np.delete(M1123, [0,2] , 1)
    M1123 = np.linalg.det(M1123)
    T_linear[0,0] = np.sqrt(det_linear/M11)
    T_linear[1,0] = M12/np.sqrt(M11*M1122)
    T_linear[1,1] = np.sqrt(M11/M1122)
    T_linear[2,0] = M1223/np.sqrt(inmatrix[3,3])/np.sqrt(M1122)
    T_linear[2,1] = M1123/np.sqrt(inmatrix[3,3])/np.sqrt(M1122)
    T_linear[2,2] = np.sqrt(M1122)/np.sqrt(inmatrix[3,3])
    T_linear[3,0] = inmatrix[3,0]/np.sqrt(inmatrix[3,3])
    T_linear[3,1] = inmatrix[3,1]/np.sqrt(inmatrix[3,3])
    T_linear[3,2] = inmatrix[3,2]/np.sqrt(inmatrix[3,3])
    T_linear[3,3] = np.sqrt(inmatrix[3,3])
    return T_linear

def MLE(x, amp, gate):
    T = np.matrix([[x[0], 0 , 0, 0],
                   [x[4] + 1j*x[10], x[1], 0, 0],
                   [x[5] + 1j*x[11], x[6] + 1j*x[12], x[2], 0],
                   [x[7] + 1j*x[13], x[8] + 1j*x[14], x[9] + 1j*x[15], x[3]]])
    rho = T.H*T/np.trace(T.H*T) # Get density matrix
    L = 0
    for i in range(len(amp)):
        M = np.trace(rho * gate[i])
        L += 0.5*(amp[i] - M)**2/M
    return abs(L)


#%% Extract data
refer_raw = extract_data("data/states_181120.json")
refer = copy.copy(refer_raw)
refer_key =[*refer_raw.keys()]
for key in refer_key:
    refer[key] = np.array(refer[key]) # Convert to np
    #refer[key] = np.average(refer[key],0) # Average down

state_00 = refer[refer_key[0]]
state_01 = refer[refer_key[1]]
state_10 = refer[refer_key[2]]
state_11 = refer[refer_key[3]]
v_00 = np.sqrt(state_00[0]**2 + state_00[1]**2)
v_01 = np.sqrt(state_01[0]**2 + state_01[1]**2)
v_10 = np.sqrt(state_10[0]**2 + state_10[1]**2)
v_11 = np.sqrt(state_11[0]**2 + state_11[1]**2)
refer_A = np.matrix([[1, 1, 1, 1],
                    [1, 1, -1, -1],
                    [1, -1, 1, -1],
                    [1, -1, -1, 1]])
refer_B = np.matrix([[v_00],
                    [v_01],
                    [v_10],
                    [v_11]])
beta = solve(refer_A, refer_B)
beta = np.array(beta).flatten()
I = np.matrix([[1, 0],
               [0, 1]], dtype = complex)
X = np.matrix([[0, 1],
               [1, 0]], dtype = complex)
Y = np.matrix([[0, -1j],
               [1j, 0]], dtype = complex)
Z = np.matrix([[1, 0],
               [0, -1]], dtype = complex)

index = 3
filename = ["tomo_181120-00.json",
            "tomo_181120-01.json",
            "tomo_181120-10.json",
            "tomo_181120-11.json",
            "tomo_181120-bell.json",
            "tomo_181120-hadamard-tunable.json",
            "tomo_181120-hadamard-fixed.json",
            "tomo_181120-bell_beta.json"]
raw = extract_data("data/" + filename[index])
data = copy.copy(raw)
meas = [*data.keys()]
for key in meas:
    data[key] = np.array(data[key]) # Convert to np
    #data[key] = np.average(data[key],0) # Average down

#Extract amplitude
amp = np.zeros(len(meas), dtype = complex)
for i in range(len(meas)):
    amp[i] = np.sqrt(data[meas[i]][0]**2 + data[meas[i]][1]**2)

# Create theory gate, state and density matrix
g_I = np.matrix([[1, 0],
                [0, 1]], dtype = complex)
g_X_p = np.matrix([[0, -1j],
                  [-1j, 0]], dtype = complex)
g_X_p2 = np.matrix([[1/np.sqrt(2), -1j/np.sqrt(2)],
                   [-1j/np.sqrt(2), 1/np.sqrt(2)]], dtype = complex)
g_X_mp2 = np.matrix([[1/np.sqrt(2), 1j/np.sqrt(2)],
                    [1j/np.sqrt(2), 1/np.sqrt(2)]], dtype = complex)
g_Y_p = np.matrix([[0, -1],
                  [1, 0]], dtype = complex)
g_Y_p2 = np.matrix([[1/np.sqrt(2), -1/np.sqrt(2)],
                   [1/np.sqrt(2), 1/np.sqrt(2)]], dtype = complex)
g_Y_mp2 = np.matrix([[1/np.sqrt(2), 1/np.sqrt(2)],
                    [-1/np.sqrt(2), 1/np.sqrt(2)]], dtype = complex)
g_vZ_p2 = np.matrix([[np.exp(-1j*np.pi/4), 0],
                    [0, np.exp(1j*np.pi/4)]], dtype = complex)
g_H = np.matrix([[-1j/np.sqrt(2), -1j/np.sqrt(2)],
                 [-1j/np.sqrt(2), 1j/np.sqrt(2)]], dtype = complex)
qb_a = np.matrix([[1],
                  [0]], dtype = complex)
qb_b = np.matrix([[1],
                  [0]], dtype = complex)

g_cnot = np.matrix([[1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0]], dtype = complex)

g_swipht = 1j*0.4577*np.kron(I,I) + 0.5*np.kron(I,X) - 1j*0.4577*np.kron(Z,I) + 0.5*np.kron(Z,X) + 0.2012*np.kron(I,Z) - 0.2012*np.kron(Z,Z)
# Create theory density matrix
rho_th = [np.kron(g_I*qb_a, g_I*qb_b)*np.kron(g_I*qb_a, g_I*qb_b).H,
          np.kron(g_I*qb_a, g_X_p*qb_b)*np.kron(g_I*qb_a, g_X_p*qb_b).H,
          np.kron(g_X_p*qb_a, g_I*qb_b)*np.kron(g_X_p*qb_a, g_I*qb_b).H,
          np.kron(g_X_p*qb_a, g_X_p*qb_b)*np.kron(g_X_p*qb_a, g_X_p*qb_b).H,
          g_swipht*np.kron(g_H*qb_a, g_I*qb_b)*(g_swipht*np.kron(g_H*qb_a, g_I*qb_b)).H,
          np.kron(g_I*qb_a, g_H*qb_b)*np.kron(g_I*qb_a, g_H*qb_b).H,
          np.kron(g_H*qb_a, g_I*qb_b)*np.kron(g_H*qb_a, g_I*qb_b).H,
          g_cnot*np.kron(g_I*qb_a, g_X_p2*qb_b)*(g_cnot*np.kron(g_I*qb_a, g_X_p2*qb_b)).H]
rho_th_plot = np.array(rho_th[index])

gate = [
        beta[0]*np.kron(I,I) + beta[1]*np.kron(Z,I) + beta[2]*np.kron(I,Z) + beta[3]*np.kron(Z,Z),
        beta[0]*np.kron(I,I) - beta[1]*np.kron(Z,I) + beta[2]*np.kron(I,Z) - beta[3]*np.kron(Z,Z),
        beta[0]*np.kron(I,I) + beta[1]*np.kron(Z,I) - beta[2]*np.kron(I,Z) - beta[3]*np.kron(Z,Z),
        beta[0]*np.kron(I,I) + beta[1]*np.kron(Y,I) + beta[2]*np.kron(I,Z) + beta[3]*np.kron(Y,Z),
        beta[0]*np.kron(I,I) + beta[1]*np.kron(Y,I) + beta[2]*np.kron(I,Y) + beta[3]*np.kron(Y,Y),
        beta[0]*np.kron(I,I) + beta[1]*np.kron(Y,I) - beta[2]*np.kron(I,X) - beta[3]*np.kron(Y,X),
        beta[0]*np.kron(I,I) + beta[1]*np.kron(Y,I) - beta[2]*np.kron(I,Z) - beta[3]*np.kron(Y,Z),
        beta[0]*np.kron(I,I) - beta[1]*np.kron(X,I) + beta[2]*np.kron(I,Z) - beta[3]*np.kron(X,Z),
        beta[0]*np.kron(I,I) - beta[1]*np.kron(X,I) + beta[2]*np.kron(I,Y) - beta[3]*np.kron(X,Y),
        beta[0]*np.kron(I,I) - beta[1]*np.kron(X,I) - beta[2]*np.kron(I,X) + beta[3]*np.kron(X,X),
        beta[0]*np.kron(I,I) - beta[1]*np.kron(X,I) - beta[2]*np.kron(I,Z) + beta[3]*np.kron(X,Z),
        beta[0]*np.kron(I,I) + beta[1]*np.kron(Z,I) + beta[2]*np.kron(I,Y) + beta[3]*np.kron(Z,Y),
        beta[0]*np.kron(I,I) - beta[1]*np.kron(Z,I) + beta[2]*np.kron(I,Y) - beta[3]*np.kron(Z,Y),
        beta[0]*np.kron(I,I) + beta[1]*np.kron(Z,I) - beta[2]*np.kron(I,X) - beta[3]*np.kron(Z,X),
        beta[0]*np.kron(I,I) - beta[1]*np.kron(Z,I) - beta[2]*np.kron(I,X) + beta[3]*np.kron(Z,X),
        beta[0]*np.kron(I,I) - beta[1]*np.kron(Z,I) - beta[2]*np.kron(I,Z) + beta[3]*np.kron(Z,Z)
        ]

#%% Linear calculation method
rho_linear = np.zeros((4,4), dtype = complex)
#########################
set1 = [0, 1, 2, 15]
set1_A = np.zeros((4,4), dtype = complex)
set1_B = np.zeros((4,1), dtype = complex)
for i in range(4):
    set1_A[i,0] = gate[set1[i]][0,0]
    set1_A[i,1] = gate[set1[i]][1,1]
    set1_A[i,2] = gate[set1[i]][2,2]
    set1_A[i,3] = gate[set1[i]][3,3]
    set1_B[i,0] = complex(amp[set1[i]])
set1_X = solve(set1_A, set1_B)
rho_linear[0,0] = set1_X[0,0]
rho_linear[1,1] = set1_X[1,0]
rho_linear[2,2] = set1_X[2,0]
rho_linear[3,3] = set1_X[3,0]

#########################
set2 = [11, 12, 13, 14]
set2_A = np.zeros((4,4), dtype = complex)
set2_B = np.zeros((4,1), dtype = complex)
for i in range(4):
    set2_A[i, 0] = gate[set2[i]][1,0]
    set2_A[i, 1] = gate[set2[i]][0,1]
    set2_A[i, 2] = gate[set2[i]][3,2]
    set2_A[i, 3] = gate[set2[i]][2,3]
    minus2_1 = rho_linear[0,0]*gate[set2[i]][0,0] + rho_linear[1,1]*gate[set2[i]][1,1] + rho_linear[2,2]*gate[set2[i]][2,2] + rho_linear[3,3]*gate[set2[i]][3,3]
    set2_B[i, 0] = complex(amp[set2[i]]) - minus2_1
set2_X = solve(set2_A, set2_B)
rho_linear[0,1] = set2_X[0,0]
rho_linear[1,0] = set2_X[1,0]
rho_linear[2,3] = set2_X[2,0]
rho_linear[3,2] = set2_X[3,0]

#########################
set3 = [3, 6, 7, 10]
set3_A = np.zeros((4,4), dtype = complex)
set3_B = np.zeros((4,1), dtype = complex)
for i in range(4):
    set3_A[i,0] = gate[set3[i]][2,0]
    set3_A[i,1] = gate[set3[i]][3,1]
    set3_A[i,2] = gate[set3[i]][0,2]
    set3_A[i,3] = gate[set3[i]][1,3]
    minus3_1 = rho_linear[0,0]*gate[set3[i]][0,0] + rho_linear[1,1]*gate[set3[i]][1,1] + rho_linear[2,2]*gate[set3[i]][2,2] + rho_linear[3,3]*gate[set3[i]][3,3]
    minus3_2 = rho_linear[0,1]*gate[set3[i]][1,0] + rho_linear[1,0]*gate[set3[i]][0,1] + rho_linear[2,3]*gate[set3[i]][3,2] + rho_linear[3,2]*gate[set3[i]][2,3]
    set3_B[i,0] = complex(amp[set3[i]]) - minus3_1 - minus3_2

set3_X = solve(set3_A, set3_B)
rho_linear[0,2] = set3_X[0,0]
rho_linear[1,3] = set3_X[1,0]
rho_linear[2,0] = set3_X[2,0]
rho_linear[3,1] = set3_X[3,0]

#########################
set4 = [4, 5, 8, 9]
set4_A = np.zeros((4,4), dtype = complex)
set4_B = np.zeros((4,1), dtype = complex)
for i in range(4):
    set4_A[i,0] = gate[set4[i]][3,0]
    set4_A[i,1] = gate[set4[i]][2,1]
    set4_A[i,2] = gate[set4[i]][1,2]
    set4_A[i,3] = gate[set4[i]][0,3]
    minus4_1 = rho_linear[0,0]*gate[set4[i]][0,0] + rho_linear[1,1]*gate[set4[i]][1,1] + rho_linear[2,2]*gate[set4[i]][2,2] + rho_linear[3,3]*gate[set4[i]][3,3]
    minus4_2 = rho_linear[0,1]*gate[set4[i]][1,0] + rho_linear[1,0]*gate[set4[i]][0,1] + rho_linear[2,3]*gate[set4[i]][3,2] + rho_linear[3,2]*gate[set4[i]][2,3]
    minus4_3 = rho_linear[0,2]*gate[set4[i]][2,0] + rho_linear[1,3]*gate[set4[i]][3,1] + rho_linear[3,1]*gate[set4[i]][1,3] + rho_linear[2,0]*gate[set4[i]][0,2]
    set4_B[i,0] = complex(amp[set4[i]]) - minus4_1 - minus4_2 - minus4_3

set4_X = solve(set4_A, set4_B)
rho_linear[0,3] = set4_X[0,0]
rho_linear[1,2] = set4_X[1,0]
rho_linear[2,1] = set4_X[2,0]
rho_linear[3,0] = set4_X[3,0]

#%% Maximum likelihood estimation
T_linear = cholesky(rho_linear)
#%%
t_guess = np.zeros(16)
t_guess[0] = T_linear[0,0].real
t_guess[1] = T_linear[1,1].real
t_guess[2] = T_linear[2,2].real
t_guess[3] = T_linear[3,3].real
t_guess[4] = T_linear[1,0].real
t_guess[5] = T_linear[2,0].real
t_guess[6] = T_linear[2,1].real
t_guess[7] = T_linear[3,0].real
t_guess[8] = T_linear[3,1].real
t_guess[9] = T_linear[3,2].real
t_guess[10] = T_linear[1,0].imag
t_guess[11] = T_linear[2,0].imag
t_guess[12] = T_linear[2,1].imag
t_guess[13] = T_linear[3,0].imag
t_guess[14] = T_linear[3,1].imag
t_guess[15] = T_linear[3,2].imag

res = minimize(MLE, t_guess , args = (amp, gate), tol = 1e-9)
print(res.success)
t = res.x
T = np.matrix([[t[0], 0 , 0, 0],
               [t[4] + 1j*t[10], t[1], 0, 0],
               [t[5] + 1j*t[11], t[6] + 1j*t[12], t[2], 0],
               [t[7] + 1j*t[13], t[8] + 1j*t[14], t[9] + 1j*t[15], t[3]]])
rho_fit = np.array(T.H*T/np.trace(T.H*T))
print(rho_fit)
np.save("rho_fit.npy", rho_fit)

r = sqrtm(rho_th_plot)
fidelity = abs(np.trace(sqrtm(r@rho_fit@r)))*100
