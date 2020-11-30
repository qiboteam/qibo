import numpy as np
from scipy.linalg import expm
import utils


def reconstruct(T_linear):
    t_guess = np.zeros(16)
    t_guess[0] = T_linear[0,0].real
    t_guess[1] = T_linear[1,1].real
    t_guess[2] = T_linear[2,2].real
    t_guess[3] = T_linear[3,3].real
    t_guess[4] = T_linear[1,0].real
    t_guess[5] = T_linear[1,0].imag
    t_guess[6] = T_linear[2,1].real
    t_guess[7] = T_linear[2,1].imag
    t_guess[8] = T_linear[3,2].real
    t_guess[9] = T_linear[3,2].imag
    t_guess[10] = T_linear[2,1].real
    t_guess[11] = T_linear[2,1].imag
    t_guess[12] = T_linear[3,2].real
    t_guess[13] = T_linear[3,2].imag
    t_guess[14] = T_linear[3,1].real
    t_guess[15] = T_linear[3,1].imag
    x = t_guess
    T = np.matrix([[x[0], 0 , 0, 0],
                   [x[4] + 1j*x[5], x[1], 0, 0],
                   [x[10] + 1j*x[11], x[6] + 1j*x[7], x[2], 0],
                   [x[14] + 1j*x[15], x[12] + 1j*x[13], x[8] + 1j*x[9], x[3]]])
    return T.H*T / np.trace(T.H*T) # Get density matrix


n = 4

u = np.matrix(expm(1j * np.random.random((n, n))))
d = 10 * np.random.random(4)
rho = u.dot(np.matrix(np.diag(d)).dot(u.H))
rho[np.diag_indices(n)] = rho[np.diag_indices(n)] / np.trace(rho)

np.testing.assert_allclose(rho, rho.H)
print("Rho is Hermitian")


T_linear = utils.cholesky(rho)
#rho_recon = t.H * t / np.trace(t.H * t)
rho_recon = reconstruct(T_linear)
np.testing.assert_allclose(rho_recon, rho_recon.H)
print("Reconstruction is Hermitian")

np.testing.assert_allclose(rho, rho_recon)
