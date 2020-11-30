import json
import numpy as np
from scipy.linalg import solve


def extract_data(filename):
    with open(filename,"r") as r:
        r = r.read()
        raw = json.loads(r)
    return {k: np.array(v) for k, v in raw.items()}


def find_beta(filename):
    """Finds beta from state data."""
    states = extract_data(filename)
    vdict = {k: np.sqrt(v[0] ** 2 + v[1] ** 2) for k, v in states.items()}
    refer_A = np.matrix([[1, 1, 1, 1],
                        [1, 1, -1, -1],
                        [1, -1, 1, -1],
                        [1, -1, -1, 1]])
    refer_B = np.array([vdict["00"], vdict["01"], vdict["10"], vdict["11"]])
    beta = solve(refer_A, refer_B)
    return np.array(beta).flatten()


def cholesky(inmatrix):
    T_linear = np.zeros((4,4), dtype = complex)
    det_linear = np.linalg.det(inmatrix)
    M11 = np.copy(inmatrix)
    M11 = np.delete(M11, 0, 0)
    M11 = np.delete(M11, 0, 1)
    M11 = np.linalg.det(M11)
    M12 = np.copy(inmatrix)
    M12 = np.delete(M12, 0, 0)
    M12 = np.delete(M12, 1, 1)
    M12 = np.linalg.det(M12)
    M1122 = np.copy(inmatrix)
    M1122 = np.delete(M1122, [0,1] , 0)
    M1122 = np.delete(M1122, [0,1] , 1)
    M1122 = np.linalg.det(M1122)
    M1223 = np.copy(inmatrix)
    M1223 = np.delete(M1223, [0,1] , 0)
    M1223 = np.delete(M1223, [1,2] , 1)
    M1223 = np.linalg.det(M1223)
    M1123 = np.copy(inmatrix)
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


def to_vector(x):
    """Transforms Cholesky matrix to real vector."""
    diag = np.diag(x).real
    tril = x[np.tril_indices(len(x), k=-1)]
    return np.concatenate([diag, tril.real, tril.imag])


def from_vector(x):
    """Inverse of ``to_vector``."""
    n = int(np.sqrt(len(x)))
    k = (len(x) - n) // 2

    idx = np.tril_indices(n, k=-1)
    m_re = np.diag(x[:n])
    m_re[idx] = x[n:-k]
    m_im = np.zeros_like(m_re)
    m_im[idx] = x[-k:]
    return np.matrix(m_re + 1j * m_im)


def MLE(x, amp, gate):
    T = from_vector(x)
    rho = T.H*T/np.trace(T.H*T)
    L = 0
    for i in range(len(amp)):
        M = np.trace(rho * gate[i])
        L += 0.5*(amp[i] - M)**2/M
    return abs(L)


class Matrices:
    I = np.matrix([[1, 0], [0, 1]], dtype = complex)
    X = np.matrix([[0, 1], [1, 0]], dtype = complex)
    Y = np.matrix([[0, -1j], [1j, 0]], dtype = complex)
    Z = np.matrix([[1, 0], [0, -1]], dtype = complex)

    g_I = np.matrix([[1, 0], [0, 1]], dtype = complex)
    g_X_p = np.matrix([[0, -1j], [-1j, 0]], dtype = complex)
    g_X_p2 = np.matrix([[1, -1j], [-1j, 1]], dtype = complex) / np.sqrt(2)
    g_X_mp2 = np.matrix([[1, 1j], [1j, 1]], dtype = complex) / np.sqrt(2)
    g_Y_p = np.matrix([[0, -1], [1, 0]], dtype = complex)
    g_Y_p2 = np.matrix([[1, -1], [1, 1]], dtype = complex) / np.sqrt(2)
    g_Y_mp2 = np.matrix([[1, 1], [-1, 1]], dtype = complex) / np.sqrt(2)
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

    rho_th = [np.kron(g_I*qb_a, g_I*qb_b)*np.kron(g_I*qb_a, g_I*qb_b).H,
              np.kron(g_I*qb_a, g_X_p*qb_b)*np.kron(g_I*qb_a, g_X_p*qb_b).H,
              np.kron(g_X_p*qb_a, g_I*qb_b)*np.kron(g_X_p*qb_a, g_I*qb_b).H,
              np.kron(g_X_p*qb_a, g_X_p*qb_b)*np.kron(g_X_p*qb_a, g_X_p*qb_b).H,
              g_swipht*np.kron(g_H*qb_a, g_I*qb_b)*(g_swipht*np.kron(g_H*qb_a, g_I*qb_b)).H,
              np.kron(g_I*qb_a, g_H*qb_b)*np.kron(g_I*qb_a, g_H*qb_b).H,
              np.kron(g_H*qb_a, g_I*qb_b)*np.kron(g_H*qb_a, g_I*qb_b).H,
              g_cnot*np.kron(g_I*qb_a, g_X_p2*qb_b)*(g_cnot*np.kron(g_I*qb_a, g_X_p2*qb_b)).H]

    def rho_th_plot(self, index):
        return np.array(self.rho_th[index])

    def gate(self, beta):
        I = np.matrix([[1, 0], [0, 1]], dtype = complex)
        X = np.matrix([[0, 1], [1, 0]], dtype = complex)
        Y = np.matrix([[0, -1j], [1j, 0]], dtype = complex)
        Z = np.matrix([[1, 0], [0, -1]], dtype = complex)
        return [
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


matrices = Matrices()
