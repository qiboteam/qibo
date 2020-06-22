import numpy as np
from qibo import gates
from qibo.models import Circuit
import matplotlib.pyplot as plt
import unary_functions as un
import aux_functions as aux

S0 = 2
K = 1.9
sig = 0.4
r = 0.05
T = 0.1
data = (S0, sig, r, T, K)

shots = 100000
bins = 10

circuit, (values, pdf) = un.load_quantum_sim(bins, S0, sig, r, T)
prob_sim = un.run_quantum_sim(bins, circuit, shots)

mu = (r - 0.5 * sig ** 2) * T + np.log(S0)
mean = np.exp(
        mu + 0.5 * T * sig ** 2)
variance = (np.exp(T * sig ** 2) - 1) * np.exp(2 * mu + T * sig ** 2)

S = np.linspace(max(mean-3*np.sqrt(variance),0), mean+3*np.sqrt(variance), bins) #Generate a the exact target distribution to benchmark against quantum results
#lnp = aux.log_normal(Sp, mu, sig, T)
width = (S[1] - S[0]) / 1.2

fig, ax = plt.subplots()
ax.bar(S, prob_sim, width, label='Quantum', alpha=0.8)
plt.ylabel('Probability')
plt.xlabel('Option price')
plt.title('Option price distribution for {} qubits '.format(bins))
ax.legend()
fig.tight_layout()

plt.show()

circuit, S = un.load_payoff_quantum_sim(bins, S0, sig, r, T, K)
qu_payoff_sim = un.run_payoff_quantum_sim(bins, circuit, shots, S, K)
cl_payoff = aux.classical_payoff(S0, sig, r, T, K, samples=1000000)
print(qu_payoff_sim)
print(cl_payoff)

m_s = np.arange(0, 4 + 1, 1)
circuits = [[]]*len(m_s)
for j, m in enumerate(m_s):
    qc = un.load_Q_operator(bins, m, S0, sig, r, T, K)
    circuits[j] = qc

ones_s = [[]]*len(m_s)
zeroes_s = [[]] * len(m_s)
for j, m in enumerate(m_s):
    ones, zeroes = un.run_Q_operator(bins, circuits[j], shots)
    ones_s[j] = int(ones)
    zeroes_s[j] = int(zeroes)
theta_max_s, error_theta_s = aux.get_theta(m_s, ones_s, zeroes_s)
a_s, error_s = np.sin(theta_max_s) ** 2, np.abs(np.sin(2 * theta_max_s) * error_theta_s)
print(a_s)
print(error_s)

un.paint_AE(a_s, error_s, cl_payoff, bins, S0, sig, r, T, K)