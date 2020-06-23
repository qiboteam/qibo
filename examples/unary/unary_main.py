import numpy as np
from qibo import gates
from qibo.models import Circuit
import matplotlib.pyplot as plt
import unary_functions as un
import aux_functions as aux
from scipy.integrate import trapz

S0 = 2
K = 1.9
sig = 0.4
r = 0.05
T = 0.1
data = (S0, sig, r, T, K)

shots = 1000000
bins = 16

M = 10

circuit, (values, pdf) = un.load_quantum_sim(bins, S0, sig, r, T)
prob_sim = un.run_quantum_sim(bins, circuit, shots)

un.paint_prob_distribution(bins, prob_sim, S0, sig, r, T)

a_s, error_s = un.amplitude_estimation(bins, M, data)

un.paint_AE(a_s, error_s, bins, M, data)