import functions as fun
import aux_functions as aux
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--bins", default=16, type=int)
args = vars(parser.parse_args())


def main(data, bins, M, shots):
    S0, sig, r, T, K = data
    circuit, (values, pdf) = fun.load_quantum_sim(bins, S0, sig, r, T)
    prob_sim = fun.run_quantum_sim(bins, circuit, shots)
    fun.paint_prob_distribution(bins, prob_sim, S0, sig, r, T)
    print('Histogram printed for unary simulation with {} qubits.\n'.format(bins))
    circuit, S = fun.load_payoff_quantum_sim(bins, S0, sig, r, T, K)
    qu_payoff_sim = fun.run_payoff_quantum_sim(bins, circuit, shots, S, K)
    cl_payoff = aux.classical_payoff(S0, sig, r, T, K, samples=1000000)
    error = fun.diff_qu_cl(qu_payoff_sim, cl_payoff)
    print('Exact value of the expected payoff:      {}\n'.format(cl_payoff))
    print('Expected payoff from quantum simulation: {}\n'.format(qu_payoff_sim))
    print('Percentage error: {} %\n'.format(error))
    print('-'*60+'\n')
    print('Amplitude estimation with a total of {} runs.\n'.format(M))
    a_s, error_s = fun.amplitude_estimation(bins, M, data)
    fun.paint_AE(a_s, error_s, bins, M, data)
    print('Amplitude estimation results printed.')


S0 = 2
K = 1.9
sig = 0.4
r = 0.05
T = 0.1
data = (S0, sig, r, T, K)
shots = 1000000
bins = args.get('bins')
M = 10


main(data, bins, M, shots)
