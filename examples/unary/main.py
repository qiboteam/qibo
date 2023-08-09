import functions as fun
import aux_functions as aux
import argparse


def main(data, bins, M, shots):
    S0, sig, r, T, K = data
    
    # Create circuit to load the probability distribution
    circuit, (values, pdf) = fun.load_quantum_sim(bins, S0, sig, r, T)

    # Measure the probability distribution
    prob_sim = fun.run_quantum_sim(bins, circuit, shots)

    # Generate the probability distribution plots
    fun.paint_prob_distribution(bins, prob_sim, S0, sig, r, T)
    print('Histogram printed for unary simulation with {} qubits.\n'.format(bins))

    # Create circuit to compute the expected payoff
    circuit, S = fun.load_payoff_quantum_sim(bins, S0, sig, r, T, K)

    # Run the circuit for expected payoff
    qu_payoff_sim = fun.run_payoff_quantum_sim(bins, circuit, shots, S, K)

    # Computing exact classical payoff
    cl_payoff = aux.classical_payoff(S0, sig, r, T, K, samples=1000000)

    # Finding differences between exact value and quantum approximation
    error = fun.diff_qu_cl(qu_payoff_sim, cl_payoff)
    print('Exact value of the expected payoff:      {}\n'.format(cl_payoff))
    print('Expected payoff from quantum simulation: {}\n'.format(qu_payoff_sim))
    print('Percentage error: {} %\n'.format(error))
    print('-'*60+'\n')

    # Applying amplitude estimation
    a_s, error_s = fun.amplitude_estimation(bins, M, data)
    print('Amplitude estimation with a total of {} runs.\n'.format(M))
    fun.paint_AE(a_s, error_s, bins, M, data)
    print('Amplitude estimation result plots generated.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins", default=16, type=int)
    parser.add_argument("--M", default=10, type=int, help='Max AE runs.')
    parser.add_argument("--S0", default=2, type=float, help='Initial value of asset.')
    parser.add_argument("--K", default=1.9, type=float, help='Strike price.')
    parser.add_argument("--sig", default=0.4, type=float, help='Volatility.')
    parser.add_argument("--r", default=0.05, type=float, help='Market rate.')
    parser.add_argument("--T", default=0.1, type=float, help='Maturity date.')
    args = vars(parser.parse_args())
    S0 = args.get('S0')
    K = args.get('K')
    sig = args.get('sig')
    r = args.get('r')
    T = args.get('T')
    data = (S0, sig, r, T, K)
    shots = 1000000
    bins = args.get('bins')
    max_m = args.get('M')
    main(data, bins, max_m, shots)
