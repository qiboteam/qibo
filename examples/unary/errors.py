import numpy as np
import binary as bin
import unary as un

from aux_functions import *
from noise_mapping import *
from time import time
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


"""
This file creates the Python class defining all quantities to perform 
"""

class errors:
    """
    Class to encode all computations performed for a given problem and a given set of errors
    """
    def __init__(self, data, max_gate_error, steps):
        """
        Initialize
        :param data: (S0, sig, r, T, K)
        :param max_gate_error: Maximum single-qubit gate error allowed (recommended: 0.005)
        :param steps: Number of error steps (recommended: 51, 101)
        """
        self.data = data
        self.S0, self.sig, self.r, self.T, self.K = self.data
        self.cl_payoff = classical_payoff(self.S0, self.sig, self.r, self.T, self.K)
        # CNOT error is 2 * gate_error
        # Measurement error is 10 * gate_error
        self.max_gate_error = max_gate_error
        self.steps = steps
        self.error_steps = np.linspace(0, max_gate_error, steps)
        self.list_errors = ['bitflip', 'phaseflip', 'bitphaseflip',
                            'depolarizing', 'thermal', 'measurement']
        try:
            """
            Create folder with results
            """
            os.makedirs(name_folder_data(self.data))
        except:
            pass

    def select_error(self, error_name, measure_error=False, thermal_error=False):
        """
        Function to import errors from noise_mapping.py
        :param error_name: Name of the error
        :param measure_error: Measurement errors included
        :param thermal_error: Thermal relaxation errors included
        :return: Function creating the desired error
        """
        if 'bitflip' in error_name:
            noise = noise_model_bit
            if measure_error:
                error_name += '_m'
            if thermal_error:
                error_name += '_t'
        elif 'phaseflip' in error_name:
            noise = noise_model_phase
            if measure_error:
                error_name += '_m'
            if thermal_error:
                error_name += '_t'
        elif 'bitphaseflip' in error_name:
            noise = noise_model_bitphase
            if measure_error:
                error_name += '_m'
            if thermal_error:
                error_name += '_t'
        elif 'depolarizing' in error_name:
            noise = noise_model_depolarizing
            if measure_error:
                error_name += '_m'
            if thermal_error:
                error_name += '_t'
        elif error_name == 'thermal':
            noise = noise_model_thermal
            if measure_error:
                error_name += '_m'
        elif error_name == 'measurement':
            noise = noise_model_measure
        else:
            raise NameError('Error not indexed')

        return noise

    def change_name(self, error_name, measure_error, thermal_error):
        """
        Auxiliary function to maintain coherence between names. Useful for saving data.
        :param error_name: Name of the error
        :param measure_error: Measurement errors included
        :param thermal_error: Thermal relaxation errors included
        :return: Updated name of error
        """
        if measure_error and '_m' not in error_name:
            error_name += '_m'
        if thermal_error and '_t' not in error_name:
            error_name += '_t'

        return error_name


    def compute_save_errors_binary(self, bins, error_name, repeats, measure_error=False, thermal_error=False, shots=10000):
        """
        Function to compute and save errors in the binary case without amplitude estimation, for noisy circuits
        :param bins: Number of bins desired
        :param error_name: Name of the desired error
        :param repeats: Number of repetitions of the experiment
        :param measure_error: Measurement errors included
        :param thermal_error: Thermal relaxation errors included
        :param shots: Number of shots for the sampling
        :return: None. Saves data in files.
        """
        qubits = int(np.log2(bins))
        results = np.zeros((len(self.error_steps), repeats))
        noise = self.select_error(error_name, measure_error=measure_error, thermal_error=thermal_error)
        error_name = self.change_name(error_name, measure_error, thermal_error)

        for i, error in enumerate(self.error_steps):
            noise_model = noise(error, measure=measure_error, thermal=thermal_error)
            basis_gates = noise_model.basis_gates
            c, k, high, low, qc = bin.load_payoff_quantum_sim(qubits, self.S0, self.sig, self.r, self.T, self.K)
            for r in range(repeats):
                qu_payoff_sim = bin.run_payoff_quantum_sim(qubits, c, k, high, low, qc, shots, basis_gates, noise_model)
                diff = bin.diff_qu_cl(qu_payoff_sim, self.cl_payoff)
                results[i, r] = diff
        try:
            os.makedirs(name_folder_data(self.data) + '/%s_bins/binary/'%bins)
        except:
            pass
        np.savetxt(name_folder_data(self.data) + '/%s_bins/binary/'%bins + error_name + '_gate(%s)_steps(%s)_repeats(%s).npz'%(self.max_gate_error, self.steps, repeats), results)


    def compute_save_errors_unary(self, bins, error_name, repeats, measure_error=False, thermal_error=False, shots=10000):
        """
        Function to compute and save errors in the unary case without amplitude estimation, for noisy circuits
        :param bins: Number of bins desired
        :param error_name: Name of the desired error
        :param repeats: Number of repetitions of the experiment
        :param measure_error: Measurement errors included
        :param thermal_error: Thermal relaxation errors included
        :param shots: Number of shots for the sampling
        :return: None. Saves data in files.
        """
        qubits = bins
        results = np.zeros((len(self.error_steps), repeats))
        noise = self.select_error(error_name, measure_error=measure_error, thermal_error=thermal_error)
        error_name = self.change_name(error_name, measure_error, thermal_error)

        for i, error in enumerate(self.error_steps):
            noise_model = noise(error, measure=measure_error, thermal=thermal_error)
            basis_gates = noise_model.basis_gates
            qc, S = un.load_payoff_quantum_sim(qubits, self.S0, self.sig, self.r, self.T, self.K)
            for r in range(repeats):
                qu_payoff_sim = un.run_payoff_quantum_sim(qubits, qc, shots, S, self.K, basis_gates, noise_model)
                diff = un.diff_qu_cl(qu_payoff_sim, self.cl_payoff)
                results[i, r] = diff
        try:
            os.makedirs(name_folder_data(self.data) + '/%s_bins/unary/' % bins)
        except:
            pass
        np.savetxt(name_folder_data(self.data) + '/%s_bins/unary/'%bins +
                   error_name + '_gate(%s)_steps(%s)_repeats(%s).npz'%(self.max_gate_error, self.steps, repeats), results)

    def paint_errors(self, bins, error_name, repeats, bounds=0.15, measure_error=False, thermal_error=False):
        """
        Function to paint errors in the binary and unary case without amplitude estimation, for noisy circuits
        :param bins: Number of bins desired
        :param error_name: Name of the desired error
        :param repeats: Number of repetitions of the experiment
        :param bounds: Proportion of lowest and highest instances to be excluded from the painting
        :param measure_error: Measurement errors included
        :param thermal_error: Thermal relaxation errors included
        :return: None. Saves images in files.
        """
        error_name = self.change_name(error_name, measure_error, thermal_error)

        matrix_unary = np.loadtxt(name_folder_data(self.data) + '/%s_bins/unary/'%bins
                            + error_name + '_gate(%s)_steps(%s)_repeats(%s).npz'%(self.max_gate_error, self.steps, repeats))
        matrix_unary = np.sort(matrix_unary, axis=1)
        mins_unary = matrix_unary[:, int(bounds * (repeats))]
        maxs_unary = matrix_unary[:, int(-(bounds) * (repeats)-1)]
        means_unary = np.mean(matrix_unary, axis=1)
        matrix_binary = np.loadtxt(name_folder_data(self.data) + '/%s_bins/binary/' % bins
                                  + error_name + '_gate(%s)_steps(%s)_repeats(%s).npz' % (
                                  self.max_gate_error, self.steps, repeats))
        matrix_binary = np.sort(matrix_binary, axis=1)
        mins_binary = matrix_binary[:, int(bounds * (repeats))]
        maxs_binary = matrix_binary[:, int(-(bounds) * (repeats)-1)]
        means_binary = np.mean(matrix_binary, axis=1)

        fig, ax = plt.subplots()
        ax.scatter(100 * self.error_steps, means_unary, s=20, color='C0', label='unary', marker='x')
        ax.scatter(100 * self.error_steps, means_binary, s=20, color='C1', label='binary', marker='+')
        ax.fill_between(100 * self.error_steps, maxs_unary, mins_unary, alpha=0.2, facecolor='C0')
        ax.fill_between(100 * self.error_steps, maxs_binary, mins_binary, alpha=0.2, facecolor='C1')
        ax.set(ylim=[0, 50])
        plt.ylabel('percentage off classical value (%)')
        plt.xlabel('single-qubit gate error (%)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(name_folder_data(self.data) + '/%s_bins/' % bins
                                  + error_name + '_gate(%s)_steps(%s)_repeats(%s).pdf' % (
                                  self.max_gate_error, self.steps, repeats))


    def compute_save_outcomes_binary(self, bins, error_name, error_value, repeats, measure_error=False, thermal_error=False, shots=10000):
        """
        Function to compute and save the probability distribution in the binary case without amplitude estimation, for noisy circuits
        :param bins: Number of bins desired
        :param error_name: Name of the desired error
        :param repeats: Number of repetitions of the experiment
        :param measure_error: Measurement errors included
        :param thermal_error: Thermal relaxation errors included
        :param shots: Number of shots for the sampling
        :return: None. Saves data in files.
        """
        error_name = self.change_name(error_name, measure_error, thermal_error)
        qubits = int(np.log2(bins))
        try:
            os.makedirs(name_folder_data(self.data) + '/%s_bins/binary/probs' % bins)
        except:
            pass
        noise = self.select_error(error_name, measure_error=measure_error, thermal_error=thermal_error)
        noise_model = noise(error_value, measure=measure_error, thermal=thermal_error)
        basis_gates = noise_model.basis_gates
        qc, (values, pdf) = bin.load_quantum_sim(qubits, self.S0, self.sig, self.r, self.T)[:2]
        probs = np.zeros((len(values), repeats))
        for r in range(repeats):
            probs[:, r] = bin.run_quantum_sim(qubits, qc, shots, basis_gates, noise_model)

        np.savetxt(name_folder_data(self.data) + '/%s_bins/binary/probs/'%bins +
                   error_name + '_gate(%s)_repeats(%s).npz'%(error_value, repeats), probs)

        return probs

    def compute_save_outcomes_unary(self, bins, error_name, error_value, repeats, measure_error=False, thermal_error=False, shots=10000):
        """
        Function to compute and save the probability distribution in the unary case without amplitude estimation, for noisy circuits
        :param bins: Number of bins desired
        :param error_name: Name of the desired error
        :param repeats: Number of repetitions of the experiment
        :param measure_error: Measurement errors included
        :param thermal_error: Thermal relaxation errors included
        :param shots: Number of shots for the sampling
        :return: None. Saves data in files.
        """
        error_name = self.change_name(error_name, measure_error, thermal_error)
        qubits = bins
        try:
            os.makedirs(name_folder_data(self.data) + '/%s_bins/unary/probs' % bins)
        except:
            pass
        noise = self.select_error(error_name, measure_error=measure_error, thermal_error=thermal_error)
        noise_model = noise(error_value, measure=measure_error, thermal=thermal_error)
        basis_gates = noise_model.basis_gates
        qc, (values, pdf) = un.load_quantum_sim(qubits, self.S0, self.sig, self.r, self.T)[:2]
        probs = np.zeros((len(values), repeats))
        for r in range(repeats):
            res = un.run_quantum_sim(qubits, qc, shots, basis_gates, noise_model)
            probs[:, r] = [res[2**i] for i in range(qubits)]
            probs[:, r] /= np.sum(probs[:, r])

        np.savetxt(name_folder_data(self.data) + '/%s_bins/unary/probs/'%bins +
                    error_name + '_gate(%s)_repeats(%s).npz'%(error_value, repeats), probs)

        return probs

    def paint_outcomes(self, bins, error_name, error_value, repeats, bounds=0.15, measure_error=False, thermal_error=False):
        """
        Function to paint probability distributions in the binary and unary case without amplitude estimation, for noisy circuits
        :param bins: Number of bins desired
        :param error_name: Name of the desired error
        :param repeats: Number of repetitions of the experiment
        :param bounds: Proportion of lowest and highest instances to be excluded from the painting
        :param measure_error: Measurement errors included
        :param thermal_error: Thermal relaxation errors included
        :return: None. Saves images in files.
        """
        error_name = self.change_name(error_name, measure_error, thermal_error)
        matrix_unary = np.loadtxt(name_folder_data(self.data) + '/%s_bins/unary/probs/'%bins +
                            error_name + '_gate(%s)_repeats(%s).npz'%(error_value, repeats))
        matrix_unary = np.sort(matrix_unary, axis=1)
        mins_unary = matrix_unary[:, int(bounds * (repeats))]
        maxs_unary = matrix_unary[:, int(-(bounds) * (repeats)-1)]
        means_unary = np.mean(matrix_unary, axis=1)
        matrix_binary = np.loadtxt(name_folder_data(self.data) + '/%s_bins/binary/probs/'%bins +
                            error_name + '_gate(%s)_repeats(%s).npz'%(error_value, repeats))
        matrix_binary = np.sort(matrix_binary, axis=1)
        mins_binary = matrix_binary[:, int(bounds * (repeats))]
        maxs_binary = matrix_binary[:, int(-(bounds) * (repeats)-1)]
        means_binary = np.mean(matrix_binary, axis=1)

        qc, (values, pdf) = un.load_quantum_sim(bins, self.S0, self.sig, self.r, self.T)[:2]

        width = (values[1] - values[0]) / 1.3
        exact_values = np.linspace(np.min(values), np.max(values), bins * 100)
        mu = (self.r - 0.5 * self.sig ** 2) * self.T + np.log(self.S0)
        exact_pdf = log_normal(exact_values, mu, self.sig * np.sqrt(self.T))
        exact_pdf = exact_pdf * pdf[0] / exact_pdf[0]
        fig, ax = plt.subplots()

        ax.bar(values + width / 4, maxs_binary, width / 2, alpha=0.3, color='C1')
        ax.bar(values + width / 4, mins_binary, width / 2, alpha=.75, color='C1', label='binary')
        ax.bar(values - width / 4, maxs_unary, width / 2, alpha=0.3, color='C0')
        ax.bar(values - width / 4, mins_unary, width / 2, alpha=.75, color='C0', label='unary')
        ax.plot(exact_values, exact_pdf, color='black', label='PDF')
        ax.scatter(values - width / 4, means_unary, s=20, color='C0', marker='x', zorder=10)
        ax.scatter(values + width / 4, means_binary, s=20, color='C1', marker='x', zorder=10)
        ax.scatter(values, pdf, s=1250, color='black', marker='_', zorder=9)
        plt.ylabel('Probability')
        plt.xlabel('Option price')
        ax.legend()
        fig.tight_layout()
        fig.savefig(name_folder_data(self.data) + '/%s_bins/' % bins
                    + error_name + '_gate(%s)_repeats(%s)_probs.pdf' % (
                        error_value, repeats))

    def compute_save_KL_binary(self, bins, error_name, repeats, measure_error=False, thermal_error=False, shots=10000):
        """
        Function to compute and save Kullback-Leibler divergences of probability distributions in the binary
            case without amplitude estimation, for noisy circuits
        :param bins: Number of bins desired
        :param error_name: Name of the desired error
        :param repeats: Number of repetitions of the experiment
        :param measure_error: Measurement errors included
        :param thermal_error: Thermal relaxation errors included
        :param shots: Number of shots for the sampling
        :return: None. Saves data in files.
        """
        error_name = self.change_name(error_name, measure_error, thermal_error)

        divergences = np.zeros((len(self.error_steps), repeats))
        for i, error in enumerate(self.error_steps):
            try:
                probs = np.loadtxt(name_folder_data(self.data) + '/%s_bins/binary/probs/' % bins +
                           error_name + '_gate(%s)_repeats(%s).npz' % (error, repeats))

            except:
                probs = self.compute_save_outcomes_binary(bins, error_name, error, repeats,
                                                          measure_error=measure_error, thermal_error=thermal_error, shots=shots)

            qc, (values, pdf), (mu, mean, variance) = un.load_quantum_sim(bins, self.S0, self.sig, self.r, self.T)
            for j in range(probs.shape[1]):
                divergences[i, j] = KL(probs[:, j], pdf)

            np.savetxt(name_folder_data(self.data) + '/%s_bins/binary/probs/' % bins +
                error_name + '_gate(%s)_repeats(%s)_div.npz' % (self.max_gate_error, repeats), divergences)

    def compute_save_KL_unary(self, bins, error_name, repeats, measure_error=False, thermal_error=False, shots=10000):
        """
        Function to compute and save Kullback-Leibler divergences of probability distributions in the unary
            case without amplitude estimation, for noisy circuits
        :param bins: Number of bins desired
        :param error_name: Name of the desired error:param repeats: Number of repetitions of the experiment
        :param repeats: Number of repetitions of the experiment
        :param measure_error: Measurement errors included
        :param thermal_error: Thermal relaxation errors included
        :param shots: Number of shots for the sampling
        :return: None. Saves data in files.
        """
        error_name = self.change_name(error_name, measure_error, thermal_error)

        divergences = np.zeros((len(self.error_steps), repeats))
        for i, error in enumerate(self.error_steps):
            try:

                probs = np.loadtxt(name_folder_data(self.data) + '/%s_bins/unary/probs/' % bins +
                                   error_name + '_gate(%s)_repeats(%s).npz' % (error, repeats))
            except:
                probs = self.compute_save_outcomes_unary(bins, error_name, error, repeats,
                                                          measure_error=measure_error, thermal_error=thermal_error, shots=shots)

            qc, (values, pdf) = un.load_quantum_sim(bins, self.S0, self.sig, self.r, self.T)[:2]
            for j in range(probs.shape[1]):
                divergences[i, j] = KL(probs[:, j], pdf)

            np.savetxt(name_folder_data(self.data) + '/%s_bins/unary/probs/' % bins +
                   error_name + '_gate(%s)_repeats(%s)_div.npz' % (self.max_gate_error, repeats), divergences)

    def paint_divergences(self, bins, error_name, repeats, bounds=0.15, measure_error=False, thermal_error=False):
        """
        Function to paint Kullback-Leibler divergences in the binary and unary case without amplitude estimation, for noisy circuits
        :param bins: Number of bins desired
        :param error_name: Name of the desired error
        :param repeats: Number of repetitions of the experiment
        :param bounds: Proportion of lowest and highest instances to be excluded from the painting
        :param measure_error: Measurement errors included
        :param thermal_error: Thermal relaxation errors included
        :return: None. Saves images in files.
        """
        error_name = self.change_name(error_name, measure_error, thermal_error)

        matrix_unary = np.loadtxt(name_folder_data(self.data) + '/%s_bins/unary/probs/'%bins
                            + error_name + '_gate(%s)_repeats(%s)_div.npz' % (self.max_gate_error, repeats))
        matrix_binary = np.loadtxt(name_folder_data(self.data) + '/%s_bins/binary/probs/' % bins
                                         + error_name + '_gate(%s)_repeats(%s)_div.npz' % (self.max_gate_error, repeats))
        matrix_unary = np.sort(matrix_unary, axis=1)
        mins_unary = matrix_unary[:, int(bounds * (repeats))]
        maxs_unary = matrix_unary[:, int(-(bounds) * (repeats)-1)]
        means_unary = np.mean(matrix_unary, axis=1)
        matrix_binary = np.sort(matrix_binary, axis=1)
        mins_binary = matrix_binary[:, int(bounds * (repeats))]
        maxs_binary = matrix_binary[:, int(-(bounds) * (repeats)-1)]
        means_binary = np.mean(matrix_binary, axis=1)

        fig, ax = plt.subplots()
        ax.scatter(100 * self.error_steps, means_unary, s=20, color='C0', label='unary', marker='x')
        ax.scatter(100 * self.error_steps, means_binary, s=20, color='C1', label='binary', marker='+')
        ax.fill_between(100 * self.error_steps, maxs_unary, mins_unary, alpha=0.2, facecolor='C0')
        ax.fill_between(100 * self.error_steps, maxs_binary, mins_binary, alpha=0.2, facecolor='C1')
        plt.ylabel('KL Divergence')
        plt.xlabel('single-qubit gate error (%)')
        plt.yscale('log')
        ax.legend()
        fig.tight_layout()
        fig.savefig(name_folder_data(self.data) + '/%s_bins/' % bins
                                  + error_name + '_gate(%s)_steps(%s)_repeats(%s)_div.pdf' % (
                                  self.max_gate_error, self.steps, repeats))

    def compute_save_errors_binary_amplitude_estimation(self, bins, error_name, repeats, M=4, measure_error=False
                                                        , thermal_error=False, shots=500):
        """
        Function to compute and save errors in the binary case for amplitude estimation, for noisy circuits
        :param bins: Number of bins desired
        :param error_name: Name of the desired error
        :param repeats: Number of repetitions of the experiment
        :param M: Maximum number of applications of Q in one iteration
        :param measure_error: Measurement errors included
        :param thermal_error: Thermal relaxation errors included
        :param shots: Number of shots for the sampling
        :return: None. Saves data in files.
        """
        qubits = int(np.log2(bins))
        error_name = self.change_name(error_name, measure_error, thermal_error)
        noise = self.select_error(error_name, measure_error=measure_error, thermal_error=thermal_error)
        m_s = np.arange(0, M + 1, 1)

        error_payoff = np.empty((len(m_s), self.steps, repeats))
        confidence = np.empty_like(error_payoff)

        for i, error in enumerate(self.error_steps):
            circuits = [[]]*len(m_s)
            noise_model = noise(error, measure=measure_error, thermal=thermal_error)
            basis_gates = noise_model.basis_gates
            for j, m in enumerate(m_s):
                qc = bin.load_Q_operator(qubits, m, self.S0, self.sig, self.r, self.T, self.K)[0]
                circuits[j] = qc

            for r in range(repeats):
                ones_s = [[]]*len(m_s)
                zeroes_s = [[]] * len(m_s)
                for j, m in enumerate(m_s):
                    ones, zeroes = bin.run_Q_operator(circuits[j], shots, basis_gates, noise_model)
                    ones_s[j] = int(ones)
                    zeroes_s[j] = int(zeroes)
                theta_max_s, error_theta_s = get_theta(m_s, ones_s, zeroes_s)
                a_s, error_s = np.sin(theta_max_s) ** 2, np.abs(np.sin(2 * theta_max_s) * error_theta_s)
                error_payoff[:, i, r] = a_s
                confidence[:, i, r] =np.abs(error_s)

        try:
            os.makedirs(name_folder_data(self.data) + '/%s_bins/binary/amplitude_estimation' % bins)
        except:
            pass
        for j, m in enumerate(m_s):
            np.savetxt(name_folder_data(
                self.data) + '/%s_bins/binary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s).npz' % (
                       self.max_gate_error, self.steps, repeats, m), error_payoff[j])
            np.savetxt(name_folder_data(
                self.data) + '/%s_bins/binary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s)_confidence.npz' % (
                       self.max_gate_error, self.steps, repeats, m), confidence[j])

    def compute_save_errors_unary_amplitude_estimation(self, bins, error_name, repeats, M=4, measure_error=False, thermal_error=False,
                                   shots=500):
        """
        Function to compute and save errors in the unary case for amplitude estimation, for noisy circuits
        :param bins: Number of bins desired
        :param error_name: Name of the desired error
        :param repeats: Number of repetitions of the experiment
        :param M: Maximum number of applications of Q in one iteration
        :param measure_error: Measurement errors included
        :param thermal_error: Thermal relaxation errors included
        :param shots: Number of shots for the sampling
        :return: None. Saves data in files.
        """
        qubits = int(bins)
        error_name = self.change_name(error_name, measure_error, thermal_error)
        noise = self.select_error(error_name, measure_error=measure_error, thermal_error=thermal_error)
        m_s = np.arange(0, M + 1, 1)
        error_payoff = np.empty((len(m_s), self.steps, repeats))
        confidence = np.empty_like(error_payoff)

        for i, error in enumerate(self.error_steps):
            circuits = [[]]*len(m_s)
            noise_model = noise(error, measure=measure_error, thermal=thermal_error)
            basis_gates = noise_model.basis_gates
            for j, m in enumerate(m_s):
                qc = un.load_Q_operator(qubits, m, self.S0, self.sig, self.r, self.T, self.K)
                circuits[j] = qc

            for r in range(repeats):
                ones_s = [[]]*len(m_s)
                zeroes_s = [[]] * len(m_s)
                for j, m in enumerate(m_s):
                    ones, zeroes = un.run_Q_operator(circuits[j], shots, basis_gates, noise_model)
                    ones_s[j] = int(ones)
                    zeroes_s[j] = int(zeroes)
                theta_max_s, error_theta_s = get_theta(m_s, ones_s, zeroes_s)
                a_s, error_s = np.sin(theta_max_s) ** 2, np.abs(np.sin(2 * theta_max_s) * error_theta_s)
                error_payoff[:, i, r] = a_s
                confidence[:, i, r] = np.abs(error_s)

        try:
            os.makedirs(name_folder_data(self.data) + '/%s_bins/unary/amplitude_estimation' % bins)
        except:
            pass
        for j, m in enumerate(m_s):
            np.savetxt(name_folder_data(
                self.data) + '/%s_bins/unary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s).npz' % (
                       self.max_gate_error, self.steps, repeats, m), error_payoff[j])
            np.savetxt(name_folder_data(
                self.data) + '/%s_bins/unary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s)_confidence.npz' % (
                       self.max_gate_error, self.steps, repeats, m), confidence[j])

    def error_emplitude_estimation(self, bins, error_name, repeats, M=4, measure_error=False, thermal_error=False,
                                   shots=500, alpha=0.05, u=0, e=0.05):
        """
        Function to paint outcomes and errors for amplitude estimation
        :param bins:
        :param error_name:
        :param repeats:
        :param M:
        :param measure_error:
        :param thermal_error:
        :param shots:
        :param alpha:
        :return: None. Saves data in files.
        """

        (values, pdf) = un.get_pdf(bins, self.S0, self.sig, self.r, self.T)[0]
        a_un = np.sum(pdf[values >= self.K] * (values[values >= self.K] - self.K))
        error_name = self.change_name(error_name, measure_error, thermal_error)

        fig, ax = plt.subplots()
        un_data = np.empty(M+1)
        un_conf = np.empty(M + 1)
        bin_data = np.empty(M + 1)
        bin_conf = np.empty(M + 1)
        for m in range(M+1):
            un_data_ = np.loadtxt(name_folder_data(
                    self.data) + '/%s_bins/unary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s).npz' % (
                           self.max_gate_error, self.steps, repeats, m))[0]
            un_conf_ = np.loadtxt(name_folder_data(
                self.data) + '/%s_bins/unary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s)_confidence.npz' % (
                                      self.max_gate_error, self.steps, repeats, m))[0]
            bin_data_ = np.loadtxt(name_folder_data(
                self.data) + '/%s_bins/binary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s).npz' % (
                                     self.max_gate_error, self.steps, repeats, m))[0]
            bin_conf_ = np.loadtxt(name_folder_data(
                self.data) + '/%s_bins/binary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s)_confidence.npz' % (
                                       self.max_gate_error, self.steps, repeats, m))[0]

            un_data[m], un_conf[m] = errors_experiment(un_data_, un_conf_)
            bin_data[m], bin_conf[m] = errors_experiment(bin_data_, bin_conf_)

        a_bin = errors_experiment(bin_data, bin_conf)[0]
        (values, pdf) = bin.get_pdf(bins, self.S0, self.sig, self.r, self.T)[1]
        high, low = np.max(values), np.min(values)
        k = int(np.floor(bins * (self.K - low) / (high - low)))
        c = (2 * e) ** (1 / (2 * u + 2))
        payoff_bin = (high - low) / bins * (a_bin - .5 + c) * (bins - 1 - k) / (2 * c)
        '''
        a_bin must be calculated in this way because the error allowed in the payoff detracts the precision 
        of Amplitude Estimation
        '''
        un_data *= (np.max(values) - self.K)
        un_conf *= (np.max(values) - self.K)
        bin_data = (high - low) / bins * (bin_data - .5 + c) * (bins - 1 - k) / (2 * c)
        bin_conf = (high - low) / bins * (bin_conf) * (bins - 1 - k) / (2 * c)
        ax.scatter(np.arange(M+1), un_data, c='C0', marker='x', label='unary', zorder=10)
        ax.fill_between(np.arange(M+1), un_data - un_conf, un_data + un_conf, color='C0', alpha=0.3)
        # ax.plot([0, M], [a_un, a_un], c='blue', ls='--')
        ax.plot([0, M], [self.cl_payoff, self.cl_payoff], c='black', ls='--', label='Cl. payoff')
        ax.plot([0, M], [a_un, a_un], c='blue', ls='--')

        ax.scatter(np.arange(M + 1), bin_data, c='C1', marker='+', label='binary', zorder=10)
        ax.fill_between(np.arange(M + 1), bin_data - bin_conf, bin_data + bin_conf, color='C1', alpha=0.3, zorder=10)
        ax.plot([0, M], [payoff_bin, payoff_bin], c='orangered', ls='--')
        ax.set(xlabel='AE iterations', ylabel='Payoff', xticks=np.arange(0, M + 1), xticklabels=np.arange(0, M + 1))
        ax.legend()

        fig.savefig(name_folder_data(
            self.data) + '/%s_bins/' % bins + error_name + '_amplitude_estimation_perfect_circuit_results.pdf')

        z = erfinv(1 - alpha/2)
        a_max = (np.max(values) - self.K)
        fig, bx = plt.subplots()
        bx.scatter(np.arange(M + 1) + 0.1, un_conf, c='C0', label='unary', marker='x', zorder=3, s=100)
        bound_down = np.sqrt(un_data) * np.sqrt(a_max - un_data) * z / np.sqrt(shots) / np.cumsum(
            1 + 2 * (np.arange(M + 1)))
        bound_up = np.sqrt(un_data) * np.sqrt(a_max - un_data) * z / np.sqrt(shots) / np.sqrt(np.cumsum(
            1 + 2 * (np.arange(M + 1))))
        bx.plot(np.arange(M + 1) + 0.1, bound_up, ls=':', c='C0')
        bx.plot(np.arange(M + 1) + 0.1, bound_down, ls='-.', c='C0')

        bx.scatter(np.arange(M + 1) - 0.1, bin_conf, c='C1', label='binary', marker='+', zorder=3, s=100)
        a_max = (high - low) / bins * (bins - 1 - k) / (2 * c)
        bound_down = np.sqrt(bin_data) * np.sqrt(a_max - bin_data) * z / np.sqrt(shots) / np.cumsum(
            1 + 2 * (np.arange(M + 1)))
        bound_up = np.sqrt(bin_data) * np.sqrt(a_max - bin_data) * z / np.sqrt(shots) / np.sqrt(np.cumsum(
            1 + 2 * (np.arange(M + 1))))
        bx.plot(np.arange(M + 1) - 0.1, bound_up, ls=':', c='C1')
        bx.plot(np.arange(M + 1) - 0.1, bound_down, ls='-.', c='C1')

        bx.set(xlabel='AE iterations', xticks=np.arange(0, M+1), xticklabels=np.arange(0, M+1),
               ylabel=r'$\Delta {\rm Payoff} $', ylim=[0.0005,0.05])
        plt.yscale('log')
        custom_lines = [Line2D([0], [0], color='C0', marker='x', lw=0, markersize=10),
                        Line2D([0], [0], color='C1', marker='+', lw=0, markersize=10),
                        Line2D([0], [0], color='black', lw=1, ls=':'),
                        Line2D([0], [0], color='black', lw=1, ls='-.')]
        bx.legend(custom_lines, ['Unary', 'Binary', 'Cl. sampling', 'Optimal AE'])
        fig.tight_layout()

        fig.savefig(name_folder_data(
                self.data) + '/%s_bins/' % bins + error_name + '_amplitude_estimation_perfect_circuit_error.pdf')



    def paint_amplitude_estimation_unary(self, bins, error_name, repeats, M=4, measure_error=False, thermal_error=False, shots=500, alpha=0.05):
        """
        Function to paint error and confidences for amplitude estimation in the unary case
        :param bins:
        :param error_name:
        :param repeats:
        :param M:
        :param measure_error:
        :param thermal_error:
        :param shots:
        :param alpha:
        :return: None. Saves data in files.
        """
        z = erfinv(1 - alpha / 2)
        (values, pdf) = un.get_pdf(bins, self.S0, self.sig, self.r, self.T)[0]
        payoff_un = np.sum(pdf[values >= self.K] * (values[values >= self.K] - self.K))
        error_name = self.change_name(error_name, measure_error, thermal_error)
        m_s = np.arange(0, M + 1, 1)
        fig_0, ax_0 = plt.subplots()
        fig_1, ax_1 = plt.subplots()
        custom_lines=[]
        for j, m in enumerate(m_s):
            payoff = np.loadtxt(name_folder_data(
                self.data) + '/%s_bins/unary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s).npz' % (
                       self.max_gate_error, self.steps, repeats, m)) * (np.max(values) - self.K)
            payoff_confidences= np.loadtxt(name_folder_data(
                self.data) + '/%s_bins/unary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s)_confidence.npz' % (
                                             self.max_gate_error, self.steps, repeats, m)) * (np.max(values) - self.K)

            data = np.empty((self.steps, 2))
            data_a = np.empty((self.steps, 2))
            conf = np.empty((self.steps, 2))
            conf_a = np.empty((self.steps, 2))
            for _ in range(self.steps):
                data[_], conf[_] = experimental_data(np.abs(payoff[_] - payoff_un) / payoff_un,
                                                     payoff_confidences[_] / payoff_un)
                data_a[_], conf_a[_] = experimental_data(payoff[_] / payoff_un, payoff_confidences[_]/payoff_un)

            ax_0.scatter(100*self.error_steps, 100*(data[:,0]), color='C%s' % (j), label=r'M=%s' % m, marker='x')
            ax_0.fill_between(100*self.error_steps, 100*(data[:,0] - data[:, 1]), 100*(data[:,0] + data[:, 1]), color='C%s' % (j), alpha=0.3)

            custom_lines.append(Line2D([0], [0], color='C%s' % (j), lw=0, marker='x'))

            ax_1.scatter(100*self.error_steps, 100*conf_a[:, 0], color='C%s' % (j), marker='+', zorder=0, s=30, label=r'M=%s' % m)
            a_max = (np.max(values) - self.K) / payoff_un
            bound_down = np.sqrt(data_a[:, 0]) * np.sqrt(a_max - data_a[:, 0]) * z / np.sqrt(shots) / np.sum(
                1 + 2 * (np.arange(m + 1)))
            bound_up = np.sqrt(data_a[:, 0]) * np.sqrt(a_max - data_a[:, 0]) * z / np.sqrt(shots) / np.sqrt(np.sum(
                1 + 2 * (np.arange(m + 1))))
            ax_1.plot(100 * self.error_steps, 100*bound_down, ls='-.', color='C%s' % (j), zorder=2)
            ax_1.plot(100 * self.error_steps, 100*bound_up, ls=':', color='C%s' % (j), zorder=2)

        ax_0.set(xlabel='single-qubit gate error (%)', ylabel='percentage off optimal payoff (%)', ylim=[0, 175])
        ax_1.set(xlabel='single-qubit gate error (%)', ylabel='$\Delta$ payoff (%)', ylim=[0.05, 50], yscale='log')
        ax_0.legend(loc='upper left')
        custom_lines.append(Line2D([0], [0], color='black', lw=1, ls=':'))
        custom_lines.append(Line2D([0], [0], color='black', lw=1, ls='-.'))
        ax_1.legend(custom_lines, [r'M=%s' % m for m in m_s] + ['Cl. sampling', 'Optimal AE'])
        fig_0.tight_layout()
        fig_1.tight_layout()
        fig_0.savefig(name_folder_data(
                self.data) + '/%s_bins/unary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_data.pdf' % (
                            self.max_gate_error, self.steps, repeats))
        fig_1.savefig(name_folder_data(
            self.data) + '/%s_bins/unary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_sample_error.pdf' % (
                          self.max_gate_error, self.steps, repeats))


    def paint_amplitude_estimation_binary(self, bins, error_name, repeats, M=4, measure_error=False,
                                         thermal_error=False, shots=500, alpha=0.05, u=0, e=0.05):
        """
        Function to paint error and confidences for amplitude estimation in the binary case
        :param bins:
        :param error_name:
        :param repeats:
        :param M:
        :param measure_error:
        :param thermal_error:
        :param shots:
        :param alpha:
        :return: None. Saves data in files.
        """

        z = erfinv(1 - alpha / 2)
        error_name = self.change_name(error_name, measure_error, thermal_error)
        m_s = np.arange(0, M + 1, 1)
        fig_0, ax_0 = plt.subplots()
        fig_1, ax_1 = plt.subplots()
        custom_lines=[]
        (values, pdf) = bin.get_pdf(bins, self.S0, self.sig, self.r, self.T)[1]
        high, low = np.max(values), np.min(values)
        k = int(np.floor(bins * (self.K - low) / (high - low)))
        c = (2 * e) ** (1 / (2 * u + 2))
        for j, m in enumerate(m_s):
            a = np.loadtxt(name_folder_data(
                self.data) + '/%s_bins/binary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s).npz' % (
                       self.max_gate_error, self.steps, repeats, m))
            payoff = (high - low) / bins * (a - .5 + c) * (bins - 1 - k) / (2 * c)
            confidences= np.loadtxt(name_folder_data(
                self.data) + '/%s_bins/binary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_M(%s)_confidence.npz' % (
                                             self.max_gate_error, self.steps, repeats, m))
            payoff_confidences = (high - low) / bins * (confidences) * (bins - 1 - k) / (2 * c)

            data = np.empty((self.steps, 2))
            data_a = np.empty((self.steps, 2))
            conf = np.empty((self.steps, 2))
            conf_a = np.empty((self.steps, 2))

            for _ in range(self.steps):
                if _ == 0:
                    a_bin = errors_experiment(a[_], confidences[_])[0]
                    payoff_bin = (high - low) / bins * (a_bin - .5 + c) * (bins - 1 - k) / (2 * c)

                data[_], conf[_] = experimental_data(np.abs(payoff[_] - payoff_bin) / payoff_bin, payoff_confidences[_]/payoff_bin)
                data_a[_], conf_a[_] = experimental_data(payoff[_]/payoff_bin, payoff_confidences[_]/payoff_bin)

            ax_0.scatter(100 * self.error_steps, 100 * (data[:, 0]), color='C%s' % (j), label=r'M=%s' % m,
                         marker='+')
            ax_0.fill_between(100 * self.error_steps, 100 * (data[:, 0] - data[:, 1]),
                              100 * (data[:, 0] + data[:, 1]), color='C%s' % (j), alpha=0.3)

            custom_lines.append(Line2D([0], [0], color='C%s' % (j), lw=0, marker='+'))

            ax_1.scatter(100 * self.error_steps, 100 * conf_a[:, 0], color='C%s' % (j), marker='+', zorder=0, s=30,
                         label=r'M=%s' % m)

            a_max = (high - low) / bins * (bins - 1 - k) / (2 * c) / payoff_bin
            bound_down = np.sqrt(data_a[:, 0]) * np.sqrt(a_max - data_a[:, 0]) * z / np.sqrt(shots) / np.sum(
                1 + 2 * (np.arange(m + 1)))
            bound_up = np.sqrt(data_a[:, 0]) * np.sqrt(a_max - data_a[:, 0]) * z / np.sqrt(shots) / np.sqrt(np.sum(
                1 + 2 * (np.arange(m + 1))))
            ax_1.plot(100 * self.error_steps, 100 * bound_down, ls='-.', color='C%s' % (j), zorder=2)
            ax_1.plot(100 * self.error_steps, 100 * bound_up, ls=':', color='C%s' % (j), zorder=2)

        ax_0.set(xlabel='single-qubit gate error (%)', ylabel='percentage off optimal payoff (%)', ylim=[0, 175])
        ax_1.set(xlabel='single-qubit gate error (%)', ylabel='$\Delta$ payoff (%)', ylim=[0.05, 50],
                 yscale='log')
        ax_0.legend(loc='upper left')
        custom_lines.append(Line2D([0], [0], color='black', lw=1, ls=':'))
        custom_lines.append(Line2D([0], [0], color='black', lw=1, ls='-.'))
        ax_1.legend(custom_lines, [r'M=%s' % m for m in m_s] + ['Cl. sampling', 'Optimal AE'])
        fig_0.tight_layout()
        fig_1.tight_layout()
        fig_0.savefig(name_folder_data(
                self.data) + '/%s_bins/binary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_data.pdf' % (
                            self.max_gate_error, self.steps, repeats))
        fig_1.savefig(name_folder_data(
            self.data) + '/%s_bins/binary/amplitude_estimation/' % bins + error_name + '_gate(%s)_steps(%s)_repeats(%s)_sample_error.pdf' % (
                          self.max_gate_error, self.steps, repeats))

