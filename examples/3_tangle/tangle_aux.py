from QuantumState import QCircuit
import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.special import erf
import itertools, os, random
import matplotlib.pyplot as plt
from matplotlib import path, patches

def create_random_state(seed):
    if seed=='GHZ':
        state = 1 / np.sqrt(2) * np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=complex)
    else:
        np.random.seed(seed)
        state = (np.random.rand(8) - .5) + 1j*(np.random.rand(8) - .5)
        state = state / np.linalg.norm(state)

    return state

def circuit(initial_state, parameters):
    q = QCircuit(3)
    q.psi = np.copy(initial_state)
    q.U3(0, parameters[0])
    q.U3(1, parameters[1])
    q.U3(2, parameters[2])
    return q

def circuit_with_error(initial_state, parameters, errors):
    # errors es una lista de 0's y 1's en forma 3 x 4
    q = circuit(initial_state, parameters)
    for i in range(3):
        if errors[i, 0] == 1:
            q.X(i)
        if errors[i, 1] == 1:
            q.Y(i)
        if errors[i, 2] == 1:
            q.Z(i)
        if errors[i, 3] == 1:
            q.X(i)

    return q

def gen_errors(t, single_qubit = 0.001, measure = 0.01):
    single = np.random.binomial(1, single_qubit * t, size=(3, 3))
    measure = np.random.binomial(1, measure * t, size=(3, 1))

    return np.concatenate((single, measure), axis=1)

def statevector_simulation(initial_state, parameters):
    q = circuit(initial_state, parameters)
    samplings = np.abs(q.psi) ** 2
    return samplings

def complete_circuit(initial_state, t, parameters, shots=10000, single=0.001, measure=0.01):
    samplings = np.zeros(8)
    cases = np.zeros(13)
    for s in range(shots):
        error = gen_errors(t, single_qubit=single, measure=measure)
        cases[np.sum(error)] += 1
        q = circuit_with_error(initial_state, parameters, error)
        samplings += q.sampling(1)
    samplings = samplings / np.sum(samplings)
    return samplings

def ftm(n):
    #acronym: flat_to_matrix
    return (n // 4, n % 4)

def complete_circuit_efficient(initial_state, t, parameters, shots=10000,  single=0.001, measure=0.01):
    single = single * t
    measure = measure * t
    occurrences = np.array([[single, single, single, measure],
                      [single, single, single, measure],
                      [single, single, single, measure]])

    pos = np.arange(12)
    shots_done = 0
    sampling = np.zeros(8)
    q = circuit_with_error(initial_state, parameters, np.zeros((3,4)))
    appearences= 1 - occurrences
    prob = np.prod(appearences)
    sampling += q.sampling(int(prob * shots))
    shots_done += int(prob * shots)
    stop=False
    if shots_done == shots:
        stop=True
    o=0
    while stop==False:
        o += 1
        positions = list(itertools.combinations(pos, o))
        for p in positions:
            errors = np.zeros((3,4))
            for n in p:
                errors[ftm(n)] = 1
            q = circuit_with_error(initial_state, parameters, errors)
            appearences = errors * occurrences
            appearences[appearences == 0] = 1 - occurrences[appearences == 0]
            prob = np.prod(appearences)
            if prob * shots >= 1:
                sampling += q.sampling(int(prob * shots))
                shots_done += int(prob * shots)
            else:
                stop=True
                break

    o += 1
    positions = list(itertools.combinations(pos, o))
    positions = random.choices(positions, k=shots-shots_done)
    for p in positions:
        errors = np.zeros((3, 4))
        for n in p:
            errors[ftm(n)] = 1
        q = circuit_with_error(initial_state, parameters, errors)
        appearences = errors * occurrences
        appearences[appearences == 0] = 1 - occurrences[appearences == 0]
        sampling += q.sampling(1)
        shots_done += 1

    sampling = sampling / np.sum(sampling)
    return sampling

def cost_function_sampling(parameters, initial_state, shots, t):
    parameters = parameters.reshape((3,3))
    samplings = complete_circuit_efficient(initial_state, t, parameters, shots=shots)
    ans = samplings[1] + samplings[2] + samplings[3]
    return ans

def cost_function_statevector(parameters, initial_state):
    parameters = parameters.reshape((3,3))
    samplings = statevector_simulation(initial_state, parameters)
    ans = samplings[1] + samplings[2] + samplings[3]
    return ans


def optimizer_statevector(seed):
    initial_state = create_random_state(seed)
    parameters = np.zeros(9)
    result = minimize(cost_function_statevector, parameters, args=(initial_state))

    return result.x, result.fun, result.nfev

def optimizer_sampling(seed, t, initial_parameters=np.zeros(9), shots=10000, method='Powell'):
    initial_state = create_random_state(seed)
    result = minimize(cost_function_sampling, initial_parameters, args=(initial_state, shots, t), method=method)

    return result.x, result.fun, result.nfev

def optimizer_hybrid(seed, shots=10000, t=0, method='Powell'):
    opt_parameters, error, n_evs_statevector = optimizer_statevector(seed)

    opt_parameters, error, n_evs_sampling = optimizer_sampling(seed, t, opt_parameters, shots=shots, method=method)

    return opt_parameters, error, (n_evs_statevector, n_evs_sampling)


def write_optimizer_sampling(seed, t, shots=10000, initial_parameters=np.zeros(9), method='Powell'):
    min = 1
    opt_params = initial_parameters
    k=0
    while min > t * 0.02 + 0.001:
        print('Try', seed, t)
        k += 1
        opt_params_, min_, nfev_ = optimizer_sampling(seed, t, initial_parameters=initial_parameters, shots=shots, method=method)
        if min_ < min:
            opt_params = opt_params_
            min = min_
            nfev = nfev_
        np.random.seed(np.random.randint(0, 1000))
        initial_parameters = np.random.rand(9)
        print(min, nfev)
        if k == 5:
            break


    folder = 'results_simulation/seed_' + str(seed) + '/error_' + str(t)
    create_folder(folder)
    np.savetxt(folder + '/opt_parameters.txt', opt_params.reshape((3,3)))
    np.savetxt(folder + '/opti_info.txt', np.array([min, nfev]))

    return opt_params


def hyperdeterminant(state):
    """
    :param state: quantum state
    :return: the hyperdeterminant of the state
    """
    hyp = (
    state[0]**2 * state[7]**2 + state[1]**2 * state[6]**2 + state[2]**2 * state[5]**2 + state[3]**2 * state[4]**2
    ) - 2*(
    state[0]*state[7]*state[3]*state[4] + state[0]*state[7]*state[5]*state[2] + state[0]*state[7]*state[6]*state[1] + state[3]*state[4]*state[5]*state[2] + state[3]*state[4]*state[6]*state[1] + state[5]*state[2]*state[6]*state[1]
    ) + 4 * (
    state[0]*state[6]*state[5]*state[3] + state[7]*state[1]*state[2]*state[4]
    )
    return hyp

def opt_hyperdeterminant(state):
    """

    :param state: quantum state
    :return: the hyperdeterminant of the state
    """
    hyp = (
    state[0]**2 * state[7]**2
    )
    return hyp

def get_tangle_state(seed):
    state = create_random_state(seed)
    return 4 * np.abs(hyperdeterminant(state))

def get_tangle_measure(seed, t, parameters, norm, shots=10000, single=0.001, measure=0.01, number_tangles=10):
    tangles = np.empty(number_tangles)
    initial_state = create_random_state(seed)
    for i in range(number_tangles):
        sampling = complete_circuit_efficient(initial_state, t, parameters, shots=shots, single=single, measure=measure)
        if norm==True:
            sampling[1] = 0
            sampling[2] = 0
            sampling[3] = 0
            sampling = sampling / np.sum(sampling)

        tangles[i] = 4 * sampling[0] * sampling[7]
    return tangles

def get_tangle_measure_2(seed, t, parameters, norm=True, shots=10000, single=0.001, measure=0.01, number_tangles=10):
    tangles = np.empty(number_tangles)
    initial_state = create_random_state(seed)
    for i in range(number_tangles):
        sampling = complete_circuit(initial_state, parameters, shots=shots, t=t, single=single, measure=measure)
        if norm==True:
            sampling[1] = 0
            sampling[2] = 0
            sampling[3] = 0
            sampling = sampling / np.sum(sampling)


        tangles[i] = 4 * sampling[0] * sampling[7]

    return tangles

def write_tangles(seed, t, norm, shots=10000, number_tangles=10, single=0.001, measure=0.01):
    state = create_random_state(seed)
    exact_tangle = 4 * np.abs(hyperdeterminant(state))
    folder = 'results_simulation/seed_' + str(seed) + '/error_' + str(t)
    create_folder(folder)
    opt_params = np.loadtxt(folder + '/opt_parameters.txt')
    tangles = get_tangle_measure(seed, t, opt_params, norm=norm, shots=shots, number_tangles=number_tangles, single=single, measure=measure)
    if norm==False:
        np.savetxt(folder + '/tangles_with_error.txt', tangles)
    else:
        np.savetxt(folder + '/tangles_with_error_norm.txt', tangles)
    np.savetxt(folder + '/exact_tangle.txt', np.asfarray([exact_tangle]))
    return exact_tangle, tangles


def error_GHZ(ts, number_of_tangles, single=0.001, measure=0.01, shots=10000):
    tangles = np.empty((len(ts), number_of_tangles))
    for i, t in enumerate(ts):
        tangles[i] = get_tangle_measure('GHZ', t, np.zeros((3,3)), norm=False, shots=shots, number_tangles=number_of_tangles, single=single, measure=measure)

    fig, ax = plt.subplots()
    ax.plot(ts, np.mean(tangles, axis=1), color='C0', label='Not Norm')

    lower_bound = np.empty(len(ts))
    upper_bound = np.empty(len(ts))
    for i, tang in enumerate(tangles):
        tang.sort()
        lower_bound[i] = np.min(tang)
        upper_bound[i] = np.max(tang)

    ax.fill_between(ts, lower_bound, upper_bound, alpha=0.3, color='C0')

    np.savetxt('results_simulation/seed_GHZ/direct_tangles.txt', tangles)

    tangles = np.empty((len(ts), number_of_tangles))
    for i, t in enumerate(ts):
        tangles[i] = get_tangle_measure('GHZ', t, np.zeros((3, 3)), norm=True, shots=shots,
                                        number_tangles=number_of_tangles, single=single, measure=measure)

    ax.plot(ts, np.mean(tangles, axis=1), color='C1', label='Norm')

    lower_bound = np.empty(len(ts))
    upper_bound = np.empty(len(ts))
    for i, tang in enumerate(tangles):
        tang.sort()
        lower_bound[i] = np.min(tang)
        upper_bound[i] = np.max(tang)

    ax.fill_between(ts, lower_bound, upper_bound, alpha=0.3, color='C1')
    ax.set(xlabel='Measure gate error (%)', ylabel='Tangle GHZ')
    ax.legend()

    fig.savefig('results_simulation/seed_GHZ/direct_error.png')
    np.savetxt('results_simulation/seed_GHZ/direct_tangles_norm.txt', tangles)


def error_GHZ_2():
    ts = np.arange(0, 6)
    fig, ax = plt.subplots()
    tangles = np.loadtxt('results_simulation/seed_GHZ/direct_tangles_norm.txt')
    means = np.mean(tangles, axis=1)
    Ts = np.linspace(min(ts), max(ts), len(means))
    ax.plot(Ts, means, color='C1', label='Norm')

    lower_bound = np.empty(len(means))
    upper_bound = np.empty(len(means))
    for i, tang in enumerate(tangles):
        tang.sort()
        lower_bound[i] = np.min(tang)
        upper_bound[i] = np.max(tang)

    ax.fill_between(Ts, lower_bound, upper_bound, alpha=0.3, color='C1')

    tangles = np.loadtxt('results_simulation/seed_GHZ/direct_tangles.txt')
    means = np.mean(tangles, axis=1)
    Ts = np.linspace(min(ts), max(ts), len(means))
    ax.plot(Ts, means, color='C0', label='Not Norm')

    lower_bound = np.empty(len(means))
    upper_bound = np.empty(len(means))
    for i, tang in enumerate(tangles):
        tang.sort()
        lower_bound[i] = np.min(tang)
        upper_bound[i] = np.max(tang)

    ax.fill_between(Ts, lower_bound, upper_bound, alpha=0.3, color='C0')


    tangles = np.empty((6, 10))
    tangles_norm = np.empty((6, 10))
    for t in ts:
        folder = 'results_simulation/seed_GHZ/error_' + str(t)
        tangles[t] = np.loadtxt(folder + '/tangles_with_error.txt')
        tangles_norm[t] = np.loadtxt(folder + '/tangles_with_error_norm.txt')

    tangle_error = np.mean(tangles, axis=1)
    mean_error = np.array([np.max(tangles, axis=1) - tangle_error, np.min(tangles, axis=1) - tangle_error])
    ax.scatter(ts, tangle_error, color='blue', s=15)
    ax.errorbar(ts, tangle_error, yerr=mean_error, color='blue', linestyle='', markersize=10)
    tangle_error_norm = np.mean(tangles_norm, axis=1)
    mean_error_norm = np.array([np.max(tangles_norm, axis=1) - tangle_error_norm, -np.min(tangles_norm, axis=1) + tangle_error_norm])
    ax.scatter(ts, tangle_error_norm, color='red', s=15)
    ax.errorbar(ts, tangle_error_norm, yerr=mean_error_norm, color='red', linestyle='', markersize=10)
    ax.set(xlabel='Measure gate error (%)', ylabel='Tangle GHZ')
    ax.legend()
    fig.savefig(('results_simulation/seed_GHZ/direct_and_optimized_error.png'))



def measured_tangles(max_seed, error=1, tol=0.001):
    fig, (ax, bx) = plt.subplots(ncols=2, figsize=(10,5))
    for s in range(max_seed + 1):
        folder = 'results_simulation/seed_' + str(s)+'/error_' + str(error)
        f = np.loadtxt(folder + '/opti_info.txt')[0]
        if f > tol:
            p = 0
        else:
            p = 1
        e_tangle = np.loadtxt(folder + '/exact_tangle.txt')
        tangles = np.loadtxt(folder + '/tangles_with_error.txt') * p
        tangles[tangles == 0] = -1
        tangles_norm = np.loadtxt(folder + '/tangles_with_error_norm.txt') * p
        tangles_norm[tangles_norm == 0] = -1
        ax.scatter(e_tangle * np.ones(len(tangles)), tangles, color='C0', alpha=.5, s=0.2)
        bx.scatter(e_tangle * np.ones(len(tangles_norm)), tangles_norm, color='C1', alpha=0.5, s=0.2)

    ax.set(xlabel='Exact tangle', ylabel='Measured tangle', xlim=[0,1], ylim=[0,1])
    ax.plot([0,1], [0,1], color='black')
    bx.set(xlabel='Exact tangle', xlim=[0, 1], ylim=[0, 1])
    bx.plot([0, 1], [0, 1], color='black')

    fig.savefig('results_simulation/measured_tangles_error:{}_tol:{}.png'.format(error, tol))

def relative_errors(max_seed, max_error=5, tol=0.02):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    data = [[]] * (max_error + 1)
    data_norm = [[]] * (max_error + 1)
    for error in range(max_error + 1):
        tangles = []
        tangles_norm = []
        p=0
        for s in range(max_seed + 1):
            folder = 'results_simulation/seed_' + str(s) + '/error_' + str(error)
            f = np.loadtxt(folder + '/opti_info.txt')[0]
            if f < tol * error + 0.001:
                p += 1
                e_tangle = np.loadtxt(folder + '/exact_tangle.txt')
                tangles.append(np.abs(np.loadtxt(folder + '/tangles_with_error.txt') - e_tangle) / e_tangle)
                tangles_norm.append(np.abs(np.loadtxt(folder + '/tangles_with_error_norm.txt') - e_tangle) / e_tangle)

        tangles=np.array(tangles)
        tangles_mean = np.mean(tangles, axis=1)
        tangles_norm_mean = np.mean(tangles_norm, axis=1)
        tangles_mean.sort()
        data[error].append([np.mean(tangles_mean),
                            np.mean(tangles_mean) - tangles_mean[150 * p // 1000],
                            tangles_mean[-150 * p // 1000] - np.mean(tangles_mean)])
        tangles_norm_mean.sort()
        data_norm[error].append([np.mean(tangles_norm_mean),
                            np.mean(tangles_norm_mean) - tangles_norm_mean[150 * p // 1000],
                            tangles_norm_mean[-150 * p // 1000] - np.mean(tangles_norm_mean)])

        '''        tangles_norm_mean.sort()
        data_norm[error, 0] = np.mean(tangles_norm_mean)
        data_norm[error, 1] = data_norm[error, 0] - tangles_norm_mean[150 * max_seed // 1000]
        data_norm[error, 2] = tangles_norm_mean[-150 * max_seed // 1000] - data_norm[error, 0]'''

    data = np.array(data[0])
    data_norm = np.array(data_norm[0])
    ax.errorbar(np.arange(0, max_error + 1) - 0.05, data[:, 0], data[:, 1:].transpose(), color='C0', marker='.', capsize=5, label='Not Norm')
    ax.errorbar(np.arange(0, max_error + 1) + 0.05, data_norm[:, 0], data_norm[:, 1:].transpose(), color='C1', marker='.',
                capsize=5, label='Norm')
    ax.set(xlabel='Measure gate error (%)', ylabel='Absolute tangle relative error', xlim=[-.2, max_error + .2], ylim=[0, 0.6])
    ax.legend()


    fig.savefig('results_simulation/relative_errors_abs.png')


def compute_chi(max_seed, max_error=5, tol=0.02):
    for error in range(max_error + 1):
        chi = 0
        chi_norm = 0
        p=0
        for seed in range(max_seed + 1):
            folder = 'results_simulation/seed_' + str(seed) + '/error_' + str(error)
            f = np.loadtxt(folder + '/opti_info.txt')[0]
            if f <= tol*error + 0.001:
                p += 1
                e_tangle = np.loadtxt(folder + '/exact_tangle.txt')
                tangles = np.loadtxt(folder + '/tangles_with_error.txt')
                tangles_norm = np.loadtxt(folder + '/tangles_with_error_norm.txt')
                chi += (np.mean(tangles) - e_tangle) ** 2 / np.var(tangles)
                chi_norm += (np.mean(tangles_norm) - e_tangle) ** 2 / np.var(tangles_norm)
        chi = chi / p
        chi_norm = chi_norm / p
        print('Error {}'.format(error))
        print('Chi {}'.format(chi))
        print('Chi norm {}'.format(chi_norm))

def compute_relative_error(max_seed, max_error=5, num_tangles=10):
    rel_error = np.empty((max_error + 1, max_seed + 1))
    rel_error_norm = np.empty_like(rel_error)
    for error in range(max_error + 1):
        for seed in range(max_seed + 1):
            folder = 'results_simulation/seed_' + str(seed) + '/error_' + str(error)
            e_tangle = np.loadtxt(folder + '/exact_tangle.txt')
            tangles = np.loadtxt(folder + '/tangles_with_error.txt')
            tangles_norm = np.loadtxt(folder + '/tangles_with_error_norm.txt')
            rel_error[error, seed] = (np.mean(tangles) - e_tangle)**2
            rel_error_norm[error, seed] = (np.mean(tangles_norm) - e_tangle)**2


    print(np.argmax(rel_error))
    print(np.min(rel_error))

    rel_mean = np.mean(rel_error, axis=1)
    rel_max = np.max(rel_error, axis=1)
    rel_min = np.min(rel_error, axis=1)

    rel_mean_norm = np.mean(rel_error_norm, axis=1)
    rel_max_norm = np.max(rel_error_norm, axis=1)
    rel_min_norm = np.min(rel_error_norm, axis=1)

    fig, ax = plt.subplots()
    x = np.arange(0, max_error + 1)
    ax.plot(x, rel_mean, color='C0', marker='.', label='Not norm')
    ax.fill_between(x, rel_min, rel_max,  color='C0', alpha=0.3)
    ax.plot(x, rel_mean_norm, color='C1', marker='.', label='Norm')
    ax.fill_between(x, rel_min_norm, rel_max_norm, color='C1', alpha=0.3)
    ax.set(yscale='log')

    fig.savefig('results_simulation/rel_error.png')

def create_folder(directory):
    """
    Auxiliar function for creating directories with name directory

    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def compute_random_tangle(seed):
    state = create_random_state(seed)
    return 4 * np.abs(hyperdeterminant(state))

def random_tangles(N):
    tangles = np.empty(N)
    for i in range(N):
        tangles[i] = compute_random_tangle(i)

    return tangles


def random_tangle_hist(N):
    tangles = random_tangles(N)
    fig, ax = plt.subplots()

    n, bins = np.histogram(tangles, 50, density=True)

    # get the corners of the rectangles for the histogram
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n

    # we need a (numrects x numsides x 2) numpy array for the path helper
    # function to build a compound path
    XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

    # get the Path object
    barpath = path.Path.make_compound_path_from_polys(XY)

    # make a patch out of it
    patch = patches.PathPatch(barpath)
    ax.add_patch(patch)

    # update the view limits
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), 2)
    ax.set(xlabel='Tangle', ylabel='Probability density')
    fig.savefig('histogram_random.png')
    np.savetxt('results_simulation/n_hist.txt', n)
    np.savetxt('results_simulation/bins_hist.txt', bins)

def fit_tangle_dist():
    n = np.loadtxt('results_simulation/n_hist.txt')
    bins_2 = np.loadtxt('results_simulation/bins_hist.txt')
    bins = np.array([(bins_2[i] + bins_2[i + 1]) / 2 for i in range(len(bins_2) - 1)])


    p_opt, p_cov = curve_fit(fit_lognormal, bins, n, bounds=(0, np.inf))
    print(p_opt)

    fig, ax = plt.subplots()
    ax.plot(bins, fit_lognormal(bins, p_opt[0], p_opt[1]), color='red')
    left = np.array(bins_2[:-1])
    right = np.array(bins_2[1:])
    bottom = np.zeros(len(left))
    top = bottom + n

    # we need a (numrects x numsides x 2) numpy array for the path helper
    # function to build a compound path
    XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

    # get the Path object
    barpath = path.Path.make_compound_path_from_polys(XY)

    # make a patch out of it
    patch = patches.PathPatch(barpath)
    ax.add_patch(patch)

    # update the view limits
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())

    ax.set(xlabel='Tangle', ylabel='Probability density')
    fig.savefig('histogram_random_adjust.png')


def fit_function(x, a, C, e):
    #return (a**2 - np.sqrt(np.pi / 2) * a**3 * erf(1 / np.sqrt(2*a)))/(2 * np.pi) * (1 - x) * x * np.exp(-x**2 / 2 / a**2)
    return C * x**e * (1 - x**e) * np.exp(-x ** 2 / 2 / a ** 2)

def fit_lognormal(x, mu, sigma):
    return 1 / x / sigma / np.sqrt(2 * np.pi) * np.exp(-(np.log(x) - mu)**2 / 2 / sigma**2)
