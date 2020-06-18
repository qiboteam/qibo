import numpy as np
from scipy.optimize import minimize
import os
from qutip import Qobj, gate_expand_1toN
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import tensorflow as tf
from tensorflow import optimizers
from torch import optim
import torch

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

def u3(phi, phi2, phi3, N=None, target=0):
    if N is not None:
        return gate_expand_1toN(u3(phi, phi2, phi3), N, target)
    else:
        return Qobj([[np.cos(phi / 2), -1.0*np.exp(1j * phi2)*np.sin(phi / 2)],
                     [np.exp(1j * phi3)*np.sin(phi / 2), np.exp(1j * (phi2+phi3))*np.cos(phi / 2)]])

def create_random_state(seed):
    np.random.seed(seed)
    state = (np.random.rand(8) - .5) + 1j*(np.random.rand(8) - .5)
    state = state / np.linalg.norm(state)

    return state

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

    n, bins = np.histogram(tangles, 20, density=True)

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
    ax.set_ylim(bottom.min(), top.max())
    ax.set(xlabel='Tangle', ylabel='Probability')

    fig.savefig('histogram_random.png') #Habrá que mejorar esto


def tangle_circuit_error(opt_parameters, initial_state, error):
    single = 0.001 * error
    measure = 0.01 * error
    r = np.random.rand(12)
    state = u3(opt_parameters[0], opt_parameters[1], opt_parameters[2], N=3, target=0) * initial_state

    if r[0] < single:
        state = u3(np.pi, np.pi, 0, N=3, target=0) * state
    if r[1] < single:
        state = u3(np.pi, np.pi / 2, np.pi / 2, N=3, target=0) * state
    if r[2] < single:
        state = u3(0, np.pi, 0, N=3, target=0) * state
    if r[3] < measure:
        state = u3(np.pi, np.pi, 0, N=3, target=0) * state
    state = u3(opt_parameters[3], opt_parameters[4], opt_parameters[5], N=3, target=1) * state
    if r[4] < single:
        state = u3(np.pi, np.pi, 0, N=3, target=1) * state
    if r[5] < single:
        state = u3(np.pi, np.pi / 2, np.pi / 2, N=3, target=1) * state
    if r[6] < single:
        state = u3(0, np.pi, 0, N=3, target=1) * state
    if r[7] < measure:
        state = u3(np.pi, np.pi, 0, N=3, target=1) * state
    state = u3(opt_parameters[6], opt_parameters[7], opt_parameters[8], N=3, target=2) * state
    if r[8] < single:
        state = u3(np.pi, np.pi, 0, N=3, target=2) * state
    if r[9] < single:
        state = u3(np.pi, np.pi / 2, np.pi / 2, N=3, target=2) * state
    if r[10] < single:
        state = u3(0, np.pi, 0, N=3, target=2) * state
    if r[11] < measure:
        state = u3(np.pi, np.pi, 0, N=3, target=2) * state
    cum_probs = np.cumsum(np.abs(state[:].flatten())**2)
    n = np.random.rand()
    for i, s in enumerate(cum_probs):
        if n < s:
            break
    return i


def measure_circuit_error(opt_parameters, state, shots, error):
    outcomes = np.zeros(8)
    for s in range(shots):
        i = tangle_circuit_error(opt_parameters, state, error)
        outcomes[i] += 1
    outcomes[2] = 0
    outcomes[4] = 0
    outcomes[6] = 0
    outcomes = outcomes / np.sum(outcomes)

    return outcomes

def circuit_error(params, state, shots, error):
    outcomes = np.zeros(8)
    for s in range(shots):
        i = tangle_circuit_error(params, state, error)
        outcomes[i] = outcomes[i] + 1
    ans = (outcomes[2] + outcomes[4] + outcomes[6]) / np.sum(outcomes)
    return ans

def minimize_circuit_error(state, shots, error):
    #params = np.zeros(9)
    #result = minimize(circuit_error, params, args=(state, shots, error))

    #result = minimize_tf(state, shots, error)
    #result = minimize_pytorch(state, shots, error)
    result = minimize_spsa(state, shots, error)
    #return result.x, result.fun, result.nfev


def minimize_tf(state, shots, error):
    # Declare the TensorFlow variables
    params = tf.Variable(np.zeros(9), trainable=True)
    state = tf.Variable(state, trainable=False)
    shots = tf.Variable(shots, trainable=False, dtype=tf.int16)
    error = tf.Variable(error, trainable=False, dtype=tf.int16)
    # Number of iterations
    N = 10000

    # The ADAM optimizer
    adam = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)

    for s in range(N):
        print(s)
        with tf.GradientTape() as tape:
            print(tape)
            loss = circuit_error(params, state, shots, error)
        grads = tape.gradient(loss, [params])
        adam.apply_gradients(zip(grads, [params]))


def minimize_pytorch(state, shots, error):
    params = torch.DoubleTensor(np.zeros(9))
    optimizer = optim.Adam([params], lr=0.0001)
    N=1000
    for i in range(N):
        optimizer.zero_grad()
        loss = circuit_error(params, state, shots, error)
        loss.backward()
        optimizer.step()

def minimize_spsa(state, shots, error):
    k = 1
    s = 0
    params = np.random.rand(9)
    opt_params = params
    opt_f = circuit_error(opt_params, state, shots, error)
    a = 2
    c = 1
    A = 0
    alpha = 0.602
    gamma = 0.101
    while s < 50:
        a_k = a / (k + A) ** alpha
        c_k = c / k ** gamma
        delta = 2 * np.random.randint(2, size=9) - 1
        params_plus = params + c_k * delta
        params_minus = params - c_k * delta
        y_plus = circuit_error(params_plus, state, shots, error)
        y_minus = circuit_error(params_minus, state, shots, error)
        g = (y_plus - y_minus) / 2 / c_k * delta
        new_params = params - a_k * g
        y_new = circuit_error(params, state, shots, error)
        if y_plus == min(y_plus, min(y_minus, y_new)):
            params = params_plus
            y = y_plus
        elif y_minus == min(y_minus, min(y_plus, y_new)):
            params = params_minus
            y = y_minus
        else:
            params = new_params
            y = y_new
        if y < opt_f:
            opt_f = y
            opt_params = params
        k += 1
        r = np.random.rand()
        if r < 0.1:
            print('restart!')
            k = 1
            s += 1
        print(y)

    return opt_params, opt_f


def get_all_tangle(seed, error=5):
    if seed == 'GHZ':
        state = 1 / np.sqrt(2) * np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=complex)
    elif seed == 'W':
        state = 1 / np.sqrt(3) * np.array([0, 1, 1, 0, 1, 0, 0, 0], dtype=complex)
    else:
        state = create_random_state(seed)
    tangle = 4 * np.abs(hyperdeterminant(state))
    opt_parameters, f_local, n_evs = minimize_circuit_error(state, 1024, error)
    opt_tangle = 4 * measure_circuit_error(opt_parameters, state, 1024, error)[0] * measure_circuit_error(opt_parameters, state, 1024, error)[7]

    return state, opt_parameters, (f_local, n_evs), (tangle, opt_tangle)

def write_text(seed):
    for error in np.arange(5, 6):
        state, opt_parameters, f_local, (tangle, opt_tangle) = get_all_tangle(seed, error)
        folder = 'results_simulation/seed_'+str(seed) + '/error_' + str(error)
        create_folder(folder)
        print(f_local)
        np.savetxt(folder + '/state.txt', state)
        np.savetxt(folder + '/opt_parameters.txt', opt_parameters)
        np.savetxt(folder + '/min_values.txt', np.array(f_local))
        np.savetxt(folder + '/tangles.txt', np.array([tangle, opt_tangle]))


def read_text(seed):
    folder = 'results_simulation/' + str(seed)
    state = np.loadtxt(folder + '/state.txt', dtype=complex)
    opt_parameters = np.loadtxt(folder + '/opt_parameters.txt')
    min_values = np.loadtxt(folder + '/min_values.txt')
    tangles = np.loadtxt(folder + '/tangles.txt')

    return state, opt_parameters, min_values, tangles


def tangle_circuit_errors(seed, shots=1024, N=10, statistics=10):
    state, opt_parameters, min_values, tangles = read_text(seed)
    errors = np.linspace(0, 1, N)
    errors_tangles = np.empty((N, 3))

    for i, e in enumerate(errors):
        outcomes = np.zeros(statistics)
        for s in range(statistics):
            C = measure_circuit_error(opt_parameters, state, shots, e)
            outcomes[s] = np.abs((4 * C[0] * C[7]))

        outcomes = np.abs(outcomes - tangles[1])
        outcomes = np.sort(outcomes)
        errors_tangles[i, 0] = np.mean(outcomes)
        errors_tangles[i, 1] = outcomes[2 * statistics // 10]
        errors_tangles[i, 2] = outcomes[-2 * statistics // 10]
        print(errors_tangles[i])

    errors_tangles = errors_tangles / tangles[1]
    folder = 'results_simulation/' + str(seed)
    np.savetxt(folder + '/error_tangles' + '.txt', errors_tangles)

    return errors_tangles


def random_to_circuit_tangle(max_seed):
    random_tangles = np.empty(max_seed + 1)
    circuit_tangles = np.empty(max_seed + 1)
    for seed in range(max_seed + 1):
        folder = 'results_simulation/' + str(seed)
        tangles = np.loadtxt(folder + '/tangles.txt')
        random_tangles[seed] = tangles[0]
        circuit_tangles[seed] = tangles[1]

    fig, ax = plt.subplots()
    ax.hist(random_tangles, bins=100)
    ax.set(xlabel='Tangle', ylabel='Samples', title='Random states')
    fig.savefig('results_simulation/{}_hist_exact.png'.format(max_seed))

    fig, ax = plt.subplots()
    ax.hist(circuit_tangles, bins=100)
    ax.set(xlabel='Tangle', ylabel='Samples', title='Circuit states')
    fig.savefig('results_simulation/{}_hist_circuit.png'.format(max_seed))

    fig, ax = plt.subplots()
    ax.scatter(random_tangles, circuit_tangles, color='red', alpha=0.1)
    ax.plot([0,1], [0,1], color='black')
    ax.set(xlabel='Exact tangle', ylabel='Circuit tangle')
    fig.savefig('results_simulation/{}_random_circuit.png'.format(max_seed))


def tangle_errors(seed):
    folder = 'results_simulation/' + str(seed)
    errors = np.loadtxt(folder + '/error_tangles.txt')
    X = np.linspace(0, 0.5, errors.shape[0])
    fig, ax = plt.subplots()
    ax.plot(X, 100*errors[:, 0], color='black', linestyle='-', label='Error')
    ax.fill_between(X, 100*errors[:, 1], 100*errors[:, 2], color='red', alpha=0.25)
    ax.set(xlabel='Single-qubit gate error (%)', ylabel='Tangle error (%)')
    fig.savefig('results_simulation/{}/errors_tangle.png'.format(seed))


def theoretical_error_GHZ(norm=False):
    initial_state = 1 / np.sqrt(2) * np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=complex)
    tangle = np.zeros(12)
    probs = np.zeros(12)
    for r in range(1, 2**12):
        state = initial_state
        prob = 1
        r = np.binary_repr(r, 12)
        if r[0] == '1':
            state = u3(np.pi, np.pi, 0, N=3, target=0) * state
            prob *= 0.005
        if r[1] == '1':
            state = u3(np.pi, np.pi / 2, np.pi / 2, N=3, target=0) * state
            prob *= 0.005
        if r[2] == '1':
            state = u3(0, np.pi, 0, N=3, target=0) * state
            prob *= 0.005
        if r[3] == '1':
            state = u3(np.pi, np.pi, 0, N=3, target=0) * state
            prob *= 0.05
        if r[4] == '1':
            state = u3(np.pi, np.pi, 0, N=3, target=1) * state
            prob *= 0.005
        if r[5] == '1':
            state = u3(np.pi, np.pi / 2, np.pi / 2, N=3, target=1) * state
            prob *= 0.005
        if r[6] == '1':
            state = u3(0, np.pi, 0, N=3, target=1) * state
            prob *= 0.005
        if r[7] == '1':
            state = u3(np.pi, np.pi, 0, N=3, target=1) * state
            prob *= 0.05
        if r[8] == '1':
            state = u3(np.pi, np.pi, 0, N=3, target=2) * state
            prob *= 0.005
        if r[9] == '1':
            state = u3(np.pi, np.pi / 2, np.pi / 2, N=3, target=2) * state
            prob *= 0.005
        if r[10] == '1':
            state = u3(0, np.pi, 0, N=3, target=2) * state
            prob *= 0.005
        if r[11] == '1':
            state = u3(np.pi, np.pi, 0, N=3, target=2) * state
            prob *= 0.05

        if norm==True:
            state[1] = 0
            state[2] = 0
            state[3] = 0
            state = state / np.linalg.norm(state)
        n = r.count('1')
        tangle[n - 1] = tangle[n - 1] + prob * (1 - 4 * np.abs(opt_hyperdeterminant(state)))
        probs[n - 1] = probs[n - 1] + prob

    np.savetxt('results_simulation/GHZ/theoretical_tangle.txt', tangle)
    np.savetxt('results_simulation/GHZ/theoretical_probs.txt', probs)
    t = np.linspace(0, 1, 100)
    tau = np.zeros_like(t)
    for i in range(1, 13):
        tau += tangle[i - 1] * t ** i

    fig, ax = plt.subplots()
    ax.plot(t*0.5, 100 * tau, color='black')
    ax.set(xlabel='Single-qubit gate error (%)', ylabel='Tangle error (%)')
    fig.savefig('results_simulation/GHZ/theoretical_error.png')


def theoretical_error_W():
    initial_state = 1 / np.sqrt(3) * np.array([0, 1, 1, 0, 1, 0, 0, 0], dtype=complex)
    tangle = np.zeros(100)
    t = np.linspace(0, 1, 100)
    probs = 0
    state, opt_parameters, min_values, tangles = read_text('W')
    for r in range(1, 2**12):
        state = initial_state
        prob = 1
        r = np.binary_repr(r, 12)
        state = u3(opt_parameters[0], opt_parameters[1], opt_parameters[2], N=3, target=0) * state
        if r[0] == '1':
            state = u3(np.pi, np.pi, 0, N=3, target=0) * state
            prob *= 0.005
        if r[1] == '1':
            state = u3(np.pi, np.pi / 2, np.pi / 2, N=3, target=0) * state
            prob *= 0.005
        if r[2] == '1':
            state = u3(0, np.pi, 0, N=3, target=0) * state
            prob *= 0.005
        if r[3] == '1':
            state = u3(np.pi, np.pi, 0, N=3, target=0) * state
            prob *= 0.05

        state = u3(opt_parameters[3], opt_parameters[4], opt_parameters[5], N=3, target=1) * state
        if r[4] == '1':
            state = u3(np.pi, np.pi, 0, N=3, target=1) * state
            prob *= 0.005
        if r[5] == '1':
            state = u3(np.pi, np.pi / 2, np.pi / 2, N=3, target=1) * state
            prob *= 0.005
        if r[6] == '1':
            state = u3(0, np.pi, 0, N=3, target=1) * state
            prob *= 0.005
        if r[7] == '1':
            state = u3(np.pi, np.pi, 0, N=3, target=1) * state
            prob *= 0.05

        state = u3(opt_parameters[6], opt_parameters[7], opt_parameters[8], N=3, target=2) * state
        if r[8] == '1':
            state = u3(np.pi, np.pi, 0, N=3, target=2) * state
            prob *= 0.005
        if r[9] == '1':
            state = u3(np.pi, np.pi / 2, np.pi / 2, N=3, target=2) * state
            prob *= 0.005
        if r[10] == '1':
            state = u3(0, np.pi, 0, N=3, target=2) * state
            prob *= 0.005
        if r[11] == '1':
            state = u3(np.pi, np.pi, 0, N=3, target=2) * state
            prob *= 0.05
        n = r.count('1')
        tangle += prob * 4 * np.abs(opt_hyperdeterminant(state)) * t ** n
        probs += prob

    plt.plot(t, tangle)
    plt.show()


def tangle_errors_statistical(max_seed):
    folder = 'results_simulation/' + str(0)
    errors = np.loadtxt(folder + '/error_tangles.txt')
    minim_error = np.loadtxt(folder + '/min_values.txt')
    shape = (max_seed + 1,) + errors.shape
    all_errors = np.empty(shape)
    minim_errors = np.empty(max_seed + 1)
    all_errors[0] = errors
    minim_errors[0] = minim_error
    for s in range(1, max_seed + 1):
        folder = 'results_simulation/' + str(s)
        minim_error = np.loadtxt(folder + '/min_values.txt')
        all_errors[s] = np.loadtxt(folder + '/error_tangles.txt')
        minim_errors[s] = minim_error

    means = np.mean(all_errors, axis=0)
    X = np.linspace(0, 0.5, shape[1])

    fig, ax = plt.subplots()
    ax.plot(X, 100 * means[:, 0], color='black')
    ax.fill_between(X, 100 * means[:, 1], 100 * means[:, 2], color='red', alpha=0.25)
    ax.set(xlabel='Gate error (%)', ylabel='Tangle error (%)')

    fig.savefig('results_simulation/statistical_errors_{}.png'.format(max_seed))

def create_folder(directory):
    """
    Auxiliar function for creating directories with name directory

    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

