from qibo.models import Circuit
from qibo import gates
import numpy as np
from datasets import create_dataset, create_target, fig_template, world_map_template
from qibo.hamiltonians import Hamiltonian
from qibo.config import matrices
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import os

np.random.seed(0)

class single_qubit_classifier:
    def __init__(self, name, layers, grid=11, test_samples=1000):
        #Circuit.__init__(self, nqubits)
        self.name = name
        self.layers = layers
        self.training_set = create_dataset(name, grid=grid)
        self.test_set = create_dataset(name, samples=test_samples)
        self.target = create_target(name)
        self.hamiltonian = self.create_hamiltonian()
        self.params = np.random.randn(layers * 4)
        try:
            os.makedirs('results/'+self.name+'/%s_layers'%self.layers)
        except:
            pass

    def set_parameters(self, new_params):
        self.params = new_params

    def circuit(self, x):
        C = Circuit(1)
        # x = x.transpose()
        index = 0
        for l in range(self.layers):
            C.add(gates.RY(0, self.params[index] * x[0] + self.params[index + 1]))
            index += 2
            C.add(gates.RZ(0, self.params[index] * x[1] + self.params[index + 1]))
            index += 2
        return C

    def cost_function_one_point_fidelity(self, x, y):
        C = self.circuit(x)
        state = C.execute()
        cf = .5 * (1 - fidelity(state, self.target[y])) ** 2
        return cf

    def cost_function_fidelity(self, params):
        self.set_parameters(params)
        cf = 0
        for x, y in zip(self.training_set[0], self.training_set[1]):
            cf += self.cost_function_one_point_fidelity(x, y)
        cf /= len(self.training_set[0])
        return cf

    def create_hamiltonian(self):
        self.H = [Hamiltonian(1)] * 3
        measures = [matrices._npX(), matrices._npY(), matrices._npZ()]
        for n, measur in enumerate(measures):
            h_ = Hamiltonian(1)
            h_.hamiltonian = measur
            self.H[n] = h_

        return self.H

    def create_target_hamiltonians(self):
        self.target_values = [[]] * len(self.target)
        for i, t in enumerate(self.target):
            self.target_values[i] = [h.expectation(t) for h in self.H]

    def cost_function_one_point_observables(self, x, y):
        C = self.circuit(x)
        state = C.execute()
        values = [h.expectation(state) for h in self.H]
        cf = tf.constant(np.arccos(np.dot(values, self.target_values[y])) ** 2)
        return cf

    def cost_function_observables(self, params):
        self.set_parameters(params)
        cf = 0
        for x, y in zip(self.training_set[0], self.training_set[1]):
            cf += self.cost_function_one_point_observables(x, y)
        cf /= len(self.training_set[0])
        return cf

    def minimize(self, method='BFGS', options=None, compile=True, fidelity=True):
        if fidelity:
            loss = self.cost_function_fidelity
        else:
            self.create_target_hamiltonians()
            loss = self.cost_function_observables
        if method == 'cma':
            # Genetic optimizer
            import cma
            r = cma.fmin2(lambda p: loss(p).numpy(), self.params, 2)
            result = r[1].result.fbest
            parameters = r[1].result.xbest

        elif method == 'sgd':
            from qibo.tensorflow.gates import TensorflowGate
            circuit = self.circuit(self.training_set[0])
            for gate in circuit.queue:
                if not isinstance(gate, TensorflowGate):
                    raise RuntimeError('SGD VQE requires native Tensorflow '
                                       'gates because gradients are not '
                                       'supported in the custom kernels.')

            sgd_options = {"nepochs": 5001,
                           "nmessage": 1000,
                           "optimizer": "Adamax",
                           "learning_rate": 0.5}
            if options is not None:
                sgd_options.update(options)

            # proceed with the training
            from qibo.config import K
            vparams = K.Variable(self.params)
            optimizer = getattr(K.optimizers, sgd_options["optimizer"])(
                learning_rate=sgd_options["learning_rate"])

            def opt_step():
                with K.GradientTape() as tape:
                    l = loss(vparams)
                grads = tape.gradient(l, [vparams])
                optimizer.apply_gradients(zip(grads, [vparams]))
                return l, vparams

            if compile:
                opt_step = K.function(opt_step)

            l_optimal, params_optimal = 10, self.params
            for e in range(sgd_options["nepochs"]):
                l, vparams = opt_step()
                if l < l_optimal:
                    l_optimal, params_optimal = l, vparams
                if e % sgd_options["nmessage"] == 0:
                    print('ite %d : loss %f' % (e, l.numpy()))

            result = self.cost_function(params_optimal).numpy()
            parameters = params_optimal.numpy()

        elif 'tf' in method:
            from qibo.tensorflow.gates import TensorflowGate
            circuit = self.circuit(self.domain[0])
            for gate in circuit.queue:
                if not isinstance(gate, TensorflowGate):
                    raise RuntimeError('SGD VQE requires native Tensorflow '
                                       'gates because gradients are not '
                                       'supported in the custom kernels.')

            # proceed with the training
            from qibo.config import K
            vparams = K.Variable(self.params)

            if 'bfgs' in method:
                def loss_gradient(x):
                    return tfp.math.value_and_gradient(lambda x: loss(x), x)
                if compile:
                    loss_gradient = K.function(loss_gradient)
                params_optimal = tfp.optimizer.bfgs_minimize(
                    loss_gradient, vparams)
            elif 'lbfgs' in method:
                def loss_gradient(x):
                    return tfp.math.value_and_gradient(lambda x: loss(x), x)
                if compile:
                    loss_gradient = K.function(loss_gradient)
                params_optimal = tfp.optimizer.lbfgs_minimize(
                    loss_gradient, vparams)
            elif 'nelder_mead' in method:
                params_optimal = tfp.optimizer.nelder_mead_minimize(
                    loss, initial_vertex=vparams)
            result = params_optimal.objective_value.numpy()
            parameters = params_optimal.position.numpy()

        else:
            # Newtonian approaches
            import numpy as np
            from scipy.optimize import minimize
            # n = self.hamiltonian.nqubits
            m = minimize(lambda p: loss(p).numpy(), self.params,
                         method=method, options=options)
            result = m.fun
            parameters = m.x

        return result, parameters

    def eval_test_set_fidelity(self):
        labels = [[0]] * len(self.test_set[0])
        for j, x in enumerate(self.test_set[0]):
            C = self.circuit(x)
            state = C.execute()
            fids = np.empty(len(self.target))
            for i, t in enumerate(self.target):
                fids[i] = fidelity(state, t)
            labels[j] = np.argmax(fids)

        return labels


    def paint_results(self):
        fig, axs = fig_template(self.name)
        guess_labels = self.eval_test_set_fidelity()
        colors_classes = get_cmap('tab10')
        norm_class = Normalize(vmin=0, vmax=10)
        x = self.test_set[0]
        x_0, x_1 = x[:, 0], x[:, 1]
        axs[0].scatter(x_0, x_1, c=guess_labels, s=2, cmap=colors_classes, norm=norm_class)
        colors_rightwrong = get_cmap('RdYlGn')
        norm_rightwrong = Normalize(vmin=-.1, vmax=1.1)

        checks = [int(g == l) for g, l in zip(guess_labels, self.test_set[1])]
        axs[1].scatter(x_0, x_1, c=checks, s=2, cmap=colors_rightwrong, norm=norm_rightwrong)

        fig.savefig('results/'+self.name+'/%s_layers/test_set.pdf'%self.layers)


    def paint_world_map(self):
        angles = np.zeros((len(self.test_set[0]), 2))
        from datasets import laea_x, laea_y
        fig, ax = world_map_template()
        colors_classes = get_cmap('tab10')
        norm_class = Normalize(vmin=0, vmax=10)
        for i, x in enumerate(self.test_set[0]):
            C = self.circuit(x)
            state = C.execute().numpy()
            angles[i, 0] = np.pi / 2 - np.arccos(np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2)
            angles[i, 1] = np.angle(state[1] / state[0])

        ax.scatter(laea_x(angles[:, 1], angles[:, 0]), laea_y(angles[:, 1], angles[:, 0]), c=self.test_set[1],
                          cmap=colors_classes, s=15, norm=norm_class)

        angles_0 = np.zeros(len(self.target))
        angles_1 = np.zeros(len(self.target))
        for i, state in enumerate(self.target):
            angles_0[i] = np.pi / 2 - np.arccos(np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2)
            angles_1[i] = np.angle(state[1] / state[0])

        ax.scatter(laea_x(angles_1, angles_0), laea_y(angles_1, angles_0), c=list(range(i + 1)),
                       cmap=colors_classes, s=500, norm=norm_class, marker='P')

        ax.axis('off')

        fig.savefig('results/'+self.name+'/%s_layers/world_map.pdf'%self.layers)

def fidelity(state1, state2):
    return tf.constant(np.abs(tf.reduce_sum(np.conj(state2) * state1))**2)