from qibo.models import Circuit
from qibo import gates
import numpy as np
from datasets import create_dataset, create_target
from qibo.hamiltonians import Hamiltonian
from qibo.config import matrices
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

np.random.seed(0)

class single_qubit_classifier:
    def __init__(self, name, layers, grid=11, test_samples=1000):
        #Circuit.__init__(self, nqubits)
        self.layers = layers
        self.training_set = create_dataset(name, grid=grid)
        self.test_set = create_dataset(name, samples = test_samples)
        self.target = create_target(name)
        self.hamiltonian = self.create_hamiltonian()
        self.params = np.random.randn(layers * 4)

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
        cf = .5 * (1 - np.dot(values, self.target_values[y])) ** 2
        return cf

    def cost_function_observables(self, params):
        self.set_parameters(params)
        cf = 0
        for x, y in self.training_set:
            cf += self.cost_function_one_point_observables(x, y)
        cf /= len(self.training_set[0])
        return cf

    def minimize(self, method='BFGS', options=None, compile=True, fidelity=True):
        if fidelity:
            loss = self.cost_function_fidelity
        else:
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
            state = self.circuit(x)
            fids = np.empty(len(self.target))
            for i, t in enumerate(self.target):
                fids[i] = fidelity(state, t)
            labels[j] = np.argmax(fids)

        return labels


    def paint_results(self):
        # Pintar los resultados
        pass


    # La minimización puede ser como en el VQE de qibo, la misma estructura es válida, y todos los minimizadores deberían funcionar bien
    # Otro cantar será saber cuál es el minimizador bueno

    def paint_representation_1D(self):
        fig, axs = plt.subplots(nrows=self.num_functions)

        if self.num_functions == 1:
                axs.plot(self.domain, self.functions[0](self.domain), color='black', label='Target Function')
                outcomes = np.zeros_like(self.domain)
                for j, x in enumerate(self.domain):
                    C = self.circuit(x)
                    state = C.execute()
                    outcomes[j] = self.hamiltonian[0].expectation(state)

                axs.plot(self.domain, outcomes, color='C0',label='Approximation')
                axs.legend()
        else:
            for i in range(self.num_functions):
                axs[i].plot(self.domain, self.functions[i](self.domain).flatten(), color='black')
                outcomes = np.zeros_like(self.domain)
                for j, x in enumerate(self.domain):
                    C = self.circuit(x)
                    state = C.execute()
                    outcomes[j] = self.hamiltonian[i].expectation(state)

                axs[i].plot(self.domain, outcomes, label=self.measurements[i])
                axs[i].legend()

        plt.show()

    def paint_representation_2D(self):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if self.num_functions == 1:
            ax = fig.gca(projection='3d')
            print('shape', self.target[0].shape)
            ax.plot_trisurf(self.domain[:, 0], self.domain[:, 1], self.target[:, 0], label='Target Function',  linewidth=0.2, antialiased=True)
            outcomes = np.zeros_like(self.domain)
            for j, x in enumerate(self.domain):
                C = self.circuit(x)
                state = C.execute()
                outcomes[j] = self.hamiltonian[0].expectation(state)

            ax.plot_trisurf(self.domain[:, 0], self.domain[:, 1], outcomes[:, 0] + 0.1,label='Approximation',  linewidth=0.2, antialiased=True)
            #ax.legend()
        else:
            for i in range(self.num_functions):
                ax = fig.add_subplot(1,1,i + 1, projection='3d')
                ax.plot(self.domain, self.functions[i](self.domain).flatten(), color='black')
                outcomes = np.zeros_like(self.domain)
                for j, x in enumerate(self.domain):
                    C = self.circuit(x)
                    state = C.execute()
                    outcomes[j] = self.hamiltonian[i].expectation(state)

                ax.scatter(self.domain[:, 0], self.domain[:, 1], outcomes, color='C0', label=self.measurements[i])
                ax.legend()

        plt.show()

def fidelity(state1, state2):
    return tf.constant(np.abs(tf.reduce_sum(np.conj(state2) * state1))**2)