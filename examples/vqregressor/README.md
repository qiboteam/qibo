# Parameter Shift Rule for an hardware-compatible Variational Quantum Regressor

Code at: [https://github.com/qiboteam/qibo/tree/vqregressor/examples/vqregressor](https://github.com/qiboteam/qibo/tree/vqregressor/examples/vqregressor)

### Problem overview

We want to tackle a simple one dimensional regression problem using a single qubit Variational Quantum Circuit (VQC) as model,
initialized using a [re-uploading strategy](https://arxiv.org/abs/1907.02085). In particular, in this example we
fit the function $y = \sin (2x)$, picking the $x$ points from the interval $\mathcal{I}=[-1,1]$.
The optimization is performed using an [Adam](https://arxiv.org/abs/1412.6980) optimizer.
It needs the circuit's gradients, which we evaluate through the [Parameter Shift Rule](https://arxiv.org/abs/1811.11184) (PSR).

A method like this is
needed because in quantum computation we can't perform the [Back-Propagation Algorithm](https://www.nature.com/articles/323533a0) on the hardware:
in that case the values of the target function in the middle of the propagation are needed, but for evaluating them on the hardware we have to measure,
and measuring we provoke the collapse of the system and the loss of all the information wealth. The PSR provide us with a numerical tool, with which we
can perform a gradient descent even on the physical qubit.

### The Parameter Shift Rule in a nutshell

Let's consider a circuit $\mathcal{U}(\vec{\theta})$, in which we build up a gate of the form $\mathcal{G}=\exp \bigl[-i\mu G \bigr]$ with $\mu \in \vec{\theta}$,
(which represents an unitary operator with at most two eigenvalues $\pm r$), and an observable $B$.
Finally, let $q_f$ be the state we obtain by applying $\mathcal{U}$ to $| 0 \rangle$.

We are interested in evaluating the gradients of the following expression:

$$ f(\mu) \equiv \langle q_f | B | q_f \rangle,$$

where we specify that $f$ depends directly on $\mu$. We are interested in this result because the expectation value of $B$ is typically involved
in computing predictions in quantum machine learning problems. The PSR allows us to calculate the derivative of $f(\mu)$ with respect to $\mu$ evaluating
$f$ twice more:

$$ \partial_{\mu} f(\mu) = r \bigl( f(\mu^+) - f(\mu^-) \bigr), $$

where $\mu^{\pm}=\mu \pm s$ and $s = \pi/4 r$. Finally, if we pick $G$ from the rotations generators $\frac{1}{2}$ { $\sigma_x, \sigma_y, \sigma_z$ },
we can use $s=\pi/2$ and $r=1/2$.

In the end, we have to use PSR into a gradient descent strategy. We choose an [MSE loss function](https://en.wikipedia.org/wiki/Mean_squared_error), which leads to the following explicit formula:

$$ \partial_{\mu} J_{mse} = 2 \langle B \rangle_{x_j} \partial_{\mu} \langle B \rangle_{x_j} - 2y\langle B \rangle_{x_j},  $$

where we indicate with the subscript $x_j$ the dipendence of $J$ on $x_j$ and $y$ is the correct label of $x_j$ under the true law.

### This example

As mentioned above, we use a Variational Quantum Circuit based on a re-uploading strategy. In particular, we use the following architecture:

![ansatz](https://github.com/qiboteam/qibo/blob/vqregressor/examples/vqregressor/images/ansatz.png)

At the end of the circuit execution we perform a measurement on the qubit. After $N_{shots}$ measurements, we use the difference of the probabilities
of occurrence of the two states $|0 \rangle$ and $| 1 \rangle$ as estimator for $y$.

### How to use it?

In this example we use only two files:

- `vqregressor.py` contains the variational quantum regressor's implementation, with all the methods required for the optimization;
- `main.py` contains a commented example of usage of the regressor.

The user can change the target function modifying the method `vqregressor.label_points`, in which the true law is written and normalized. Once in the folder, one have to run a command like the following:

`python3 main.py --layers 1 --learning_rate 0.05 --epochs 200 --batches 1 --ndata 30 --J_treshold 1e-4`

for performing an optimization. At the end of the process it shows a plot containing true labels of the training sample and the predictions purposed
by the model in a form like the following:

![results](https://github.com/qiboteam/qibo/blob/vqregressor/examples/vqregressor/images/results.png)
