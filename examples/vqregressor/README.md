# Variational Quantum Regression through Parameter Shift Rule

We want to tackle a simple one dimensional regression problem using a Variational Quantum Circuit (VQC) as model. In particular, in this example we 
fit the function $y = \sin (2x)$, picking $x$ points from the interval $\mathcal{I}=[-1,1]$. 
The optimization is performed using an [Adam](https://arxiv.org/abs/1412.6980) optimizer.
It needs the circuit's gradients, which we evaluate through the [Parameter Shift Rule](https://arxiv.org/abs/1811.11184) (PSR). 

## The Parameter Shift Rule in a nutshell

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
