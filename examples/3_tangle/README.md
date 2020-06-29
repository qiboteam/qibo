# Measuring the tangle of three-qubit states
Based in the paper [Entropy 2020, 22(4), 436](http://dx.doi.org/10.3390/e22040436). Here a quick explanation is given.
For further details, go to the original source.

## Introduction

In this example we present a quantum circuit that transforms an unknown three-qubit state into its
canonical form, up to relative phases, given many copies of the original state. The circuit is made
of three single-qubit parametrized quantum gates, and the optimal values for the parameters are
learned in a variational fashion. Once this transformation is achieved, direct measurement of outcome
probabilities in the computational basis provides an estimate of the tangle, which quantifies genuine
tripartite entanglement. 

The description of entanglement in a three-qubit system belongs to the vast problem
of classifying and quantifying multipartite entanglement in a reliable way. Although the concept of
entanglement is of central importance in the fields of Quantum Information and Computation or
in Condensed Matter Physics, there is no known general theory of entanglement yet. As the
number of qubits increases, an exponentially large number of entanglement invariants under
local unitaries can be constructed, and different entanglement classes can be distinguished.
Furthermore, the possibility of measuring these entanglement quantifiers on actual states seems out
of reach for more than a few qubits.

The mainstream approach to deal with multipartite entanglement consists of considering
different bipartitions of the system of n qubits and analyze the entanglement that characterizes
them. In contradistinction to bipartite states, there is no simple equivalent to the Singular Value
Decomposition for tripartite systems. In that case, a canonical representation allows to set
several coefficients of the original state to zero and fix some of its relative phases through local unitaries.

When dealing with pure bipartite states, a variational quantum algorithm can be trained
on several copies of the original state in order to discover the local unitaries that reveal its Schmidt
form. Then, direct measurements in the computational basis provide the eigenvalues of the Singular
Value Decomposition, which in turn are used to compute entanglement entropies. Here, we shall
explore a similar strategy to obtain the canonical form and measure the tangle of three-qubit states.
We propose a quantum circuit made of three local unitaries, each acting on one of the qubits. The action
of these unitaries cast the state into its canonical form, up to relative phases, and can be determined in
a variational way. Once this transformation is achieved, the frequencies of measurement outputs in
the computational basis are used to compute the tangle of the three-qubit system, which quantifies
genuine tripartite entanglement.

This method is different to the standard procedure for measuring the tangle of a given quantum state, that involves 
performing quantum tomography. Such method requires knowledge of 4^3 observables, obtained through
3^3 different measurement settings. In contrast, the algorithm herein proposed only needs one
measurement setting, namely measuring in the computational basis, but several copies of the state
are demanded for the optimization. Overall, both methods involve a similar number of copies.
However, our proposal also returns the canonical form of the state.

## Mathematical description of the problem

An arbitrary three-qubit quantum state in its most general form can be written as 
$$
\vert \psi \rangle = \sum_{i,j,k = 0}^{1} t_{ijk}\vert ijk \rangle,
$$
where $\{| ijk \rangle \}$ are the computational-basis states, and the complex coefficients in the tensor $t_{ijk}$ obey
a normalization relation. A genuine entanglement measure of a three-qubit system $\vert \psi \rangle $ is the
tangle, denoted by $\tau$. It can be obtained from Cayley’s hyperdeterminant, which is a generalization
of a square-matrix determinant. To be precise, $\tau  = 4 | {\rm Hdet} ( t_{ijk} )| $. 

We recall now a property that is a cornerstone of entanglement theory. No entanglement structure is affected 
by Local Unitary operations. The algorithm we propose here consists in use local operations to transform the 
quantum state into its pseudo-canonical form by setting several amplitudes of the original state to zero. In this case, 
we begin with a three-qubit quantum state that has $2 * 2^3 - 2 = 14$ degrees of freedom. As it is possible to apply
three local unitary gates with 3 parameters each, a total amount of $9$ parameters can be fixed. Therefore, it is 
possible to transform any three-qubit state into a canonical form with only $5$ parameters preserving all
the entanglement properties. The shape of this canonical form will be 
$$
\vert \psi \rangle = \lambda_0 \vert 000 \rangle + \lambda_1 e^{i\phi} \vert 100 \rangle + \lambda_2 \vert 101 \rangle + \lambda_3 \vert 110 \rangle + \lambda_4 \vert 111 \rangle
$$

The tangle of the canonical form is equal to that of the original form. However, due to the choice of the local
unitary operations, it is possible to check that

$$
\tau = 4 \lambda_0^2 \lambda_4^2
$$

## A quantum algorithm for measuring the tangle

The algorithm proposed is easily described through a quantum circuit. An unknown three-qubit quantum state comes into
the quantum computer. A local unitary is applied to each partition of the state. The exact parameters of the 
local unitary gates are learnt in a variational way. After this transformations, the state is ready to
measure and obtain an estimate of the tangle. 
<img src="circuit.png"; style="float: left; margin-right: 10px;" />

We need to define a strategy to obtain the local operations performing the transformation. This process will be driven 
by minimizing a cost function. 
$$
(\vec \theta_A, \vec \theta_B, \vec \theta_C)_{\rm opt} = {\rm argmin}\left( \mathcal{C}(\vec \theta_A, \vec \theta_B, \vec \theta_C)\right)
$$
where the cost function is
$$
\mathcal{C}(\vec \theta_A, \vec \theta_B, \vec \theta_C) =  \sum_i \,|\bra{i\,}\, U(\vec \theta_A, \vec \theta_B, \vec \theta_C) \ket\psi_{ABC} \,|^2 \quad,\quad i\in\{001,010,011\}.
$$

Notice that this method does not transform the unknown state into its canonical form, but in an up-to-phases
canonical form defined as 

$$
\ket{\tilde{\varphi}} = \lambda_0 \ket{000} +  \lambda_1 e^{i\phi_1}\ket{100} +  \lambda_2 e^{i\phi_2}\ket{101} + \lambda_3 e^{i\phi_3}\ket{110} + \lambda_4 e^{i\phi_4}\ket{111}
$$

Such transformation is less restrictive than the canonical transformation. Therefore, there exist many possible optimal parameters.

Once the optimal parameters are obtained, it is straightforward to measure the tangle $\tau$ in an actual quantum computer. This quantity will be equal to 
\begin{equation}
\tau = 4 \,|\braket{000}{\tilde{\varphi}}\braket{111}{\tilde{\varphi}}|^2 = 4 \,P_{000} P_{111}\,,
\end{equation}
where $P_{ijk}$ is the probability of measuring $\ket{ijk}$. The statistical additive error of $P_{ijk}$ is given by 
the sampling process of a multinomial distribution, that is, $\sqrt{P_{ijk}(1 - P_{ijk}) / M}$, where $M$ is the number 
of measurements.

We propose a manner to mitigate random errors occurring when computing the tangle, via post-selection. After the 
optimization is completed, and a low value of the cost function is obtained, it is licit to assume that $\ket\psi_{ABC}$ 
has been properly transformed into $\ket{\tilde{\varphi}}$. Thus, if the outcome of a measurement is either 
$\ket{001}, \ket{010} {\rm \, or \,} \ket{011}$ after the transformation into the up-to-phases canonical form, 
it is due to an error in the circuit. In this case, this outcome can be discarded.

## Usage of the example

The `tangle.py` file contains all the functions needed to execute this example. `N` random three-qubit states are
created and transformed into the up-to-phases canonical form through the `canonize` function. The value of the cost function
and the parameters defining the transformation are obtained. Then, exact and measured tangles are computed and plotted in 
a figure. 

```
tangles = np.empty(N)
opt_tangles = np.empty(N)

for i in range(N):
    state = create_random_state(i)
    tangles[i] = compute_random_tangle(i)
    fun, params = canonize(state, p=p, shots=shots)
    opt_tangles[i] = canonical_tangle(state, params, post_selection=False)
    print(fun, opt_tangles[i])

fig, ax = plt.subplots()
ax.scatter(tangles, opt_tangles, s=20)
ax.plot([0., 1.], [0., 1.], color='black')
ax.set(xlabel='Exact tangle', ylabel='Measured tangle')
plt.show()
```