# Double-bracket quantum algorithms

Qibo features a model implementing double-bracke quantum algorithms (DBQAs) which are helpful for approximating eigenstates based on the ability to run the evolution under the input Hamiltonian.

More specifically, given a Hamiltonian $H_0$, how can we find a circuit which after applying to the reference state (usually $|0\rangle^{\otimes L}$ for $L$ qubits) will approximate an eigenstate?

A standard way is to run variational quantum circuits. For example, Qibo already features  the `VQE` model [2] which provides the implementation of the variational quantum eigensolver framework.
DBQAs allow to go beyond VQE in that they take a different approach to compiling the quantum circuit approximating the eigenstate.

## What is the unitary of DBQA?

Given $H_0$ we begin by assuming that we were given a diagonal and hermitian operator $D_0$ and a time $s_0$.
The `dbi` module provides numerical strategies for selecting them.
For any such choice we define the bracket
$$ W_0 = [D_0, H_0]$$
and the double-bracket rotation (DBR) of the input Hamiltonian to time $s$
$$H_0(s) = e^{sW} H e^{- s W}$$

### Why are double-bracket rotations useful?
We can show that the magnitude of the off-diagonal norms will decrease.
For this let us set the notation that $\sigma(A)$ is the restriction to the off-diagonal of the matrix A.
In `numpy` this can be implemented by `\sigma(A) = A-np.diag(A)`. In Qibo we implement this as
https://github.com/qiboteam/qibo/blob/8c9c610f5f2190b243dc9120a518a7612709bdbc/src/qibo/models/dbi/double_bracket.py#L145-L147
which is part of the basic `DoubleBracketIteration` class in the `dbi` module.

With this notation we next use the Hilbert-Schmidt scalar product and norm to measure the progress of diagonalization
 $$||\sigma(H_0(s))||^2- ||\sigma (H_0 )||^2= -2s \langle W, [H,\sigma(H)\rangle+O(s^2)$$
This equation tells us that as long as the scalar product $\langle W, [H,\sigma(H)\rangle$ is positive then after the DBR the magnitude of the off-diagonal couplings in $H_0(s)$ is less than in $H_0$.

For the implementation of the DBR unitary $U_0(s) = e^{-s W_0}$ see
https://github.com/qiboteam/qibo/blob/363a6e5e689e5b907a7602bd1cc8d9811c60ee69/src/qibo/models/dbi/double_bracket.py#L68

### How to choose $D$?

For theoretical considerations the canonical bracket is useful.
For this we need the notation of the dephasing channel $\Delta(H)$ which is equivalent to `np.diag(h)`.
 $M = [\Delta(H),\sigma(H)]= [H,\sigma(H)]= [\Delta(H),H]$
 The canonical bracket appears on its own in the monotonicity relation above and gives an unconditional reduction of the magnitude of the off-diagonal terms
 $$||\sigma(H_0(s))||^2- ||\sigma (H_0 )||^2= -2s ||M||^2+O(s^2)$$
- the multi qubit Pauli Z generator with $Z(\mu) = (Z_1)^{\mu_1}\ldots (Z_L)^{\mu_L}$ where we optimize over all binary strings $\mu\in \{0,1\}^L$
- the magnetic field $D = \sum_i B_i Z_i$
- the two qubit Ising model $D  = \sum_i B_i Z_i + \sum_{i,j} J_{i,j} Z_i Z_j$, please follow the tutorial by Matteo and use the QIBO ising model for that with $h=0$


### How to choose s?

The theory above shows that in generic cases the DBR will have a linear diagonalization effect (as quantified by $||\sigma(H_0(s))||$).
This can be further expanded with Taylor expansion and the Qibo implementation comes with methods for fitting the first local minimum.
Additionally a grid search for the optimal step is provided for an exhaustive evaluation and hyperopt can be used for a more efficient 'unstructured' optimization; additionally simulated annealing is provided which sometimes outperforms hyperopt (and grid search), see example notebooks.
The latter methods may output DBR durations $s_k$ which correspond to secondary local minima.





[1] https://arxiv.org/abs/2206.11772

[2] https://github.com/qiboteam/vqe-sun
