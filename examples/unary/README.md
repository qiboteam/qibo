# Quantum unary approacih to option pricing

Based in the paper [arXiv:1912.01618](https://arxiv.org/abs/1912.01618). Here a quick explanation is given.
For further details and references, go to the original source.

## Introduction

Quantum computing provides new strategies to address problems that nowadays are considered difficult to solve by 
classical means. The first quantum algorithms showing a theoretical advantage over their classical counterparts are 
known since the 1990s.
Nevertheless, current quantum devices are not powerful enough to run quantum algorithms that are able to compete against
state-of-the-art classical algorithms. Indeed, available quantum computers are in their Noisy Intermediate-Scale 
Quantum (NISQ) stage, as errors due to decoherence, noisy gate application or error read-out limit the performance of 
these new machines. These NISQ devices may nonetheless be useful tools for a variety of applications due to the 
introduction of hybrid variational methods. Some exact, non-variational, quantum algorithms are also well suited for NISQ devices. 

A field that is expected to be transformed by the improvement of quantum devices is quantitative finance. In recent 
years, there has been a surge of new methods and algorithms dealing with financial problems using quantum resources, 
such as optimization problems, which are in general hard.

Notably, pricing of financial derivatives is a prominent problem, where many of its computational obstacles are suited 
to be overcome via quantum computation. In this example we will deal with options, which are a particular type of 
financial derivatives. Options are contracts that allow the holder to buy (__call__) or sell (__put__) some asset at a pre-established price (\textit{strike}), or at a future point in time (\textit{maturity date}). The payoff of an option depends on the evolution of the asset's price, which follows a stochastic process. A simple, yet successful model for pricing options is the Black-Scholes model \cite{blackscholes-black1973}. This is an analytically-solvable model that predicts the asset's price evolution to follow a log-normal probability distribution, at a future time $t$. Then, a specified payoff function, which depends on the particular option considered, has to be integrated over this distribution to obtain the expected return of the option. Current classical algorithms rely on computationally-costly Monte Carlo simulations to estimate the expected return of options.

A few quantum algorithms have been proposed to improve on classical option pricing \cite{qfinance-stamatopoulos2019, qfinance-rebentrost2018, qfinance-woerner2019}. It has been shown that quantum computers can provide a quadratic speedup in the number of quantum circuit runs as compared to the number of classical Monte Carlo runs needed to reach a certain precision in the estimation. The basic idea is to exploit quantum Amplitude Estimation \cite{amplitude_estimation-brassard2002, counting-aaronson2019, montecarlo-montanaro2015quantum}. Nonetheless, this can only be achieved when an efficient way of loading the probability distribution of the asset price is available. The idea of using quantum Generative Adversarial Networks (qGANs) \cite{qGAN-lloyd2018, qGAN-dallaire2018} to address this issue has been analyzed \cite{qGAN-zoufal2019}.

In the following, we propose a quantum algorithm for option pricing. The key new idea is to construct a quantum circuit that works in the unary basis of the asset's value, \ie in a subspace of the full Hilbert space of $n$ qubits. Then, the evolution of the asset's price is computed using an amplitude distributor module. Furthermore, the computation of the payoff greatly simplifies. A third part of the algorithm is common to previous approaches, namely it uses Amplitude Estimation. The unary scheme brings further advantage since it allows for a post-selection strategy that results in error mitigation. Let us recall that error mitigation techniques are likely to be crucial for the success of quantum algorithms in the NISQ era. On the negative side, the number of qubits in the unary algorithm scales linearly with the number of bins, while in the binary algorithm it is logarithmic with the target precision. This results in a worse asymptotic scaling for the unary algorithm. Yet, our estimates for the number of gates indicate that the crossing point between these two is located at a number of qubits that renders a good precision ($< 1\%$) for real-world applications. Moreover, the performance of the unary algorithm is more robust to noise, as we show in simulations. Hence, our proposal seems to be better suited to be run on NISQ devices. Unary representations have also been considered in previous works \cite{spectral-poulin2018, babbush2018,steudtner2019}.

We will illustrate our new algorithm focusing on a simple European option, whose payoff is a function of only the asset's price at maturity date, the only date the contract can be executed at. This straightforward example has been chosen as a proof of concept for this new approach. We will compare the performance of our unary quantum circuit with the previous binary quantum circuit proposal, for a fixed precision or binning of the probability distribution.

The paper is organized as follows. We first introduce the basic ideas on option pricing, both classical and quantum, in Sec. \ref{sec:background}. The unary quantum algorithm is presented and analyzed in Sec. \ref{sec:unary}. We devote Sec. \ref{sec:un-vs-bin} to outline the circuit specifications and compare them for the unary and binary quantum algorithms. Sec. \ref{sec:simulations} is dedicated to describe the results obtained by means of classical simulations for both algorithms. Lastly, conclusions are drawn in Sec. \ref{sec:conclusions}. Further details on several topics are described in the Appendices.