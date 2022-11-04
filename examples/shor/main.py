#!/usr/bin/env python
import argparse
import fractions

import functions
import numpy as np


def main(N, times, A, semiclassical, enhance):
    """Factorize a number using Shor's algorithm.
    Args:
        N (int): number to factorize.
        times (int): maximum number of repetitions allowed to the algorithm.
        A (int) (optional): fix the number a used in the quantum order finding algorithm.
        semiclassical (bool): use the semiclassical iQFT method to reduce number of qubits.
        enhance (bool): classically enhance the quantum output for higher probability of success.

    Returns:
        f1 (int), f2 (int): factors of the given number N.
    """
    if N % 2 == 0:
        raise ValueError(
            f"Input number {N} needs to be odd for it to be a hard problem."
        )
    f = functions.find_factor_of_prime_power(N)
    if f is not None:
        raise ValueError(f"Input number {N} is a prime power with factor {f}.")
    n = int(np.ceil(np.log2(N)))
    for i in range(times):
        if A:
            a = A
        else:
            a = np.random.randint(2, N - 1)
        f = np.gcd(a, N)
        if 1 < f < N:
            print(
                f"N = {N} and a = {a} share factor {f}. Trying again for quantum instance.\n"
            )
            print("-" * 60 + "\n")
            continue
        print(f"Using Shor's algorithm to factorize N = {N} with a = {a}.\n")
        if semiclassical:
            s = functions.quantum_order_finding_semiclassical(N, a)
        else:
            s = functions.quantum_order_finding_full(N, a)
        sr = s / 2 ** (2 * n)
        r = fractions.Fraction.from_float(sr).limit_denominator(N).denominator
        print(f"Quantum circuit outputs r = {r}.\n")
        factors = functions.find_factors(r, a, N)
        if factors:
            return factors
        if enhance:
            for c in range(-4, 5):
                sr = (s + c) / 2 ** (2 * n)
                rr = fractions.Fraction.from_float(sr).limit_denominator(N).denominator
                if rr == r:
                    continue
                print(f"Checking for near values outputs r = {rr}.\n")
                factors = functions.find_factors(rr, a, N)
                if factors:
                    return factors
            print("Checking multiples.\n")
            for c in range(2, n):
                rr = c * r
                factors = functions.find_factors(rr, a, N)
                if factors:
                    return factors
        continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=15, type=int)
    parser.add_argument("--times", default=10, type=int)
    parser.add_argument("--A", default=None, type=int)
    parser.add_argument("--semiclassical", action="store_true")
    parser.add_argument("--enhance", action="store_true")
    args = vars(parser.parse_args())
    main(**args)
