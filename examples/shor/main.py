#!/usr/bin/env python
import numpy as np
import functions
import argparse
import fractions


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
        raise ValueError(f'Input number {N} needs to be odd for it to be a hard problem.')
    f = functions.find_factor_of_prime_power(N)
    if f is not None:
        raise ValueError(f'Input number {N} is a prime power with factor {f}.')
    n = int(np.ceil(np.log2(N)))
    for i in range(times):
        if A:
            a = A
        else:
            a = np.random.randint(2, N - 1)
        f = np.gcd(a, N)
        if 1 < f < N:
            print(f"N = {N} and a = {a} share factor {f}. Trying again for quantum instance.\n")
            print('-'*60+'\n')
            continue
        print(f"Using Shor's algorithm to factorize N = {N} with a = {a}.\n")
        if semiclassical:
            s = functions.quantum_order_finding_semiclassical(N, a)
        else:
            s = functions.quantum_order_finding_full(N, a)
        sr = s/2**(2*n)
        r = fractions.Fraction.from_float(sr).limit_denominator(N).denominator
        print(f'Quantum circuit outputs r = {r}.\n')
        for i in range(1):
            if r % 2 != 0:
                print('The value found for r is not even. Trying again.\n')
                print('-'*60+'\n')
                continue
            if a**(r//2) == -1%N:
                print('Unusable value for r found. Trying again.\n')
                print('-'*60+'\n')
                continue
            f1 = np.gcd((a**(r//2))-1, N)
            f2 = np.gcd((a**(r//2))+1, N)
            if (f1 == N or f1 == 1) and (f2 == N or f2 == 1):
                print(f'Trivial factors 1 and {N} found. Trying again.\n')
                print('-'*60+'\n')
            elif f1 != 1 and f1 != N:
                print(f'Found as factors for {N}:  {f1}  and  {N//f1}.\n')
                return f1, N//f1
                break
            elif f2 != 1 and f2 != N:
                print(f'Found as factors for {N}:  {f2}  and  {N//f2}.\n')
                return f2, N//f2
                break
        if enhance:
            for c in range(1, 5):
                ss = s+c
                srr = ss/2**(2*n)
                rr = fractions.Fraction.from_float(srr).limit_denominator(N).denominator
                if rr == r:
                    continue
                print(f'Checking for near values outputs r = {rr}.\n')
                if rr % 2 != 0:
                    print('The value found for r is not even. Trying again.\n')
                    print('-'*60+'\n')
                    continue
                if a**(rr//2) == -1%N:
                    print('Unusable value for r found. Trying again.\n')
                    print('-'*60+'\n')
                    continue
                f1 = np.gcd((a**(rr//2))-1, N)
                f2 = np.gcd((a**(rr//2))+1, N)
                if (f1 == N or f1 == 1) and (f2 == N or f2 == 1):
                    print(f'Trivial factors 1 and {N} found. Trying again.\n')
                    print('-'*60+'\n')
                    continue
                elif f1 != 1 and f1 != N:
                    print(f'Found as factors for {N}:  {f1}  and  {N//f1}.\n')
                    return f1, N//f1
                    break
                elif f2 != 1 and f2 != N:
                    print(f'Found as factors for {N}:  {f2}  and  {N//f2}.\n')
                    return f2, N//f2
                    break
            for c in range(1, 5):
                ss = s-c
                srr = ss/2**(2*n)
                rr = fractions.Fraction.from_float(srr).limit_denominator(N).denominator
                if rr == r:
                    continue
                print(f'Checking for near values outputs r = {rr}.\n')
                if rr % 2 != 0:
                    print('The value found for r is not even. Trying again.\n')
                    print('-'*60+'\n')
                    continue
                if a**(rr//2) == -1%N:
                    print('Unusable value for r found. Trying again.\n')
                    print('-'*60+'\n')
                    continue
                f1 = np.gcd((a**(rr//2))-1, N)
                f2 = np.gcd((a**(rr//2))+1, N)
                if (f1 == N or f1 == 1) and (f2 == N or f2 == 1):
                    print(f'Trivial factors 1 and {N} found. Trying again.\n')
                    print('-'*60+'\n')
                    continue
                elif f1 != 1 and f1 != N:
                    print(f'Found as factors for {N}:  {f1}  and  {N//f1}.\n')
                    return f1, N//f1
                    break
                elif f2 != 1 and f2 != N:
                    print(f'Found as factors for {N}:  {f2}  and  {N//f2}.\n')
                    return f2, N//f2
                    break
            print('Checking multiples.\n')
            for c in range(2, n):
                rr = c*r
                if rr % 2 != 0:
                    print('The value found for r is not even. Trying again.\n')
                    print('-'*60+'\n')
                    continue
                if a**(rr//2) == -1%N:
                    print('Unusable value for r found. Trying again.\n')
                    print('-'*60+'\n')
                    continue
                f1 = np.gcd((a**(rr//2))-1, N)
                f2 = np.gcd((a**(rr//2))+1, N)
                if (f1 == N or f1 == 1) and (f2 == N or f2 == 1):
                    print(f'Trivial factors 1 and {N} found. Trying again.\n')
                    print('-'*60+'\n')
                    continue
                if f1 != 1 and f1 != N:
                    print(f'With multiple r = {rr}.\n')
                    print(f'Found as factors for {N}:  {f1}  and  {N//f1}.\n')
                    return f1, N//f1
                    break
                if f2 != 1 and f2 != N:
                    print(f'With multiple r = {rr}.\n')
                    print(f'Found as factors for {N}:  {f2}  and  {N//f2}.\n')
                    return f2, N//f2
                    break
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
