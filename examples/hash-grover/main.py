#!/usr/bin/env python
import argparse

import functions
import numpy as np


def main(h_value, collisions, b):
    """Grover search for preimages of a given hash value
    Args:
        h_value (int): hash value to be converted to binary string.
        collisions (int): number of collisions or None if unknown.
        b (int): number of bits to be used for the hash string.

    Returns:
        result of the Grover search and checks if it has found a correct preimage.
    """
    q = 4
    m = 8
    rot = [1, 2]
    constant_1 = 5
    constant_2 = 9
    h = "{0:0{bits}b}".format(h_value, bits=b)
    if len(h) > 8:
        raise ValueError(
            f"Hash should be at maximum an 8-bit number but given value contains {len(h)} bits."
        )
    print(f"Target hash: {h}\n")
    if collisions:
        grover_it = int(np.pi * np.sqrt((2**8) / collisions) / 4)
        result = functions.grover(q, constant_1, constant_2, rot, h, grover_it)
        most_common = result.most_common(collisions)
        print("Solutions found:\n")
        print("Preimages:")
        for i in most_common:
            if functions.check_hash(q, i[0], h, constant_1, constant_2, rot):
                print(f"   - {i[0]}\n")
            else:
                print(
                    "   Incorrect preimage found, number of given collisions might not match.\n"
                )
        print(f"Total iterations taken: {grover_it}\n")
    else:
        measured, total_iterations = functions.grover_unknown_M(
            q, constant_1, constant_2, rot, h
        )
        print("Solution found in an iterative process.\n")
        print(f"Preimage: {measured}\n")
        print(f"Total iterations taken: {total_iterations}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", default=163, type=int)
    parser.add_argument("--bits", default=7, type=int)
    parser.add_argument("--collisions", default=None, type=int)
    args = vars(parser.parse_args())
    main(args.get("hash"), args.get("collisions"), args.get("bits"))
