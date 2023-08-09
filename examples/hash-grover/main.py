#!/usr/bin/env python
import numpy as np
import functions
import argparse


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
        raise ValueError("Hash should be at maximum an 8-bit number but given value contains {} bits.".format(len(h)))
    print('Target hash: {}\n'.format(h))
    if collisions:
        grover_it = int(np.pi*np.sqrt((2**8)/collisions)/4)
        result = functions.grover(q, constant_1, constant_2, rot, h, grover_it)
        most_common = result.most_common(collisions)
        print('Solutions found:\n')
        print('Preimages:')
        for i in most_common:
            if functions.check_hash(q, i[0], h, constant_1, constant_2, rot):
                print('   - {}\n'.format(i[0]))
            else:
                print('   Incorrect preimage found, number of given collisions might not match.\n')
        print('Total iterations taken: {}\n'.format(grover_it))
    else:
        measured, total_iterations = functions.grover_unknown_M(q, constant_1, constant_2, rot, h)
        print('Solution found in an iterative process.\n')
        print('Preimage: {}\n'.format(measured))
        print('Total iterations taken: {}\n'.format(total_iterations))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", default=163, type=int)
    parser.add_argument("--bits", default=7, type=int)
    parser.add_argument("--collisions", default=None, type=int)
    args = vars(parser.parse_args())
    main(args.get('hash'), args.get('collisions'), args.get('bits'))
    
