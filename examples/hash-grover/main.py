#!/usr/bin/env python
import numpy as np
import functions
import argparse


def main(h_value, collisions, b):
    """Grover search for the instance defined by the file_name.
    Args:
        file_name (str): name of the file that contains the information of a 3SAT instance

    Returns:
        result of the Grover search and comparison with the expected solution if given.
    """
    q = 4
    m = 8
    rot = [1, 2]
    constant_1 = 5
    constant_2 = 9
    h = "{0:0{bits}b}".format(h_value, bits=b)
    #163 - 2 collisions
    #187 - 1 collision
    #133 - 3 collisions
    #113 - 4 collisions
    print('Target hash: {}\n'.format(h))
    if collisions:
        grover_it = int(np.pi*np.sqrt((2**8)/collisions)/4)
        result = functions.grover(q, constant_1, constant_2, rot, h, grover_it)
        #most_common = result.most_common(1)[0][0]
        most_common = result.most_common(collisions)
        #result = functions.check_hash(q, most_common, h, constant_1, constant_2, rot)
        if result:
            print('Solution found directly.\n')
            print('Preimages:')
            for i in most_common:
                print('   - {}'.format(i[0]))
            print()
            print('Total iterations taken: {}\n'.format(grover_it))
        else:
            print('Solution not found.\n')
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
    
