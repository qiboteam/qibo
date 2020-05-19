/**
 * Implementation of a binary mask.
 */
#include <math.h>

/**
 * Arguments:
 *      in_array: input array (preallocated)
 *      nqubits: number of qubits
 * Populates the in_array with the binary mask
 **/
void binary_matrix(unsigned char *in_array, int nqubits) {
  long int nn = pow(2, nqubits);
#pragma omp parallel for
  for (long int i = 0; i < nn; i++) {
    long int itmp = i;
    for (int j = 0; j < nqubits; j++) {
      in_array[i * nqubits + j] = itmp % 2;
      itmp = (long int)(itmp / 2);
    }
  }
}