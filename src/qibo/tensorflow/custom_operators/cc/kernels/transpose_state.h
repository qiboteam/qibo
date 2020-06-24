#ifndef KERNEL_TRANSPOSE_STATE_H_
#define KERNEL_TRANSPOSE_STATE_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct TransposeStateFunctor {
  void operator()(const OpKernelContext* context, const Device &d,
                  const std::vector<T*> state, T* transposed_state,
                  int nqubits, int ndevices, const int* qubit_order) const;
};

}  // namespace functor

}  // namespace tensorflow

#endif  // KERNEL_TIME_TWO_H_
