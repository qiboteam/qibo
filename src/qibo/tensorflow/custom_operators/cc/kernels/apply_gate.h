#ifndef KERNEL_APPLY_GATE_H_
#define KERNEL_APPLY_GATE_H_

#include "tensorflow/core/framework/types.h"

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct ApplyGateFunctor {
  void operator()(const OpKernelContext* context, const Device& d, T* state,
                  const T* gate, int nqubits, int target);
};

}  // namespace functor

}  // namespace tensorflow

#endif  // KERNEL_APPLY_GATE_H_
