#ifndef KERNEL_MEASUREMENTS_H_
#define KERNEL_MEASUREMENTS_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

namespace functor {

template <typename Device, typename Tint, typename Tfloat>
struct MeasureFrequenciesFunctor {
  void operator()(const Device &d, Tint* frequencies, const Tfloat* probs,
                  Tint nshots, int nqubits, int user_seed);
};

}  // namespace functor

}  // namespace tensorflow

#endif  // KERNEL_TIME_TWO_H_
