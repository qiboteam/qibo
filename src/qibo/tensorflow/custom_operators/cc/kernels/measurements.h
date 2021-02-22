#ifndef KERNEL_MEASUREMENTS_H_
#define KERNEL_MEASUREMENTS_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct MeasurementFrequenciesFunctor {
  void operator()(const Device &d, T *in, int64 size);
};

}  // namespace functor

}  // namespace tensorflow

#endif  // KERNEL_TIME_TWO_H_
