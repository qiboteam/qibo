#ifndef KERNEL_TRANSPOSE_STATE_H_
#define KERNEL_TRANSPOSE_STATE_H_

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct TransposeStateFunctor {
  void operator()(const Device &d, const T* state, T* transposed_state,
                  int nqubits, const int* qubit_order) const;
};

}  // namespace functor

}  // namespace tensorflow

#endif  // KERNEL_TIME_TWO_H_
