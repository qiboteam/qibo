#ifndef KERNEL_SPLIT_STATE_H_
#define KERNEL_SPLIT_STATE_H_

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct SplitStateFunctor {
  void operator()(const Device &d, const T *state, T* pieces, int nqubits,
                  const int* global_qubits, int nglobal) const;
};

}  // namespace functor

}  // namespace tensorflow

#endif  // KERNEL_TIME_TWO_H_
