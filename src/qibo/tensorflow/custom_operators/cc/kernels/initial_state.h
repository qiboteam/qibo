#ifndef KERNEL_INITIAL_STATE_H_
#define KERNEL_INITIAL_STATE_H_

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct InitialStateFunctor {
  void operator()(const Device &d, T *in);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_TIME_TWO_H_