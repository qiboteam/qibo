#ifndef KERNEL_APPLY_GATE_H_
#define KERNEL_APPLY_GATE_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct BaseApplyGateFunctor {
  virtual void _apply(T& state1, T& state2, const T* gate = NULL) const;

  void operator()(const OpKernelContext* context, const Device& d, T* state,
                  int nqubits, int target, int ncontrols,
                  const int32* controls, const T* gate = NULL) const;
};

template <typename Device, typename T>
struct ApplyGateFunctor: BaseApplyGateFunctor<Device, T> {};

template <typename Device, typename T>
struct ApplyXFunctor: BaseApplyGateFunctor<Device, T> {};

template <typename Device, typename T>
struct ApplyYFunctor: BaseApplyGateFunctor<Device, T> {};

template <typename Device, typename T>
struct ApplyZFunctor: BaseApplyGateFunctor<Device, T> {};

template <typename Device, typename T>
struct ApplyZPowFunctor: BaseApplyGateFunctor<Device, T> {};

template <typename Device, typename T>
struct ApplySwapFunctor {
  void operator()(const OpKernelContext* context, const Device& d, T* state,
                  int nqubits, int target1, int target2, int ncontrols,
                  const int32* controls, const T* gate = NULL);
};

}  // namespace functor

}  // namespace tensorflow

#endif  // KERNEL_APPLY_GATE_H_
