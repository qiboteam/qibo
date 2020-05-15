#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

template<class T>
class InitialStateOp : public OpKernel {
 public:
  explicit InitialStateOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    Tensor input_tensor = context->input(0);
    input_tensor.flat<T>()(0) = T(1,0);
  }
};

#define REGISTER_CPU(T) REGISTER_KERNEL_BUILDER(Name("InitialState").Device(DEVICE_CPU).TypeConstraint<T>("T"), InitialStateOp<T>);
REGISTER_CPU(complex64);
REGISTER_CPU(complex128);
