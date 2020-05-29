#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif // GOOGLE_CUDA

#include "initial_state.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// CPU specialization
template<typename T>
struct InitialStateFunctor<CPUDevice, T> {
  void operator()(const CPUDevice & d, T* inout) {
    inout[0] = T(1, 0);
  }
};

template <typename Device, typename T>
class InitialStateOp : public OpKernel {
 public:
  explicit InitialStateOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    // grabe the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    // fill output
    const auto N = input_tensor.flat<T>().size();
    for (auto i = 0; i < N; i++)
      output_tensor->flat<T>().data()[i] = input_tensor.flat<T>().data()[i];

    // call the implementation
    InitialStateFunctor<Device, T>()(
      context->eigen_device<Device>(),
      output_tensor->flat<T>().data());
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                               \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("InitialState").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      InitialStateOp<CPUDevice, T>);
REGISTER_CPU(complex64);
REGISTER_CPU(complex128);

#ifdef GOOGLE_CUDA
// Register the GPU kernels.
#define REGISTER_GPU(T)					                                      \
  extern template struct InitialStateFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(					                                  \
       Name("InitialState").Device(DEVICE_GPU).TypeConstraint<T>("T"),\
      InitialStateOp<GPUDevice, T>);
REGISTER_GPU(complex64);
REGISTER_GPU(complex128);
#endif
} // namespace functor
} // namespace tensorflow
