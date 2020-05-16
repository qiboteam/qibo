#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "initial_state.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <type_traits>

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
    Tensor input_tensor = context->input(0);

    // prevent running on GPU
    OP_REQUIRES(context, (std::is_same<Device,CPUDevice>::value == true),
                errors::Unimplemented("InitialState operator not implemented for GPU."));

    // call the implementation
    InitialStateFunctor<Device, T>()(
      context->eigen_device<Device>(),
      input_tensor.flat<T>().data());
  }
};

#define REGISTER_CPU(T)                                               \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("InitialState").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      InitialStateOp<CPUDevice, T>);
REGISTER_CPU(complex64);
REGISTER_CPU(complex128);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)
  extern template struct InitialStateOp<GPUDevice, T>;
  REGISTER_KERNEL_BUILDER(
      Name("InitialState").Device(DEVICE_GPU).TypeConstraint<T>("T"),
      InitialState<GPUDevice, T>);
REGISTER_GPU(complex64);
REGISTER_GPU(complex128);
#endif  // GOOGLE_CUDA
} // namespace functor
} // namespace tensorflow