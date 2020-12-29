#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "initial_state.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// CPU specialization
template <typename T>
struct InitialStateFunctor<CPUDevice, T> {
  void operator()(const CPUDevice &d, T *out, int64 size)
  {
    #pragma omp parallel for
    for (size_t i = 1; i < (size_t) size; i++)
      out[i] = T(0, 0);
    out[0] = T(1, 0);
  }
};

template <typename Device, typename T>
class InitialStateOp : public OpKernel {
 public:
  explicit InitialStateOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("nqubits", &nqubits_));
    OP_REQUIRES_OK(context, context->GetAttr("is_matrix", &is_matrix_));
    OP_REQUIRES_OK(context, context->GetAttr("omp_num_threads", &threads_));
    OP_REQUIRES(context, nqubits_ > 0, errors::InvalidArgument("nqubits must be positive"));
    omp_set_num_threads(threads_);
  }

  void Compute(OpKernelContext *context) override {
    // grabe the input tensor
    const int64 size = pow(2, nqubits_);

    TensorShape shape{size};
    if (is_matrix_)
      shape = TensorShape{size, size};

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output_tensor));

    // call the implementation
    InitialStateFunctor<Device, T>()(context->eigen_device<Device>(),
                                     output_tensor->flat<T>().data(),
                                     output_tensor->flat<T>().size());
  }

 private:
  int nqubits_;
  bool is_matrix_;
  int threads_;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                               \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("InitialState").Device(DEVICE_CPU).TypeConstraint<T>("dtype"), \
      InitialStateOp<CPUDevice, T>);
REGISTER_CPU(complex64);
REGISTER_CPU(complex128);

#ifdef GOOGLE_CUDA
// Register the GPU kernels.
#define REGISTER_GPU(T)                                               \
  extern template struct InitialStateFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("InitialState").Device(DEVICE_GPU).TypeConstraint<T>("dtype"), \
      InitialStateOp<GPUDevice, T>);
REGISTER_GPU(complex64);
REGISTER_GPU(complex128);
#endif
}  // namespace functor
}  // namespace tensorflow
