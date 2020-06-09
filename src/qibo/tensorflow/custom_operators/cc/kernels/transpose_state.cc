#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "transpose_state.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
struct TransposeStateFunctor<CPUDevice, T> {
  void operator()(const CPUDevice &d, const T* state, T* transposed_state,
                  int nqubits, const int* qubit_order) {
    std::vector<int64> qubit_exponents(nqubits);
    for (int q = 0; q < nqubits; q++) {
      qubit_exponents[q] = (int64) 1 << (nqubits - qubit_order[q] - 1);
    }

    const int64 nstates = (int64) 1 << nqubits;
    for (auto g = 0; g < nstates; g++) {
      int64 k = 0;
      for (int q = 0; q < nqubits; q++) {
        if ((g >> (nqubits - q - 1)) % 2) k += qubit_exponents[q];
      }
      transposed_state[g] = state[k];
    }
  };
};


template <typename Device, typename T>
class TransposeStateOp : public OpKernel {
 public:
  explicit TransposeStateOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("nqubits", &nqubits_));
    OP_REQUIRES_OK(context, context->GetAttr("qubit_order", &qubit_order_));
  }

  void Compute(OpKernelContext *context) override {
    // grabe the input tensor
    const Tensor& state = context->input(0);
    Tensor transposed_state = context->input(1);

    // call the implementation
    TransposeStateFunctor<Device, T>()(context->eigen_device<Device>(),
                                       state.flat<T>().data(),
                                       transposed_state.flat<T>().data(),
                                       nqubits_, qubit_order_.data());

    context->set_output(0, transposed_state);
  }
  private:
   int nqubits_;
   std::vector<int> qubit_order_;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                             \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("TransposeState").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      TransposeStateOp<CPUDevice, T>);
REGISTER_CPU(complex64);
REGISTER_CPU(complex128);

#ifdef GOOGLE_CUDA
// Register the GPU kernels.
#define REGISTER_GPU(T)                                             \
  extern template struct TransposeStateFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("TransposeState").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      TransposeStateOp<GPUDevice, T>);
REGISTER_GPU(complex64);
REGISTER_GPU(complex128);
#endif
}  // namespace functor
}  // namespace tensorflow
