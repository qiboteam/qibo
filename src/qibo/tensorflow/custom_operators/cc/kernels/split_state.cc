#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "split_state.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// CPU specialization
template <typename T>
struct SplitStateFunctor<CPUDevice, T> {
  void operator()(const CPUDevice &d, const T* state, T* pieces, int nqubits,
                  const int* global_qubits, int nglobal) {
    const int64 nstates = (int64) 1 << (nqubits - nglobal);
    const int ndevices = 1 << nglobal;
    std::vector<int64> ids(nglobal);

    std::cout << "nqubits = " << nqubits << std::endl;
    std::cout << "nglobal = " << nglobal << std::endl;
    std::cout << "nstates = " << nstates << std::endl;
    std::cout << "ndevices = " << ndevices << std::endl;

    for (auto g = 0; g < nstates; g++) {
      int64 i = g;
      for (auto j = 0; j < nglobal; j++) {
        const int n = global_qubits[j];
        ids[j] = (int64) 1 << n;
        i = ((int64) ((int64) i >> n) << (n + 1)) + (i & (ids[j] - 1));
      }
      for (auto d = 0; d < ndevices; d++) {
        int k = 0;
        for (auto j = 0; j < nglobal; j++) {
          if ((d >> j) % 2) {
            k += ids[j];
          }
        }
        pieces[nstates * d + g] = state[i + k];
      }
    }
  };
};

template <typename Device, typename T>
class SplitStateOp : public OpKernel {
 public:
  explicit SplitStateOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("nqubits", &nqubits_));
    OP_REQUIRES_OK(context, context->GetAttr("global_qubits", &global_qubits_));
  }

  void Compute(OpKernelContext *context) override {
    // grabe the input tensor
    const Tensor& state = context->input(0);
    Tensor pieces = context->input(1);
    const Tensor& qubits = context->input(1);
    const int nglobal = global_qubits_.size();

    // call the implementation
    SplitStateFunctor<Device, T>()(context->eigen_device<Device>(),
                                   state.flat<T>().data(),
                                   pieces.flat<T>().data(),
                                   nqubits_,
                                   global_qubits_.data(), nglobal);

    context->set_output(0, pieces);
  }
  private:
   int nqubits_;
   std::vector<int> global_qubits_;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                             \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("SplitState").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SplitStateOp<CPUDevice, T>);
REGISTER_CPU(complex64);
REGISTER_CPU(complex128);

#ifdef GOOGLE_CUDA
// Register the GPU kernels.
#define REGISTER_GPU(T)                                             \
  extern template struct SplitStateFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("SplitState").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      SplitStateOp<GPUDevice, T>);
REGISTER_GPU(complex64);
REGISTER_GPU(complex128);
#endif
}  // namespace functor
}  // namespace tensorflow
