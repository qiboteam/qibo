#include "apply_gate.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

using thread::ThreadPool;

// CPU specialization
template <typename T>
struct ApplyGateFunctor<CPUDevice, T> {
  void operator()(const OpKernelContext* context, const CPUDevice& d, T* state,
                  const T* gate, int nqubits, int target) {
    const int64 nstates = std::pow(2, nqubits);
    const int64 k = std::pow(2, nqubits - target - 1);

    auto DoWork = [&](int64 g, int64 w) {
      for (auto i = g; i < g + k; i++) {
        const auto buffer = state[i];
        state[i] = gate[0] * state[i] + gate[1] * state[i + k];
        state[i + k] = gate[2] * buffer + gate[3] * state[i + k];
      }
    };

    const ThreadPool::SchedulingParams p(
        ThreadPool::SchedulingStrategy::kFixedBlockSize, {}, 2 * k);
    auto thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    thread_pool->ParallelFor(nstates, p, DoWork);
  }
};

template <typename Device, typename T>
class ApplyGateOp : public OpKernel {
 public:
  explicit ApplyGateOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // grabe the input tensor
    Tensor state = context->input(0);
    const Tensor& gate = context->input(1);
    const int nqubits = context->input(2).flat<int32>()(0);
    const int target = context->input(3).flat<int32>()(0);

    // prevent running on GPU
    OP_REQUIRES(
        context, (std::is_same<Device, CPUDevice>::value == true),
        errors::Unimplemented("ApplyGate operator not implemented for GPU."));

    // call the implementation
    ApplyGateFunctor<Device, T>()(context, context->eigen_device<Device>(),
                                  state.flat<T>().data(), gate.flat<T>().data(),
                                  nqubits, target);

    context->set_output(0, state);
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                            \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("ApplyGate").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ApplyGateOp<CPUDevice, T>);
REGISTER_CPU(complex64);
REGISTER_CPU(complex128);

// Register the GPU kernels.
#define REGISTER_GPU(T)                                            \
  extern template struct ApplyGateFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("ApplyGate").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ApplyGateOp<GPUDevice, T>);
REGISTER_GPU(complex64);
REGISTER_GPU(complex128);
}  // namespace functor
}  // namespace tensorflow
