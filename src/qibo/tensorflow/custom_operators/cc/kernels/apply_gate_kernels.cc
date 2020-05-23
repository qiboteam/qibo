#include "apply_gate.h"
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
                  const T* gate, int nqubits, int target,
                  const int32* controls, int ncontrols) {
    const int64 nstates = std::pow(2, nqubits);
    const int64 tk = std::pow(2, nqubits - target - 1);

    int64 cktot = 0;
    std::vector<int64> cks(ncontrols);
    for (int i = 0; i < ncontrols; i++) {
      cks[i] = std::pow(2, nqubits - controls[i] - 1);
      cktot += cks[i];
    }

    auto DoWork = [&](int64 t, int64 w) {
      for (auto g = t; g < w; g += 2 * tk) {
        for (auto i = g; i < g + tk; i++) {
          bool apply = true;
          for (const auto &q: cks) {
            if (((int64) i / q) % 2) {
              apply = false;
              break;
            }
          }

          if (apply) {
            const int64 i1 = i + cktot;
            const int64 i2 = i1 + tk;
            const auto buffer = state[i1];
            state[i1] = gate[0] * state[i1] + gate[1] * state[i2];
            state[i2] = gate[2] * buffer + gate[3] * state[i2];
          }
        }
      }
    };

    const ThreadPool::SchedulingParams p(
        ThreadPool::SchedulingStrategy::kFixedBlockSize, absl::nullopt, 2 * tk);
    auto thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    thread_pool->ParallelFor(nstates, p, DoWork);
  }
};

template <typename Device, typename T>
class ApplyGateOp : public OpKernel {
 public:
  explicit ApplyGateOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("nqubits", &nqubits_));
    OP_REQUIRES_OK(context, context->GetAttr("target", &target_));
  }

  void Compute(OpKernelContext* context) override {
    // grabe the input tensor
    Tensor state = context->input(0);
    const Tensor& gate = context->input(1);
    const Tensor& controls = context->input(2);
    const int ncontrols = controls.flat<int32>().size();

    // prevent running on GPU
    OP_REQUIRES(
        context, (std::is_same<Device, CPUDevice>::value == true),
        errors::Unimplemented("ApplyGate operator not implemented for GPU."));

    // call the implementation
    ApplyGateFunctor<Device, T>()(context, context->eigen_device<Device>(),
                                  state.flat<T>().data(),
                                  gate.flat<T>().data(),
                                  nqubits_, target_,
                                  controls.flat<int32>().data(),
                                  ncontrols);

    context->set_output(0, state);
  }

 private:
  int nqubits_;
  int target_;
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
