#include <algorithm>
#include <iterator>
#include "apply_gate.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// CPU specialization
template <typename T>
struct ApplyGateFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, T* state, const T* gate, int nqubits,
                  int target, const int32* control, int ncontrol) {
    const int64 nstates = std::pow(2, nqubits);
    const int64 tk = std::pow(2, nqubits - target - 1);

    std::set<int64> cks;
    int64 cktot = 0;
    for (int i = 0; i < ncontrol; i++) {
      int64 ck_temp = std::pow(2, nqubits - control[i] - 1);
      cks.insert(ck_temp);
      cktot += ck_temp;
    }

    for (auto g = 0; g < nstates; g += 2 * tk) {
      auto i = g;
      while (i < g + tk) {
        if (cks.find(i) != cks.end()) {
          cks.erase(i);
          i += 2 * i;
        }
        else {
          const auto i1 = i + cktot;
          const auto i2 = i1 + tk;
          const auto buffer = state[i1];
          state[i1] = gate[0] * state[i1] + gate[1] * state[i2];
          state[i2] = gate[2] * buffer + gate[3] * state[i2];
          i++;
        }
      }
    }
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
    const Tensor& control = context->input(4);
    const int nqubits = context->input(2).flat<int32>()(0);
    const int target = context->input(3).flat<int32>()(0);
    const int ncontrol = control.flat<int32>().size();

    // prevent running on GPU
    OP_REQUIRES(
        context, (std::is_same<Device, CPUDevice>::value == true),
        errors::Unimplemented("ApplyGate operator not implemented for GPU."));

    // call the implementation
    ApplyGateFunctor<Device, T>()(context->eigen_device<Device>(),
                                  state.flat<T>().data(),
                                  gate.flat<T>().data(),
                                  nqubits, target,
                                  control.flat<int32>().data(),
                                  ncontrol);

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
