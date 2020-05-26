#include "apply_gate.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

using thread::ThreadPool;


template <typename T>
struct BaseApplyGateFunctor<CPUDevice, T> {
  virtual inline void _apply(T& state1, T& state2, const T* gate = NULL) {}

  void _work(int64 t, int64 w, T* state, const T* gate, const int64 tk) {
    for (auto g = t; g < w; g += 2 * tk) {
      for (auto i = g; i < g + tk; i++) {
          _apply(state[i], state[i + tk], gate);
      }
    }
  }

  void _singlecontrol_work(int64 t, int64 w, T* state, const T* gate,
                           const int64 tk, const int64 tk_reduced,
                           const int64 ck, const int mask) {
    const int64 inv_mask = ck - 1;
    for (auto g = t; g < w; g += 2 * tk_reduced) {
      for (auto i = g; i < g + tk_reduced; i++) {
        const int64 i1 = ((i & mask) << 1) + (i & inv_mask) + ck;
        const int64 i2 = i1 + tk;
        _apply(state[i1], state[i2], gate);
      }
    }
  }

  void _multicontrol_work(int64 t, int64 w, T* state, const T* gate,
                          const int64 tk, const int64 tk_reduced,
                          const std::map<int64, int64> masks) {

    for (auto g = t; g < w; g += 2 * tk_reduced) {
      for (auto i = g; i < g + tk_reduced; i++) {
        int64 i1 = i;
        for (auto const& m : masks) {
          i1 = ((i1 & m.second) << 1) + (i1 & (m.first - 1)) + m.first;
        }
        const int64 i2 = i1 + tk;
        _apply(state[i1], state[i2], gate);
      }
    }
  }

  void operator()(const OpKernelContext* context, const CPUDevice& d, T* state,
                  int nqubits, int target, int ncontrols,
                  const int32* controls, const T* gate = NULL) {
    const int64 tk = (int64) 1 << (nqubits - target - 1);
    const int64 nstates = (int64) 1 << (nqubits - ncontrols);
    int target_eff = target;
    for (int i = 0; i < ncontrols; i++) {
      if (controls[i] < target) {
        target_eff--;
      }
    }
    const int64 tk_reduced = (int64) 1 << (nqubits - target_eff - ncontrols - 1);

    const ThreadPool::SchedulingParams p(
        ThreadPool::SchedulingStrategy::kFixedBlockSize, absl::nullopt,
        2 * tk_reduced);
    auto thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;

    if (ncontrols == 0) {
      auto DoWork = [&](int64 t, int64 w) {
        _work(t, w, state, gate, tk);
      };
      thread_pool->ParallelFor(nstates, p, DoWork);

    }
    else if (ncontrols == 1) {
        const int control = controls[0];
        const int64 ck = 1 << (nqubits - control - 1);
        const int64 mask = ((1 << control) - 1) << (nqubits - control - 1);
        auto DoWork = [&](int64 t, int64 w) {
          _singlecontrol_work(t, w, state, gate, tk, tk_reduced, ck, mask);
        };
        thread_pool->ParallelFor(nstates, p, DoWork);
    }
    else {
      std::map<int64, int64> masks;
      for (int i = 0; i < ncontrols; i++) {
        const int control = controls[i];
        const int64 ck = 1 << (nqubits - control - 1);
        const int64 mask = ((1 << control) - 1) << (nqubits - control - 1);
        masks.insert(std::pair<int64, int64>(ck, mask));
      }

      auto DoWork = [&](int64 t, int64 w) {
        _multicontrol_work(t, w, state, gate, tk, tk_reduced, masks);
      };
      thread_pool->ParallelFor(nstates, p, DoWork);
    }
  }
};


// Apply general one-qubit gate via gate matrix
template <typename T>
struct ApplyGateFunctor<CPUDevice, T>: BaseApplyGateFunctor<CPUDevice, T> {
  inline void _apply(T& state1, T& state2, const T* gate = NULL) override {
    const auto buffer = state1;
    state1 = gate[0] * state1 + gate[1] * state2;
    state2 = gate[2] * buffer + gate[3] * state2;
  }
};


// Apply X gate via swap
template <typename T>
struct ApplyXFunctor<CPUDevice, T>: BaseApplyGateFunctor<CPUDevice, T> {
  inline void _apply(T& state1, T& state2, const T* gate = NULL) override {
    std::swap(state1, state2);
  }
};


// Apply Y gate via swap
template <typename T>
struct ApplyYFunctor<CPUDevice, T>: BaseApplyGateFunctor<CPUDevice, T> {
  inline void _apply(T& state1, T& state2, const T* gate = NULL) override {
    state1 *= T(0, 1);
    state2 *= - T(0, 1);
    std::swap(state1, state2);
  }
};


// Apply Z gate
template <typename T>
struct ApplyZFunctor<CPUDevice, T>: BaseApplyGateFunctor<CPUDevice, T> {
  inline void _apply(T& state1, T& state2, const T* gate = NULL) override {
    state2 *= -1;
  }
};


// Apply ZPow gate
template <typename T>
struct ApplyZPowFunctor<CPUDevice, T>: BaseApplyGateFunctor<CPUDevice, T> {
  inline void _apply(T& state1, T& state2, const T* gate = NULL) override {
    state2 *= gate[0];
  }
};


template <typename T>
struct ApplySwapFunctor<CPUDevice, T> {
  void operator()(const OpKernelContext* context, const CPUDevice& d, T* state,
                  int nqubits, int target1, int target2, int ncontrols,
                  const int32* controls, const T* gate = NULL) {
    const int t1 = std::max(target1, target2);
    const int t2 = std::min(target1, target2);
    const int64 tk1 = 1 << (nqubits - t1 - 1);
    const int64 tk2 = 1 << (nqubits - t2 - 1);
    const int64 nstates = 1 << (nqubits - 2 - ncontrols);

    if (ncontrols == 0) {
      const int64 mask1 = ((1 << t1) - 1) << (nqubits - t1 - 1);
      const int64 mask2 = ((1 << t2) - 1) << (nqubits - t2 - 1);

      for (auto g = 0; g < nstates; g += 1) {
        int64 i = ((g & mask1) << 1) + (g & (tk1 - 1));
        i = ((i & mask2) << 1) + (i & (tk2 - 1));
        std::swap(state[i + tk1], state[i + tk2]);
      }
    }
    else {
      int t1_eff = t1;
      int t2_eff = t2;
      for (int i = 0; i < ncontrols; i++) {
        if (controls[i] < t1) {
          t1_eff--;
        }
        if (controls[i] < t2) {
          t2_eff--;
        }
      }

      int64 tk1_eff = tk1;
      if (ncontrols > 0 || t1 != t1_eff) {
        tk1_eff = 1 << (nqubits - ncontrols - t1_eff - 1);
      }
      int64 tk2_eff = tk2;
      if (ncontrols > 0 || t2 != t2_eff) {
        tk2_eff = 1 << (nqubits - ncontrols - t2_eff - 1);
      }

      std::map<int64, int64> control_masks;
      for (int i = 0; i < ncontrols; i++) {
        const int control = controls[i];
        const int64 ck = 1 << (nqubits - control - 1);
        const int64 mask = ((1 << control) - 1) << (nqubits - control - 1);
        control_masks.insert(std::pair<int64, int64>(ck, mask));
      }

      const int64 mask1 = ((1 << t1_eff) - 1) << (nqubits - t1_eff - 1);
      const int64 mask2 = ((1 << t2_eff) - 1) << (nqubits - t2_eff - 1);

      for (auto g = 0; g < nstates; g += 1) {
        int64 i = ((g & mask1) << 1) + (g & (tk1_eff - 1));
        i = ((i & mask2) << 1) + (i & (tk2_eff - 1));
        for (auto const& m : control_masks) {
          i = ((i & m.second) << 1) + (i & (m.first - 1)) + m.first;
        }
        std::swap(state[i + tk1], state[i + tk2]);
      }
    }
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
                                  nqubits_, target_, ncontrols,
                                  controls.flat<int32>().data(),
                                  gate.flat<T>().data());

    context->set_output(0, state);
  }

 private:
  int nqubits_;
  int target_;
};

// TODO: Inherit all these ops from a single base that defines compute
template <typename Device, typename T>
class ApplyXOp : public OpKernel {
 public:
  explicit ApplyXOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("nqubits", &nqubits_));
    OP_REQUIRES_OK(context, context->GetAttr("target", &target_));
  }

  void Compute(OpKernelContext* context) override {
    // grabe the input tensor
    Tensor state = context->input(0);
    const Tensor& controls = context->input(1);
    const int ncontrols = controls.flat<int32>().size();

    // prevent running on GPU
    OP_REQUIRES(
        context, (std::is_same<Device, CPUDevice>::value == true),
        errors::Unimplemented("ApplyX operator not implemented for GPU."));

    // call the implementation
    ApplyXFunctor<Device, T>()(context, context->eigen_device<Device>(),
                               state.flat<T>().data(),
                               nqubits_, target_, ncontrols,
                               controls.flat<int32>().data());

    context->set_output(0, state);
  }

 private:
  int nqubits_;
  int target_;
};


template <typename Device, typename T>
class ApplyYOp : public OpKernel {
 public:
  explicit ApplyYOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("nqubits", &nqubits_));
    OP_REQUIRES_OK(context, context->GetAttr("target", &target_));
  }

  void Compute(OpKernelContext* context) override {
    // grabe the input tensor
    Tensor state = context->input(0);
    const Tensor& controls = context->input(1);
    const int ncontrols = controls.flat<int32>().size();

    // prevent running on GPU
    OP_REQUIRES(
        context, (std::is_same<Device, CPUDevice>::value == true),
        errors::Unimplemented("ApplyX operator not implemented for GPU."));

    // call the implementation
    ApplyYFunctor<Device, T>()(context, context->eigen_device<Device>(),
                               state.flat<T>().data(),
                               nqubits_, target_, ncontrols,
                               controls.flat<int32>().data());

    context->set_output(0, state);
  }

 private:
  int nqubits_;
  int target_;
};


template <typename Device, typename T>
class ApplyZOp : public OpKernel {
 public:
  explicit ApplyZOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("nqubits", &nqubits_));
    OP_REQUIRES_OK(context, context->GetAttr("target", &target_));
  }

  void Compute(OpKernelContext* context) override {
    // grabe the input tensor
    Tensor state = context->input(0);
    const Tensor& controls = context->input(1);
    const int ncontrols = controls.flat<int32>().size();

    // prevent running on GPU
    OP_REQUIRES(
        context, (std::is_same<Device, CPUDevice>::value == true),
        errors::Unimplemented("ApplyX operator not implemented for GPU."));

    // call the implementation
    ApplyZFunctor<Device, T>()(context, context->eigen_device<Device>(),
                               state.flat<T>().data(),
                               nqubits_, target_, ncontrols,
                               controls.flat<int32>().data());

    context->set_output(0, state);
  }

 private:
  int nqubits_;
  int target_;
};


template <typename Device, typename T>
class ApplyZPowOp : public OpKernel {
 public:
  explicit ApplyZPowOp(OpKernelConstruction* context) : OpKernel(context) {
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
        errors::Unimplemented("ApplyX operator not implemented for GPU."));

    // call the implementation
    ApplyZPowFunctor<Device, T>()(context, context->eigen_device<Device>(),
                                  state.flat<T>().data(),
                                  nqubits_, target_, ncontrols,
                                  controls.flat<int32>().data(),
                                  gate.flat<T>().data());

    context->set_output(0, state);
  }

 private:
  int nqubits_;
  int target_;
};


template <typename Device, typename T>
class ApplySwapOp : public OpKernel {
 public:
  explicit ApplySwapOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("nqubits", &nqubits_));
    OP_REQUIRES_OK(context, context->GetAttr("target1", &target1_));
    OP_REQUIRES_OK(context, context->GetAttr("target2", &target2_));
  }

  void Compute(OpKernelContext* context) override {
    // grabe the input tensor
    Tensor state = context->input(0);
    const Tensor& controls = context->input(1);
    const int ncontrols = controls.flat<int32>().size();

    // prevent running on GPU
    OP_REQUIRES(
        context, (std::is_same<Device, CPUDevice>::value == true),
        errors::Unimplemented("ApplyX operator not implemented for GPU."));

    // call the implementation
    ApplySwapFunctor<Device, T>()(context, context->eigen_device<Device>(),
                                  state.flat<T>().data(),
                                  nqubits_, target1_, target2_,
                                  ncontrols, controls.flat<int32>().data());

    context->set_output(0, state);
  }

 private:
  int nqubits_;
  int target1_, target2_;
};


// Register the CPU kernels.
#define REGISTER_CPU(T, NAME, OP)                             \
  REGISTER_KERNEL_BUILDER(                                    \
      Name(NAME).Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
      OP<CPUDevice, T>);

REGISTER_CPU(complex64, "ApplyGate", ApplyGateOp);
REGISTER_CPU(complex128, "ApplyGate", ApplyGateOp);
REGISTER_CPU(complex64, "ApplyX", ApplyXOp);
REGISTER_CPU(complex128, "ApplyX", ApplyXOp);
REGISTER_CPU(complex64, "ApplyY", ApplyYOp);
REGISTER_CPU(complex128, "ApplyY", ApplyYOp);
REGISTER_CPU(complex64, "ApplyZ", ApplyZOp);
REGISTER_CPU(complex128, "ApplyZ", ApplyZOp);
REGISTER_CPU(complex64, "ApplyZPow", ApplyZPowOp);
REGISTER_CPU(complex128, "ApplyZPow", ApplyZPowOp);
REGISTER_CPU(complex64, "ApplySwap", ApplySwapOp);
REGISTER_CPU(complex128, "ApplySwap", ApplySwapOp);


// Register the GPU kernels.
#define REGISTER_GPU(T, NAME, OP, FUNCTOR)                      \
  extern template struct FUNCTOR<GPUDevice, T>;                 \
  REGISTER_KERNEL_BUILDER(                                      \
      Name(NAME).Device(DEVICE_GPU).TypeConstraint<T>("T"),     \
      OP<GPUDevice, T>);
REGISTER_GPU(complex64, "ApplyGate", ApplyGateOp, ApplyGateFunctor);
REGISTER_GPU(complex128, "ApplyGate", ApplyGateOp, ApplyGateFunctor);
REGISTER_GPU(complex64, "ApplyX", ApplyXOp, ApplyXFunctor);
REGISTER_GPU(complex128, "ApplyX", ApplyXOp, ApplyXFunctor);
REGISTER_GPU(complex64, "ApplyY", ApplyYOp, ApplyYFunctor);
REGISTER_GPU(complex128, "ApplyY", ApplyYOp, ApplyYFunctor);
REGISTER_GPU(complex64, "ApplyZ", ApplyZOp, ApplyZFunctor);
REGISTER_GPU(complex128, "ApplyZ", ApplyZOp, ApplyZFunctor);
REGISTER_GPU(complex64, "ApplyZPow", ApplyZPowOp, ApplyZPowFunctor);
REGISTER_GPU(complex128, "ApplyZPow", ApplyZPowOp, ApplyZPowFunctor);
REGISTER_GPU(complex64, "ApplySwap", ApplySwapOp, ApplySwapFunctor);
REGISTER_GPU(complex128, "ApplySwap", ApplySwapOp, ApplySwapFunctor);
}  // namespace functor
}  // namespace tensorflow
