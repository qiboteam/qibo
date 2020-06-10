#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "apply_gate.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

using thread::ThreadPool;

template <typename T>
struct BaseOneQubitGateFunctor<CPUDevice, T> {
  virtual void apply(T& state1, T& state2, const T* gate = NULL) const {}

  void operator()(const OpKernelContext* context, const CPUDevice& d, T* state,
                  int nqubits, int target, int ncontrols, const int32* controls,
                  const int32* tensor_controls, const int32* qubits,
                  const T* gate = NULL) const {
    const int m = nqubits - target - 1;
    const int64 tk = (int64) 1 << m;
    int64 nstates = (int64) 1 << (nqubits - ncontrols - 1);
    int target_eff = target;
    for (int i = 0; i < ncontrols; i++) {
      if (controls[i] < target) {
        target_eff--;
      }
    }

    // Set multi-threading
    auto thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    const int ncores = (int) thread_pool->NumThreads();
    int64 nreps;
    if (ncores > 1) {
      nreps = (int64) nstates / ncores;
    }
    else {
      nreps = 1;
    }
    const ThreadPool::SchedulingParams p(
        ThreadPool::SchedulingStrategy::kFixedBlockSize, absl::nullopt,
        nreps);

    // Apply gate
    if (ncontrols == 0) {
      auto DoWork = [&](int64 t, int64 w) {
        for (auto g = t; g < w; g += 1) {
          int64 i = ((int64) ((int64) g >> m) << (m + 1)) + (g & (tk - 1));
          apply(state[i], state[i + tk], gate);
        }
      };
      thread_pool->ParallelFor(nstates, p, DoWork);
    }
    else if (ncontrols == 1) {
      const int cm = nqubits - controls[0] - 1;
      const int m1 = std::min(m, cm);
      const int m2 = std::max(m, cm);
      const int64 ck = (int64) 1 << cm;
      const int64 k1 = std::min(tk, ck);
      const int64 k2 = std::max(tk, ck);

      auto DoWork = [&](int64 t, int64 w) {
        for (auto g = t; g < w; g += 1) {
          int64 i = ((int64) ((int64) g >> m1) << (m1 + 1)) + (g & (k1 - 1)) + k1;
          i = ((int64) ((int64) i >> m2) << (m2 + 1)) + (i & (k2 - 1)) + k2;
          apply(state[i - tk], state[i], gate);
        }
      };
      thread_pool->ParallelFor(nstates, p, DoWork);
    }
    else {
      const int N = ncontrols + 1;
      auto DoWork = [&](int64 t, int64 w) {
        for (auto g = t; g < w; g += 1) {
          int64 i = g;
          for (auto iq = 0; iq < N; iq++) {
            const auto n = qubits[iq];
            int64 k = (int64) 1 << n;
            i = ((int64) ((int64) i >> n) << (n + 1)) + (i & (k - 1)) + k;
          }
          apply(state[i - tk], state[i], gate);
        }
      };
      thread_pool->ParallelFor(nstates, p, DoWork);
    }
  }
};

// Apply general one-qubit gate via gate matrix
template <typename T>
struct ApplyGateFunctor<CPUDevice, T> : BaseOneQubitGateFunctor<CPUDevice, T> {
  inline void apply(T& state1, T& state2, const T* gate = NULL) const override {
    const auto buffer = state1;
    state1 = gate[0] * state1 + gate[1] * state2;
    state2 = gate[2] * buffer + gate[3] * state2;
  }
};

// Apply X gate via swap
template <typename T>
struct ApplyXFunctor<CPUDevice, T> : BaseOneQubitGateFunctor<CPUDevice, T> {
  inline void apply(T& state1, T& state2, const T* gate = NULL) const override {
    std::swap(state1, state2);
  }
};

// Apply Y gate via swap
template <typename T>
struct ApplyYFunctor<CPUDevice, T> : BaseOneQubitGateFunctor<CPUDevice, T> {
  inline void apply(T& state1, T& state2, const T* gate = NULL) const override {
    state1 *= T(0, 1);
    state2 *= -T(0, 1);
    std::swap(state1, state2);
  }
};

// Apply Z gate
template <typename T>
struct ApplyZFunctor<CPUDevice, T> : BaseOneQubitGateFunctor<CPUDevice, T> {
  inline void apply(T& state1, T& state2, const T* gate = NULL) const override {
    state2 *= -1;
  }
};

// Apply ZPow gate
template <typename T>
struct ApplyZPowFunctor<CPUDevice, T> : BaseOneQubitGateFunctor<CPUDevice, T> {
  inline void apply(T& state1, T& state2, const T* gate = NULL) const override {
    state2 *= gate[0];
  }
};

template <typename T>
struct BaseTwoQubitGateFunctor<CPUDevice, T> {
  virtual void apply(T* state, int64 i, int64 tk1, int64 tk2,
                     const T* gate = NULL) const {}

  void operator()(const OpKernelContext* context, const CPUDevice& d, T* state,
                  int nqubits, int target1, int target2, int ncontrols,
                  const int32* controls, const int32* tensor_controls,
                  const int32* qubits, const T* gate = NULL) const {
    const int t1 = std::max(target1, target2);
    const int t2 = std::min(target1, target2);
    int m1 = nqubits - t1 - 1;
    int m2 = nqubits - t2 - 1;
    const int64 tk1 = (int64) 1 << m1;
    const int64 tk2 = (int64) 1 << m2;
    const int64 nstates = (int64) 1 << (nqubits - 2 - ncontrols);

    int64 targetk1 = tk1;
    int64 targetk2 = tk2;
    if (target1 > target2) {
      std::swap(targetk1, targetk2);
    }

    auto thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    const int ncores = (int) thread_pool->NumThreads();
    int64 nreps;
    if (ncores > 1) {
      nreps = (int64) nstates / ncores;
    }
    else {
      nreps = 1;
    }
    const ThreadPool::SchedulingParams p(
        ThreadPool::SchedulingStrategy::kFixedBlockSize, absl::nullopt,
        nreps);

    if (ncontrols == 0) {
      auto DoWork = [&](int64 t, int64 w) {
        for (auto g = t; g < w; g += 1) {
          int64 i = ((int64) ((int64) g >> m1) << (m1 + 1)) + (g & (tk1 - 1));
          i = ((int64) ((int64) i >> m2) << (m2 + 1)) + (i & (tk2 - 1));
          apply(state, i, targetk1, targetk2, gate);
        }
      };
      thread_pool->ParallelFor(nstates, p, DoWork);
    }
    else {
      const int N = ncontrols + 2;
      auto DoWork = [&](int64 t, int64 w) {
        for (auto g = t; g < w; g += 1) {
          int64 i = g;
          for (auto iq = 0; iq < N; iq++) {
            const auto m = qubits[iq];
            int64 k = (int64) 1 << m;
            i = ((int64) ((int64) i >> m) << (m + 1)) + (i & (k - 1)) + k;
          }
          apply(state, i - tk1 - tk2, targetk1, targetk2, gate);
        }
      };
      thread_pool->ParallelFor(nstates, p, DoWork);
    }
  }
};

// Apply general one-qubit gate via gate matrix
template <typename T>
struct ApplyTwoQubitGateFunctor<CPUDevice, T>
    : BaseTwoQubitGateFunctor<CPUDevice, T> {
  inline void apply(T* state, int64 i, int64 tk1, int64 tk2,
                    const T* gate = NULL) const {
    const int64 i1 = i + tk1;
    const int64 i2 = i + tk2;
    const int64 i3 = i1 + tk2;
    const auto buffer = state[i];
    state[i] = (gate[0] * state[i] + gate[1] * state[i1] + gate[2] * state[i2] +
                gate[3] * state[i3]);
    const auto buffer1 = state[i1];
    state[i1] = (gate[4] * buffer + gate[5] * state[i1] + gate[6] * state[i2] +
                 gate[7] * state[i3]);
    const auto buffer2 = state[i2];
    state[i2] = (gate[8] * buffer + gate[9] * buffer1 + gate[10] * state[i2] +
                 gate[11] * state[i3]);
    state[i3] = (gate[12] * buffer + gate[13] * buffer1 + gate[14] * buffer2 +
                 gate[15] * state[i3]);
  }
};

// Apply fSim gate from https://arxiv.org/abs/2001.08343
template <typename T>
struct ApplyFsimFunctor<CPUDevice, T> : BaseTwoQubitGateFunctor<CPUDevice, T> {
  inline void apply(T* state, int64 i, int64 tk1, int64 tk2,
                    const T* gate = NULL) const {
    const int64 i1 = i + tk1;
    const int64 i2 = i + tk2;
    const int64 i3 = i1 + tk2;
    const auto buffer = state[i1];
    state[i1] = gate[0] * state[i1] + gate[1] * state[i2];
    state[i2] = gate[2] * buffer + gate[3] * state[i2];
    state[i3] = gate[4] * state[i3];
  }
};

// Apply SWAP gate
template <typename T>
struct ApplySwapFunctor<CPUDevice, T> : BaseTwoQubitGateFunctor<CPUDevice, T> {
  inline void apply(T* state, int64 i, int64 tk1, int64 tk2,
                    const T* gate = NULL) const {
    std::swap(state[i + tk1], state[i + tk2]);
  }
};

template <typename Device, typename T, typename F, bool UseMatrix>
class OneQubitGateOp : public OpKernel {
 public:
  explicit OneQubitGateOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("controls", &controls_));
    OP_REQUIRES_OK(context, context->GetAttr("nqubits", &nqubits_));
    OP_REQUIRES_OK(context, context->GetAttr("target", &target_));
  }

  void Compute(OpKernelContext* context) override {
    // grabe the input tensor
    Tensor state = context->input(0);
    const int ncontrols = controls_.size();

    if (UseMatrix) {
      const Tensor& gate = context->input(1);
      const Tensor& tensor_controls = context->input(2);
      const Tensor& qubits = context->input(3);

      // call the implementation
      F()
      (context, context->eigen_device<Device>(), state.flat<T>().data(),
       nqubits_, target_, ncontrols, controls_.data(),
       tensor_controls.flat<int32>().data(), qubits.flat<int32>().data(),
       gate.flat<T>().data());
    } else {
      const Tensor& tensor_controls = context->input(1);
      const Tensor& qubits = context->input(2);

      // call the implementation
      F()
      (context, context->eigen_device<Device>(), state.flat<T>().data(),
       nqubits_, target_, ncontrols, controls_.data(),
       tensor_controls.flat<int32>().data(), qubits.flat<int32>().data());
    }
    context->set_output(0, state);
  }

 private:
  int nqubits_;
  int target_;
  std::vector<int> controls_;
};

template <typename Device, typename T, typename F, bool UseMatrix>
class TwoQubitGateOp : public OpKernel {
 public:
  explicit TwoQubitGateOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("controls", &controls_));
    OP_REQUIRES_OK(context, context->GetAttr("nqubits", &nqubits_));
    OP_REQUIRES_OK(context, context->GetAttr("target1", &target1_));
    OP_REQUIRES_OK(context, context->GetAttr("target2", &target2_));
  }

  void Compute(OpKernelContext* context) override {
    // grabe the input tensor
    Tensor state = context->input(0);
    const int ncontrols = controls_.size();

    if (UseMatrix) {
      const Tensor& gate = context->input(1);
      const Tensor& tensor_controls = context->input(2);
      const Tensor& qubits = context->input(3);

      // call the implementation
      F()
      (context, context->eigen_device<Device>(), state.flat<T>().data(),
       nqubits_, target1_, target2_, ncontrols, controls_.data(),
       tensor_controls.flat<int32>().data(), qubits.flat<int32>().data(), gate.flat<T>().data());
    } else {
      const Tensor& tensor_controls = context->input(1);
      const Tensor& qubits = context->input(2);

      // call the implementation
      F()
      (context, context->eigen_device<Device>(), state.flat<T>().data(),
       nqubits_, target1_, target2_, ncontrols, controls_.data(),
       tensor_controls.flat<int32>().data(), qubits.flat<int32>().data());
    }
    context->set_output(0, state);
  }

 private:
  int nqubits_;
  int target1_, target2_;
  std::vector<int> controls_;
};

// Register the CPU kernels.
#define REGISTER_CPU(T, NAME, OP, FUNCTOR, USEMATRIX)       \
  REGISTER_KERNEL_BUILDER(                                  \
      Name(NAME).Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      OP<CPUDevice, T, FUNCTOR<CPUDevice, T>, USEMATRIX>);

// Register one-qubit gate kernels.
#if GOOGLE_CUDA

// Register the GPU kernels.
#define REGISTER_GPU(T, NAME, OP, FUNCTOR, USEMATRIX)       \
  extern template struct FUNCTOR<GPUDevice, T>;             \
  REGISTER_KERNEL_BUILDER(                                  \
      Name(NAME).Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      OP<GPUDevice, T, FUNCTOR<GPUDevice, T>, USEMATRIX>);

#define REGISTER_ONEQUBIT(NAME, FUNCTOR, USEMATRIX)                   \
  REGISTER_CPU(complex64, NAME, OneQubitGateOp, FUNCTOR, USEMATRIX);  \
  REGISTER_CPU(complex128, NAME, OneQubitGateOp, FUNCTOR, USEMATRIX); \
  REGISTER_GPU(complex64, NAME, OneQubitGateOp, FUNCTOR, USEMATRIX);  \
  REGISTER_GPU(complex128, NAME, OneQubitGateOp, FUNCTOR, USEMATRIX);

// Register two-qubit gate kernels.
#define REGISTER_TWOQUBIT(NAME, FUNCTOR, USEMATRIX)                   \
  REGISTER_CPU(complex64, NAME, TwoQubitGateOp, FUNCTOR, USEMATRIX);  \
  REGISTER_CPU(complex128, NAME, TwoQubitGateOp, FUNCTOR, USEMATRIX); \
  REGISTER_GPU(complex64, NAME, TwoQubitGateOp, FUNCTOR, USEMATRIX);  \
  REGISTER_GPU(complex128, NAME, TwoQubitGateOp, FUNCTOR, USEMATRIX);

#else

#define REGISTER_ONEQUBIT(NAME, FUNCTOR, USEMATRIX)                  \
  REGISTER_CPU(complex64, NAME, OneQubitGateOp, FUNCTOR, USEMATRIX); \
  REGISTER_CPU(complex128, NAME, OneQubitGateOp, FUNCTOR, USEMATRIX);

// Register two-qubit gate kernels.
#define REGISTER_TWOQUBIT(NAME, FUNCTOR, USEMATRIX)                  \
  REGISTER_CPU(complex64, NAME, TwoQubitGateOp, FUNCTOR, USEMATRIX); \
  REGISTER_CPU(complex128, NAME, TwoQubitGateOp, FUNCTOR, USEMATRIX);

#endif

REGISTER_ONEQUBIT("ApplyGate", ApplyGateFunctor, true);
REGISTER_ONEQUBIT("ApplyZPow", ApplyZPowFunctor, true);
REGISTER_ONEQUBIT("ApplyX", ApplyXFunctor, false);
REGISTER_ONEQUBIT("ApplyY", ApplyYFunctor, false);
REGISTER_ONEQUBIT("ApplyZ", ApplyZFunctor, false);
REGISTER_TWOQUBIT("ApplyTwoQubitGate", ApplyTwoQubitGateFunctor, true);
REGISTER_TWOQUBIT("ApplyFsim", ApplyFsimFunctor, true);
REGISTER_TWOQUBIT("ApplySwap", ApplySwapFunctor, false);
}  // namespace functor
}  // namespace tensorflow
