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

  void work(int64 t, int64 w, T* state, const T* gate, int64 tk) const {
    for (auto g = t; g < w; g += 2 * tk) {
      for (auto i = g; i < g + tk; i++) {
          apply(state[i], state[i + tk], gate);
      }
    }
  }

  void singlecontrol_work(int64 t, int64 w, T* state, const T* gate,
                          int64 tk, int64 tk_reduced,
                          int64 ck, int mask) const {
    const int64 inv_mask = ck - 1;
    for (auto g = t; g < w; g += 2 * tk_reduced) {
      for (auto i = g; i < g + tk_reduced; i++) {
        const int64 i1 = ((i & mask) << 1) + (i & inv_mask) + ck;
        const int64 i2 = i1 + tk;
        apply(state[i1], state[i2], gate);
      }
    }
  }

  void multicontrol_work(int64 t, int64 w, T* state, const T* gate,
                         int64 tk, int64 tk_reduced,
                         const std::map<int64, int64>& masks) const {

    for (auto g = t; g < w; g += 2 * tk_reduced) {
      for (auto i = g; i < g + tk_reduced; i++) {
        int64 i1 = i;
        for (auto const& m : masks) {
          i1 = ((i1 & m.second) << 1) + (i1 & (m.first - 1)) + m.first;
        }
        const int64 i2 = i1 + tk;
        apply(state[i1], state[i2], gate);
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

    auto thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    const int ncores = (int) thread_pool->NumThreads() / 2;
    int64 nreps;
    if (ncores > 1) {
      nreps = (int64) nstates / ncores;
    }
    else {
      nreps = nstates;
    }
    if (nreps % (2 * tk_reduced)) {
      nreps = 2 * tk_reduced;
    }
    const ThreadPool::SchedulingParams p(
        ThreadPool::SchedulingStrategy::kFixedBlockSize, absl::nullopt,
        nreps);

    if (ncontrols == 0) {
      auto DoWork = [&](int64 t, int64 w) {
        work(t, w, state, gate, tk);
      };
      thread_pool->ParallelFor(nstates, p, DoWork);
    }
    else if (ncontrols == 1) {
        const int control = controls[0];
        const int64 ck = (int64) 1 << (nqubits - control - 1);
        const int64 mask = (int64) (((int64) 1 << control) - 1) << (nqubits - control - 1);
        auto DoWork = [&](int64 t, int64 w) {
          singlecontrol_work(t, w, state, gate, tk, tk_reduced, ck, mask);
        };
        thread_pool->ParallelFor(nstates, p, DoWork);
    }
    else {
      std::map<int64, int64> masks;
      for (int i = 0; i < ncontrols; i++) {
        const int control = controls[i];
        const int64 ck = (int64) 1 << (nqubits - control - 1);
        const int64 mask = (int64) (((int64) 1 << control) - 1) << (nqubits - control - 1);
        masks.emplace(ck, mask);
      }

      auto DoWork = [&](int64 t, int64 w) {
        multicontrol_work(t, w, state, gate, tk, tk_reduced, masks);
      };
      thread_pool->ParallelFor(nstates, p, DoWork);
    }
  }
};


// Apply general one-qubit gate via gate matrix
template <typename T>
struct ApplyGateFunctor<CPUDevice, T>: BaseOneQubitGateFunctor<CPUDevice, T> {
  inline void apply(T& state1, T& state2, const T* gate = NULL) const override {
    const auto buffer = state1;
    state1 = gate[0] * state1 + gate[1] * state2;
    state2 = gate[2] * buffer + gate[3] * state2;
  }
};


// Apply X gate via swap
template <typename T>
struct ApplyXFunctor<CPUDevice, T>: BaseOneQubitGateFunctor<CPUDevice, T> {
  inline void apply(T& state1, T& state2, const T* gate = NULL) const override {
    std::swap(state1, state2);
  }
};


// Apply Y gate via swap
template <typename T>
struct ApplyYFunctor<CPUDevice, T>: BaseOneQubitGateFunctor<CPUDevice, T> {
  inline void apply(T& state1, T& state2, const T* gate = NULL) const override {
    state1 *= T(0, 1);
    state2 *= - T(0, 1);
    std::swap(state1, state2);
  }
};


// Apply Z gate
template <typename T>
struct ApplyZFunctor<CPUDevice, T>: BaseOneQubitGateFunctor<CPUDevice, T> {
  inline void apply(T& state1, T& state2, const T* gate = NULL) const override {
    state2 *= -1;
  }
};


// Apply ZPow gate
template <typename T>
struct ApplyZPowFunctor<CPUDevice, T>: BaseOneQubitGateFunctor<CPUDevice, T> {
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
                  const int32* controls, const T* gate = NULL) {
    const int t1 = std::max(target1, target2);
    const int t2 = std::min(target1, target2);
    const int64 tk1 = (int64) 1 << (nqubits - t1 - 1);
    const int64 tk2 = (int64) 1 << (nqubits - t2 - 1);
    const int64 nstates = (int64) 1 << (nqubits - 2 - ncontrols);

    auto thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    const int ncores = (int) thread_pool->NumThreads() / 2;
    int64 nreps;
    if (ncores > 1) {
      nreps = (int64) nstates / ncores;
    }
    else {
      nreps = nstates;
    }
    nreps = 0;
    const ThreadPool::SchedulingParams p(
        ThreadPool::SchedulingStrategy::kFixedBlockSize, absl::nullopt,
        nreps);

    if (ncontrols == 0) {
      const int64 mask1 = (int64) (((int64) 1 << t1) - 1) << (nqubits - t1 - 1);
      const int64 mask2 = (int64) (((int64) 1 << t2) - 1) << (nqubits - t2 - 1);

      auto DoWork = [&](int64 t, int64 w) {
        for (auto g = t; g < w; g += 1) {
          int64 i = ((g & mask1) << 1) + (g & (tk1 - 1));
          i = ((i & mask2) << 1) + (i & (tk2 - 1));
          apply(state, i, tk1, tk2, gate);
        }
      };
      thread_pool->ParallelFor(nstates, p, DoWork);
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
        tk1_eff = (int64) 1 << (nqubits - ncontrols - t1_eff - 1);
      }
      int64 tk2_eff = tk2;
      if (ncontrols > 0 || t2 != t2_eff) {
        tk2_eff = (int64) 1 << (nqubits - ncontrols - t2_eff - 1);
      }

      std::map<int64, int64> control_masks;
      for (int i = 0; i < ncontrols; i++) {
        const int control = controls[i];
        const int64 ck = (int64) 1 << (nqubits - control - 1);
        const int64 mask = (int64) (((int64) 1 << control) - 1) << (nqubits - control - 1);
        control_masks.emplace(ck, mask);
      }

      const int64 mask1 = ((1 << t1_eff) - 1) << (nqubits - t1_eff - 1);
      const int64 mask2 = ((1 << t2_eff) - 1) << (nqubits - t2_eff - 1);

      auto DoWork = [&](int64 t, int64 w) {
        for (auto g = t; g < w; g += 1) {
          int64 i = ((g & mask1) << 1) + (g & (tk1_eff - 1));
          i = ((i & mask2) << 1) + (i & (tk2_eff - 1));
          for (auto const& m : control_masks) {
            i = ((i & m.second) << 1) + (i & (m.first - 1)) + m.first;
          }
          apply(state, i, tk1, tk2, gate);
        }
      };
      //thread_pool->ParallelFor(nstates, p, DoWork);
      for (auto g = 0; g < nstates; g += 1) {
        std::cout << "g = " << g << std::endl;

        int64 i = ((g & mask1) << 1) + (g & (tk1_eff - 1));
        i = ((i & mask2) << 1) + (i & (tk2_eff - 1));
        for (auto const& m : control_masks) {
          i = ((i & m.second) << 1) + (i & (m.first - 1)) + m.first;
        }
        apply(state, i, tk1, tk2, gate);
      }
    }
  }
};


// Apply general one-qubit gate via gate matrix
template <typename T>
struct ApplyTwoQubitGateFunctor<CPUDevice, T>: BaseTwoQubitGateFunctor<CPUDevice, T> {
  inline void apply(T* state, int64 i, int64 tk1, int64 tk2,
                    const T* gate = NULL) const {
    const int64 i1 = i + tk1;
    const int64 i2 = i + tk2;
    const int64 i3 = i1 + tk2;
    const auto buffer = state[i];
    state[i] = (gate[0] * state[i] + gate[1] * state[i1] +
                gate[2] * state[i2] + gate[3] * state[i3]);
    const auto buffer1 = state[i1];
    state[i1] = (gate[4] * buffer + gate[5] * state[i1] +
                 gate[6] * state[i2] + gate[7] * state[i3]);
    const auto buffer2 = state[i2];
    state[i2] = (gate[8] * buffer + gate[9] * buffer1 +
                 gate[10] * state[i2] + gate[11] * state[i3]);
    state[i3] = (gate[12] * buffer + gate[13] * buffer1 +
                 gate[14] * buffer2 + gate[15] * state[i3]);
  }
};


// Apply fSim gate from https://arxiv.org/abs/2001.08343
template <typename T>
struct ApplyFsimFunctor<CPUDevice, T>: BaseTwoQubitGateFunctor<CPUDevice, T> {
  inline void apply(T* state, int64 i, int64 tk1, int64 tk2,
                    const T* gate = NULL) const {
    const int64 i1 = i + tk1;
    const int64 i2 = i + tk2;
    const int64 i3 = i1 + tk2;
    const auto buffer = state[i1];

    /*
    std::cout << std::endl;
    std::cout << "Gate0 " << gate[0] << std::endl;
    std::cout << "Gate1 " << gate[1] << std::endl;
    std::cout << "Gate2 " << gate[2] << std::endl;
    std::cout << "Gate3 " << gate[3] << std::endl;
    std::cout << "Gate4 " << gate[4] << std::endl;
    std::cout << std::endl;*/
    std::cout << std::endl;
    std::cout << "i " << i << std::endl;
    std::cout << "i1 " << i1 << std::endl;
    std::cout << "i2 " << i2 << std::endl;
    std::cout << "i3 " << i3 << std::endl;
    std::cout << std::endl;

    state[i1] = gate[0] * state[i1] + gate[1] * state[i2];
    state[i2] = gate[2] * buffer + gate[3] * state[i2];
    state[i3] = gate[4] * state[i3];
  }
};


// Apply SWAP gate
template <typename T>
struct ApplySwapFunctor<CPUDevice, T>: BaseTwoQubitGateFunctor<CPUDevice, T> {
  inline void apply(T* state, int64 i, int64 tk1, int64 tk2,
                    const T* gate = NULL) const {
    std::cout << std::endl;
    std::cout << "i " << i << std::endl;
    std::cout << "i1 " << i + tk1 << std::endl;
    std::cout << "i2 " << i + tk2 << std::endl;
    std::cout << std::endl;

    std::swap(state[i + tk1], state[i + tk2]);
  }
};


template <typename Device, typename T, typename F, bool UseMatrix>
class OneQubitGateOp : public OpKernel {
 public:
  explicit OneQubitGateOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("nqubits", &nqubits_));
    OP_REQUIRES_OK(context, context->GetAttr("target", &target_));
  }

  void Compute(OpKernelContext* context) override {
    // grabe the input tensor
    Tensor state = context->input(0);

    // prevent running on GPU
    OP_REQUIRES(
        context, (std::is_same<Device, CPUDevice>::value == true),
        errors::Unimplemented("ApplyGate operator not implemented for GPU."));

    if (UseMatrix) {
      const Tensor& gate = context->input(1);
      const Tensor& controls = context->input(2);
      const int ncontrols = controls.flat<int32>().size();

      // call the implementation
      F()(context, context->eigen_device<Device>(), state.flat<T>().data(),
          nqubits_, target_, ncontrols, controls.flat<int32>().data(),
          gate.flat<T>().data());
    }
    else {
      const Tensor& controls = context->input(1);
      const int ncontrols = controls.flat<int32>().size();

      // call the implementation
      F()(context, context->eigen_device<Device>(), state.flat<T>().data(),
          nqubits_, target_, ncontrols, controls.flat<int32>().data());
    }
    context->set_output(0, state);
  }

 private:
  int nqubits_;
  int target_;
};


template <typename Device, typename T, typename F, bool UseMatrix>
class TwoQubitGateOp : public OpKernel {
 public:
  explicit TwoQubitGateOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("nqubits", &nqubits_));
    OP_REQUIRES_OK(context, context->GetAttr("target1", &target1_));
    OP_REQUIRES_OK(context, context->GetAttr("target2", &target2_));
  }

  void Compute(OpKernelContext* context) override {
    // grabe the input tensor
    Tensor state = context->input(0);

    // prevent running on GPU
    OP_REQUIRES(
        context, (std::is_same<Device, CPUDevice>::value == true),
        errors::Unimplemented("ApplyTwoQubitGate operator not implemented for GPU."));

    if (UseMatrix) {
      const Tensor& gate = context->input(1);
      const Tensor& controls = context->input(2);
      const int ncontrols = controls.flat<int32>().size();

      // call the implementation
      F()(context, context->eigen_device<Device>(), state.flat<T>().data(),
          nqubits_, target1_, target2_, ncontrols,
          controls.flat<int32>().data(), gate.flat<T>().data());
    }
    else {
      const Tensor& controls = context->input(1);
      const int ncontrols = controls.flat<int32>().size();

      // call the implementation
      F()(context, context->eigen_device<Device>(), state.flat<T>().data(),
          nqubits_, target1_, target2_, ncontrols,
          controls.flat<int32>().data());
    }
    context->set_output(0, state);
  }

 private:
  int nqubits_;
  int target1_, target2_;
};


// Register the CPU kernels.
#define REGISTER_CPU(T, NAME, OP, FUNCTOR, USEMATRIX)         \
  REGISTER_KERNEL_BUILDER(                                    \
      Name(NAME).Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
      OP<CPUDevice, T, FUNCTOR<CPUDevice, T>, USEMATRIX>);

// Register the GPU kernels.
#define REGISTER_GPU(T, NAME, OP, FUNCTOR, USEMATRIX)        \
  extern template struct FUNCTOR<GPUDevice, T>;              \
  REGISTER_KERNEL_BUILDER(                                   \
      Name(NAME).Device(DEVICE_GPU).TypeConstraint<T>("T"),  \
      OP<GPUDevice, T, FUNCTOR<GPUDevice, T>, USEMATRIX>);

// Register one-qubit gate kernels.
#define REGISTER_ONEQUBIT(NAME, FUNCTOR, USEMATRIX)                     \
  REGISTER_CPU(complex64, NAME, OneQubitGateOp, FUNCTOR, USEMATRIX);    \
  REGISTER_CPU(complex128, NAME, OneQubitGateOp, FUNCTOR, USEMATRIX);   \
  REGISTER_GPU(complex64, NAME, OneQubitGateOp, FUNCTOR, USEMATRIX);    \
  REGISTER_GPU(complex128, NAME, OneQubitGateOp, FUNCTOR, USEMATRIX);

// Register two-qubit gate kernels.
#define REGISTER_TWOQUBIT(NAME, FUNCTOR, USEMATRIX)                     \
  REGISTER_CPU(complex64, NAME, TwoQubitGateOp, FUNCTOR, USEMATRIX);    \
  REGISTER_CPU(complex128, NAME, TwoQubitGateOp, FUNCTOR, USEMATRIX);   \
  REGISTER_GPU(complex64, NAME, TwoQubitGateOp, FUNCTOR, USEMATRIX);    \
  REGISTER_GPU(complex128, NAME, TwoQubitGateOp, FUNCTOR, USEMATRIX);


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
