#include "apply_gate.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

using thread::ThreadPool;


template <typename T>
struct BaseApplyGateFunctor<CPUDevice, T> {
  virtual void apply(T& state1, T& state2, const T* gate = NULL) const {}

  void work(int64 t, int64 w, T* state, const T* gate, int64 tk) const {
    for (auto g = t; g < w; g += 2 * tk) {
      for (auto i = g; i < g + tk; i++) {
          apply(state[i], state[i + tk], gate);
      }
    }
  }

  void singlecontrol_work(int64 t, int64 w, T* state, const T* gate,
                          int64 tk, int64 tk_reduced, int m) const {
    const int64 ck = (int64) 1 << m;
    for (auto g = t; g < w; g += 2 * tk_reduced) {
      for (auto i = g; i < g + tk_reduced; i++) {
        int64 i1 = ((int64) ((int64) i >> m) << (m + 1)) + (i & (ck - 1)) + ck;
        apply(state[i1], state[i1 + tk], gate);
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
        auto DoWork = [&](int64 t, int64 w) {
          singlecontrol_work(t, w, state, gate, tk, tk_reduced,
                             nqubits - controls[0] - 1);
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
        multicontrol_work(t, w, state, gate, tk, tk_reduced, m);
      };
      thread_pool->ParallelFor(nstates, p, DoWork);
    }
  }
};


// Apply general one-qubit gate via gate matrix
template <typename T>
struct ApplyGateFunctor<CPUDevice, T>: BaseApplyGateFunctor<CPUDevice, T> {
  inline void apply(T& state1, T& state2, const T* gate = NULL) const override {
    const auto buffer = state1;
    state1 = gate[0] * state1 + gate[1] * state2;
    state2 = gate[2] * buffer + gate[3] * state2;
  }
};


// Apply X gate via swap
template <typename T>
struct ApplyXFunctor<CPUDevice, T>: BaseApplyGateFunctor<CPUDevice, T> {
  inline void apply(T& state1, T& state2, const T* gate = NULL) const override {
    std::swap(state1, state2);
  }
};


// Apply Y gate via swap
template <typename T>
struct ApplyYFunctor<CPUDevice, T>: BaseApplyGateFunctor<CPUDevice, T> {
  inline void apply(T& state1, T& state2, const T* gate = NULL) const override {
    state1 *= T(0, 1);
    state2 *= - T(0, 1);
    std::swap(state1, state2);
  }
};


// Apply Z gate
template <typename T>
struct ApplyZFunctor<CPUDevice, T>: BaseApplyGateFunctor<CPUDevice, T> {
  inline void apply(T& state1, T& state2, const T* gate = NULL) const override {
    state2 *= -1;
  }
};


// Apply ZPow gate
template <typename T>
struct ApplyZPowFunctor<CPUDevice, T>: BaseApplyGateFunctor<CPUDevice, T> {
  inline void apply(T& state1, T& state2, const T* gate = NULL) const override {
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
    int m1 = nqubits - t1 - 1;
    int m2 = nqubits - t2 - 1;
    const int64 tk1 = (int64) 1 << m1;
    const int64 tk2 = (int64) 1 << m2;
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
    const ThreadPool::SchedulingParams p(
        ThreadPool::SchedulingStrategy::kFixedBlockSize, absl::nullopt,
        nreps);

    if (ncontrols == 0) {
      auto DoWork = [&](int64 t, int64 w) {
        for (auto g = t; g < w; g += 1) {
          int64 i = ((int64) ((int64) g >> m1) << (m1 + 1)) + (g & (tk1 - 1));
          i = ((int64) ((int64) i >> m2) << (m2 + 1)) + (i & (tk2 - 1));
          std::swap(state[i + tk1], state[i + tk2]);
        }
      };
      thread_pool->ParallelFor(nstates, p, DoWork);
    }
    else {
      std::vector<int> qubits(ncontrols + 2);
      int q = 0;
      for (int i = 0; i < ncontrols; i++) {
        if (q == 0 && controls[i] < t1) {
          qubits[i + q] = m1;
          q++;
        }
        if (q == 1 && controls[i] < t2) {
          qubits[i + q] = m2;
          q++;
        }
        qubits[i + q] = nqubits - controls[i] - 1;
      }
      if (q == 0) {
        qubits[ncontrols] = m1;
        qubits[ncontrols + 1] = m2;
      }
      else if (q == 1) {
        qubits[ncontrols + 1] = m2;
      }

      auto DoWork = [&](int64 t, int64 w) {
        for (auto g = t; g < w; g += 1) {
          int64 i = g;
          for (auto const& m : qubits) {
            int64 k = (int64) 1 << m;
            i = ((int64) ((int64) i >> m) << (m + 1)) + (i & (k - 1)) + k;
          }
          std::swap(state[i - tk2], state[i - tk1]);
        }
      };
      thread_pool->ParallelFor(nstates, p, DoWork);
    }
  }
};


template <typename Device, typename T, typename F, bool UseMatrix>
class ApplyGateOp : public OpKernel {
 public:
  explicit ApplyGateOp(OpKernelConstruction* context) : OpKernel(context) {
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
        errors::Unimplemented("ApplySwap operator not implemented for GPU."));

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
#define REGISTER_CPU(T, NAME, FUNCTOR, USEMATRIX)                   \
  REGISTER_KERNEL_BUILDER(                                          \
      Name(NAME).Device(DEVICE_CPU).TypeConstraint<T>("T"),         \
      ApplyGateOp<CPUDevice, T, FUNCTOR<CPUDevice, T>, USEMATRIX>);

REGISTER_CPU(complex64, "ApplyGate", ApplyGateFunctor, true);
REGISTER_CPU(complex128, "ApplyGate", ApplyGateFunctor, true);
REGISTER_CPU(complex64, "ApplyZPow", ApplyZPowFunctor, true);
REGISTER_CPU(complex128, "ApplyZPow", ApplyZPowFunctor, true);
REGISTER_CPU(complex64, "ApplyX", ApplyXFunctor, false);
REGISTER_CPU(complex128, "ApplyX", ApplyXFunctor, false);
REGISTER_CPU(complex64, "ApplyY", ApplyYFunctor, false);
REGISTER_CPU(complex128, "ApplyY", ApplyYFunctor, false);
REGISTER_CPU(complex64, "ApplyZ", ApplyZFunctor, false);
REGISTER_CPU(complex128, "ApplyZ", ApplyZFunctor, false);

// Register SWAP kernel on CPU.
#define REGISTER_SWAP_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("ApplySwap").Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
      ApplySwapOp<CPUDevice, T>);

REGISTER_SWAP_CPU(complex64);
REGISTER_SWAP_CPU(complex128);


// Register the GPU kernels.
#define REGISTER_GPU(T, NAME, FUNCTOR, USEMATRIX)                   \
  extern template struct FUNCTOR<GPUDevice, T>;                     \
  REGISTER_KERNEL_BUILDER(                                          \
      Name(NAME).Device(DEVICE_GPU).TypeConstraint<T>("T"),         \
      ApplyGateOp<GPUDevice, T, FUNCTOR<GPUDevice, T>, USEMATRIX>);

REGISTER_GPU(complex64, "ApplyGate", ApplyGateFunctor, true);
REGISTER_GPU(complex128, "ApplyGate", ApplyGateFunctor, true);
REGISTER_GPU(complex64, "ApplyZPow", ApplyZPowFunctor, true);
REGISTER_GPU(complex128, "ApplyZPow", ApplyZPowFunctor, true);
REGISTER_GPU(complex64, "ApplyX", ApplyXFunctor, false);
REGISTER_GPU(complex128, "ApplyX", ApplyXFunctor, false);
REGISTER_GPU(complex64, "ApplyY", ApplyYFunctor, false);
REGISTER_GPU(complex128, "ApplyY", ApplyYFunctor, false);
REGISTER_GPU(complex64, "ApplyZ", ApplyZFunctor, false);
REGISTER_GPU(complex128, "ApplyZ", ApplyZFunctor, false);

// Register SWAP kernel on GPU.
#define REGISTER_SWAP_GPU(T)                                        \
  extern template struct ApplySwapFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("ApplySwap").Device(DEVICE_GPU).TypeConstraint<T>("T"),  \
      ApplySwapOp<GPUDevice, T>);

REGISTER_SWAP_GPU(complex64);
REGISTER_SWAP_GPU(complex128);

}  // namespace functor
}  // namespace tensorflow
