#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "apply_gate.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;


template <typename T>
struct BaseOneQubitGateFunctor<GPUDevice, T> {
  virtual void apply(T& state1, T& state2, const T* gate = NULL) const {}

  void work(int64 t, int64 w, T* state, const T* gate, int64 tk) const {
    for (auto g = t; g < w; g += 2 * tk) {
      for (auto i = g; i < g + tk; i++) {
          apply(state[i], state[i + tk], gate);
      }
    }
  }

  void operator()(const OpKernelContext* context, const GPUDevice& d, T* state,
                  int nqubits, int target, int ncontrols,
                  const int32* controls, const T* gate = NULL) const {
    const int64 tk = (int64) 1 << (nqubits - target - 1);
    const int64 nstates = (int64) 1 << (nqubits - ncontrols);
    int target_eff = target;
    for (int i = 0; i < ncontrols; i++) {
      if (controls[i] < target) {
        target_eff--;
      }
    }
    const int64 tk_reduced = (int64) 1 << (nqubits - target_eff - ncontrols - 1);
    /*
    // Set multi-threading
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

    // Apply gate
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
      std::vector<int64> controls_vec(ncontrols);
      for (int i = 0; i < ncontrols; i++) {
        controls_vec[i] = nqubits - controls[i] - 1;
      }
      auto DoWork = [&](int64 t, int64 w) {
        multicontrol_work(t, w, state, gate, tk, tk_reduced, controls_vec);
      };
      thread_pool->ParallelFor(nstates, p, DoWork);
    }
    */
  }
};

template<typename T>
__device__ T cmult(T a, T b)
{
  return T(a.real() * b.real() - a.imag() * b.imag(),
           a.real() * b.imag() + a.imag() * b.real());
}

template<typename T>
__device__ T cadd(T a, T b)
{
  return T(a.real() + b.real(), a.imag() + b.imag());
}

// Apply general one-qubit gate via gate matrix
template <typename T>
struct ApplyGateFunctor<GPUDevice, T>: BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void apply(T& state1, T& state2, const T* gate = NULL) const override {
    const auto buffer = state1;
    state1 = gate[0] * state1 + gate[1] * state2;
    state2 = gate[2] * buffer + gate[3] * state2;
  }
};


// Apply X gate via swap
template <typename T>
struct ApplyXFunctor<GPUDevice, T>: BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void apply(T& state1, T& state2, const T* gate = NULL) const override {
    std::swap(state1, state2);
  }
};


// Apply Y gate via swap
template <typename T>
struct ApplyYFunctor<GPUDevice, T>: BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void apply(T& state1, T& state2, const T* gate = NULL) const override {
    state1 *= T(0, 1);
    state2 *= - T(0, 1);
    std::swap(state1, state2);
  }
};


// Apply Z gate
template <typename T>
struct ApplyZFunctor<GPUDevice, T>: BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void apply(T& state1, T& state2, const T* gate = NULL) const override {
    state2 *= -1;
  }
};


// Apply ZPow gate
template <typename T>
struct ApplyZPowFunctor<GPUDevice, T>: BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void apply(T& state1, T& state2, const T* gate = NULL) const override {
    state2 *= gate[0];
  }
};

template <typename T>
struct BaseTwoQubitGateFunctor<GPUDevice, T> {
  virtual void apply(T* state, int64 i, int64 tk1, int64 tk2,
                     const T* gate = NULL) const {}

  void operator()(const OpKernelContext* context, const GPUDevice& d, T* state,
                  int nqubits, int target1, int target2, int ncontrols,
                  const int32* controls, const T* gate = NULL) const {
    const int t1 = std::max(target1, target2);
    const int t2 = std::min(target1, target2);
    int m1 = nqubits - t1 - 1;
    int m2 = nqubits - t2 - 1;
    const int64 tk1 = (int64) 1 << m1;
    const int64 tk2 = (int64) 1 << m2;
    const int64 nstates = (int64) 1 << (nqubits - 2 - ncontrols);

    /*
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
          apply(state, i, tk1, tk2, gate);
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
          apply(state, i - tk1 - tk2, tk1, tk2, gate);
        }
      };
      thread_pool->ParallelFor(nstates, p, DoWork);
    }
    */
  }
};


// Apply general one-qubit gate via gate matrix
template <typename T>
struct ApplyTwoQubitGateFunctor<GPUDevice, T>: BaseTwoQubitGateFunctor<GPUDevice, T> {
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
struct ApplyFsimFunctor<GPUDevice, T>: BaseTwoQubitGateFunctor<GPUDevice, T> {
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
struct ApplySwapFunctor<GPUDevice, T>: BaseTwoQubitGateFunctor<GPUDevice, T> {
  inline void apply(T* state, int64 i, int64 tk1, int64 tk2,
                    const T* gate = NULL) const {
    std::swap(state[i + tk1], state[i + tk2]);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
#define REGISTER_TEMPLATE(FUNCTOR)                 \
    template struct FUNCTOR<GPUDevice, complex64>; \
    template struct FUNCTOR<GPUDevice, complex128>;

REGISTER_TEMPLATE(BaseOneQubitGateFunctor);
REGISTER_TEMPLATE(BaseTwoQubitGateFunctor);
REGISTER_TEMPLATE(ApplyGateFunctor);
REGISTER_TEMPLATE(ApplyXFunctor);
REGISTER_TEMPLATE(ApplyYFunctor);
REGISTER_TEMPLATE(ApplyZFunctor);
REGISTER_TEMPLATE(ApplyZPowFunctor);
REGISTER_TEMPLATE(ApplyTwoQubitGateFunctor);
REGISTER_TEMPLATE(ApplyFsimFunctor);
REGISTER_TEMPLATE(ApplySwapFunctor);
} // end namespace functor
} // end namespace tensorflow
#endif // GOOGLE_CUDA