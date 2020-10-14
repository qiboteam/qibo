#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#define DEFAULT_BLOCK_SIZE 1024  // default number of threads

#include "apply_gate.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__device__ T cmult(T a, T b) {
  return T(a.real() * b.real() - a.imag() * b.imag(),
           a.real() * b.imag() + a.imag() * b.real());
}

template <typename T>
__device__ T cadd(T a, T b) {
  return T(a.real() + b.real(), a.imag() + b.imag());
}

template <typename T>
struct BaseOneQubitGateFunctor<GPUDevice, T> {
  virtual void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                             T* state, const T* gate, long tk, int m) const {}

  virtual void multicontrolwork(const GPUDevice& d, int numBlocks,
                                int blockSize, T* state, const T* gate, long tk,
                                int m, int ncontrols, const int* qubits,
                                int nqubits, int target) const {}

  void operator()(const OpKernelContext* context, const GPUDevice& d, T* state,
                  int nqubits, int target, int ncontrols, const int32* qubits,
                  const T* gate = NULL) const {
    const int m = nqubits - target - 1;
    const int64 tk = (int64)1 << m;
    const int64 nstates = (int64)1 << (nqubits - ncontrols - 1);

    int blockSize = DEFAULT_BLOCK_SIZE;
    int numBlocks = (nstates + blockSize - 1) / blockSize;
    if (nstates < blockSize) {
      numBlocks = 1;
      blockSize = nstates;
    }

    if (ncontrols == 0) {
      nocontrolwork(d, numBlocks, blockSize, state, gate, tk, m);
    } else {
      multicontrolwork(d, numBlocks, blockSize, state, gate, tk, m, ncontrols,
                       qubits, nqubits, target);
    }
  }
};

template <typename T>
__device__ void apply_gate(T& state1, T& state2, const T* gate) {
  const auto buffer = state1;
  state1 = cadd(cmult(gate[0], state1), cmult(gate[1], state2));
  state2 = cadd(cmult(gate[2], buffer), cmult(gate[3], state2));
}

template <typename T>
__global__ void ApplyGateKernel(T* state, const T* gate, long tk, int m) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  apply_gate(state[i], state[i + tk], gate);
}

template <typename T>
__global__ void ApplyGateMultiControlKernel(T* state, const T* gate, long tk,
                                            int m, int ncontrols,
                                            const int* qubits, int nqubits,
                                            int target) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  auto i = g;
  for (auto iq = 0; iq < ncontrols + 1; iq++) {
    const auto n = qubits[iq];
    long k = (long)1 << n;
    i = ((long)((long)i >> n) << (n + 1)) + (i & (k - 1)) + k;
  }
  apply_gate(state[i - tk], state[i], gate);
}

// Apply general one-qubit gate via gate matrix
template <typename T>
struct ApplyGateFunctor<GPUDevice, T> : BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                            T* state, const T* gate, long tk,
                            int m) const override {
    ApplyGateKernel<T>
        <<<numBlocks, blockSize, 0, d.stream()>>>(state, gate, tk, m);
  }

  inline void multicontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                               T* state, const T* gate, long tk, int m,
                               int ncontrols, const int* qubits, int nqubits,
                               int target) const override {
    ApplyGateMultiControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, tk, m, ncontrols, qubits, nqubits, target);
  }
};

template <typename T>
__device__ void apply_x(T& state1, T& state2) {
  const auto buffer = state1;
  state1 = state2;
  state2 = buffer;
}

template <typename T>
__global__ void ApplyXKernel(T* state, const T* gate, long tk, int m) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  apply_x(state[i], state[i + tk]);
}

template <typename T>
__global__ void ApplyXMultiControlKernel(T* state, const T* gate, long tk,
                                         int m, int ncontrols,
                                         const int* qubits, int nqubits,
                                         int target) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  auto i = g;
  for (auto iq = 0; iq < ncontrols + 1; iq++) {
    const auto n = qubits[iq];
    long k = (long)1 << n;
    i = ((long)((long)i >> n) << (n + 1)) + (i & (k - 1)) + k;
  }
  apply_x(state[i - tk], state[i]);
}

// Apply X gate via swap
template <typename T>
struct ApplyXFunctor<GPUDevice, T> : BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                            T* state, const T* gate, long tk,
                            int m) const override {
    ApplyXKernel<T>
        <<<numBlocks, blockSize, 0, d.stream()>>>(state, gate, tk, m);
  }

  inline void multicontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                               T* state, const T* gate, long tk, int m,
                               int ncontrols, const int* qubits, int nqubits,
                               int target) const override {
    ApplyXMultiControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, tk, m, ncontrols, qubits, nqubits, target);
  }
};

template <typename T>
__device__ void apply_y(T& state1, T& state2) {
  state1 = cmult(state1, T(0, 1));
  state2 = cmult(state2, T(0, -1));
  const auto buffer = state1;
  state1 = state2;
  state2 = buffer;
}

template <typename T>
__global__ void ApplyYKernel(T* state, const T* gate, long tk, int m) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  apply_y(state[i], state[i + tk]);
}

template <typename T>
__global__ void ApplyYMultiControlKernel(T* state, const T* gate, long tk,
                                         int m, int ncontrols,
                                         const int* qubits, int nqubits,
                                         int target) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  auto i = g;
  for (auto iq = 0; iq < ncontrols + 1; iq++) {
    const auto n = qubits[iq];
    long k = (long)1 << n;
    i = ((long)((long)i >> n) << (n + 1)) + (i & (k - 1)) + k;
  }
  apply_y(state[i - tk], state[i]);
}

// Apply Y gate via swap
template <typename T>
struct ApplyYFunctor<GPUDevice, T> : BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                            T* state, const T* gate, long tk,
                            int m) const override {
    ApplyYKernel<T>
        <<<numBlocks, blockSize, 0, d.stream()>>>(state, gate, tk, m);
  }

  inline void multicontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                               T* state, const T* gate, long tk, int m,
                               int ncontrols, const int* qubits, int nqubits,
                               int target) const override {
    ApplyYMultiControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, tk, m, ncontrols, qubits, nqubits, target);
  }
};

template <typename T>
__device__ void apply_z(T& state) {
  state = cmult(state, T(-1));
}

template <typename T>
__global__ void ApplyZKernel(T* state, const T* gate, long tk, int m) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  apply_z(state[i + tk]);
}

template <typename T>
__global__ void ApplyZMultiControlKernel(T* state, const T* gate, long tk,
                                         int m, int ncontrols,
                                         const int* qubits, int nqubits,
                                         int target) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  auto i = g;
  for (auto iq = 0; iq < ncontrols + 1; iq++) {
    const auto n = qubits[iq];
    long k = (long)1 << n;
    i = ((long)((long)i >> n) << (n + 1)) + (i & (k - 1)) + k;
  }
  apply_z(state[i]);
}

// Apply Z gate
template <typename T>
struct ApplyZFunctor<GPUDevice, T> : BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                            T* state, const T* gate, long tk,
                            int m) const override {
    ApplyZKernel<T>
        <<<numBlocks, blockSize, 0, d.stream()>>>(state, gate, tk, m);
  }

  inline void multicontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                               T* state, const T* gate, long tk, int m,
                               int ncontrols, const int* qubits, int nqubits,
                               int target) const override {
    ApplyZMultiControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, tk, m, ncontrols, qubits, nqubits, target);
  }
};

template <typename T>
__device__ void apply_zpow(T& state, T gate) {
  state = cmult(state, gate);
}

template <typename T>
__global__ void ApplyZPowKernel(T* state, const T* gate, long tk, int m) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  apply_zpow(state[i + tk], gate[0]);
}

template <typename T>
__global__ void ApplyZPowMultiControlKernel(T* state, const T* gate, long tk,
                                            int m, int ncontrols,
                                            const int* qubits, int nqubits,
                                            int target) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  auto i = g;
  for (auto iq = 0; iq < ncontrols + 1; iq++) {
    const auto n = qubits[iq];
    long k = (long)1 << n;
    i = ((long)((long)i >> n) << (n + 1)) + (i & (k - 1)) + k;
  }
  apply_zpow(state[i], gate[0]);
}

// Apply ZPow gate
template <typename T>
struct ApplyZPowFunctor<GPUDevice, T> : BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                            T* state, const T* gate, long tk,
                            int m) const override {
    ApplyZPowKernel<T>
        <<<numBlocks, blockSize, 0, d.stream()>>>(state, gate, tk, m);
  }

  inline void multicontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                               T* state, const T* gate, long tk, int m,
                               int ncontrols, const int* qubits, int nqubits,
                               int target) const override {
    ApplyZPowMultiControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, tk, m, ncontrols, qubits, nqubits, target);
  }
};

template <typename T>
struct BaseTwoQubitGateFunctor<GPUDevice, T> {
  virtual void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                             T* state, const T* gate, long ctk1, long ctk2,
                             long tk1, long tk2, int m1, int m2) const {}

  virtual void multicontrolwork(const GPUDevice& d, int numBlocks,
                                int blockSize, T* state, const T* gate,
                                long ctk1, long ctk2, long tk1, long tk2,
                                int m1, int m2, int ncontrols,
                                const int* qubits, int nqubits, int t1,
                                int t2) const {}

  void operator()(const OpKernelContext* context, const GPUDevice& d, T* state,
                  int nqubits, int target1, int target2, int ncontrols,
                  const int32* qubits, const T* gate = NULL) const {
    const int t1 = std::max(target1, target2);
    const int t2 = std::min(target1, target2);
    int m1 = nqubits - t1 - 1;
    int m2 = nqubits - t2 - 1;
    const int64 tk1 = (int64)1 << m1;
    const int64 tk2 = (int64)1 << m2;
    const int64 nstates = (int64)1 << (nqubits - 2 - ncontrols);

    int64 targetk1 = tk1;
    int64 targetk2 = tk2;
    if (target1 > target2) {
      std::swap(targetk1, targetk2);
    }

    int blockSize = DEFAULT_BLOCK_SIZE;
    int numBlocks = (nstates + blockSize - 1) / blockSize;
    if (nstates < blockSize) {
      numBlocks = 1;
      blockSize = nstates;
    }

    if (ncontrols == 0) {
      nocontrolwork(d, numBlocks, blockSize, state, gate, tk1, tk2, targetk1,
                    targetk2, m1, m2);
    } else {
      multicontrolwork(d, numBlocks, blockSize, state, gate, tk1, tk2, targetk1,
                       targetk2, m1, m2, ncontrols, qubits, nqubits, t1, t2);
    }
  };
};

template <typename T>
__device__ void apply_two_gate(T* state, long i, long tk1, long tk2,
                               const T* gate = NULL) {
  const auto i1 = i + tk1;
  const auto i2 = i + tk2;
  const auto i3 = i1 + tk2;
  const auto buffer = state[i];
  state[i] = cadd(cadd(cmult(gate[0], state[i]), cmult(gate[1], state[i1])),
                  cadd(cmult(gate[2], state[i2]), cmult(gate[3], state[i3])));
  const auto buffer1 = state[i1];
  state[i1] = cadd(cadd(cmult(gate[4], buffer), cmult(gate[5], state[i1])),
                   cadd(cmult(gate[6], state[i2]), cmult(gate[7], state[i3])));
  const auto buffer2 = state[i2];
  state[i2] =
      cadd(cadd(cmult(gate[8], buffer), cmult(gate[9], buffer1)),
           cadd(cmult(gate[10], state[i2]), cmult(gate[11], state[i3])));
  state[i3] = cadd(cadd(cmult(gate[12], buffer), cmult(gate[13], buffer1)),
                   cadd(cmult(gate[14], buffer2), cmult(gate[15], state[i3])));
}

template <typename T>
__global__ void ApplyTwoQubitGateKernel(T* state, const T* gate, long ctk1,
                                        long ctk2, long tk1, long tk2, int m1,
                                        int m2) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  auto i = ((long)((long)g >> m1) << (m1 + 1)) + (g & (ctk1 - 1));
  i = ((long)((long)i >> m2) << (m2 + 1)) + (i & (ctk2 - 1));
  apply_two_gate(state, i, tk1, tk2, gate);
}

template <typename T>
__global__ void ApplyTwoQubitGateMultiControlKernel(
    T* state, const T* gate, long ctk1, long ctk2, long tk1, long tk2, int m1,
    int m2, int ncontrols, const int* qubits, int nqubits, int t1, int t2) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  auto i = g;
  for (auto iq = 0; iq < ncontrols + 2; iq++) {
    const auto m = qubits[iq];
    long k = (long)1 << m;
    i = ((long)((long)i >> m) << (m + 1)) + (i & (k - 1)) + k;
  }
  apply_two_gate(state, i - ctk1 - ctk2, tk1, tk2, gate);
}

// Apply general one-qubit gate via gate matrix
template <typename T>
struct ApplyTwoQubitGateFunctor<GPUDevice, T>
    : BaseTwoQubitGateFunctor<GPUDevice, T> {
  inline void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                            T* state, const T* gate, long ctk1, long ctk2,
                            long tk1, long tk2, int m1, int m2) const override {
    ApplyTwoQubitGateKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, ctk1, ctk2, tk1, tk2, m1, m2);
  }

  inline void multicontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                               T* state, const T* gate, long ctk1, long ctk2,
                               long tk1, long tk2, int m1, int m2,
                               int ncontrols, const int* qubits, int nqubits,
                               int t1, int t2) const override {
    ApplyTwoQubitGateMultiControlKernel<T>
        <<<numBlocks, blockSize, 0, d.stream()>>>(state, gate, ctk1, ctk2, tk1,
                                                  tk2, m1, m2, ncontrols,
                                                  qubits, nqubits, t1, t2);
  }
};

template <typename T>
__device__ void apply_fsim(T* state, long i, long tk1, long tk2,
                           const T* gate = NULL) {
  const auto i1 = i + tk1;
  const auto i2 = i + tk2;
  const auto i3 = i1 + tk2;
  const auto buffer = state[i1];
  state[i1] = cadd(cmult(gate[0], state[i1]), cmult(gate[1], state[i2]));
  state[i2] = cadd(cmult(gate[2], buffer), cmult(gate[3], state[i2]));
  state[i3] = cmult(gate[4], state[i3]);
}

template <typename T>
__global__ void ApplyFsimKernel(T* state, const T* gate, long ctk1, long ctk2,
                                long tk1, long tk2, int m1, int m2) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  auto i = ((long)((long)g >> m1) << (m1 + 1)) + (g & (ctk1 - 1));
  i = ((long)((long)i >> m2) << (m2 + 1)) + (i & (ctk2 - 1));
  apply_fsim(state, i, tk1, tk2, gate);
}

template <typename T>
__global__ void ApplyFsimMultiControlKernel(T* state, const T* gate, long ctk1,
                                            long ctk2, long tk1, long tk2,
                                            int m1, int m2, int ncontrols,
                                            const int* qubits, int nqubits,
                                            int t1, int t2) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  auto i = g;
  for (auto iq = 0; iq < ncontrols + 2; iq++) {
    const auto m = qubits[iq];
    long k = (long)1 << m;
    i = ((long)((long)i >> m) << (m + 1)) + (i & (k - 1)) + k;
  }
  apply_fsim(state, i - ctk1 - ctk2, tk1, tk2, gate);
}

template <typename T>
struct ApplyFsimFunctor<GPUDevice, T> : BaseTwoQubitGateFunctor<GPUDevice, T> {
  inline void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                            T* state, const T* gate, long ctk1, long ctk2,
                            long tk1, long tk2, int m1, int m2) const override {
    ApplyFsimKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, ctk1, ctk2, tk1, tk2, m1, m2);
  }

  inline void multicontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                               T* state, const T* gate, long ctk1, long ctk2,
                               long tk1, long tk2, int m1, int m2,
                               int ncontrols, const int* qubits, int nqubits,
                               int t1, int t2) const override {
    ApplyFsimMultiControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, ctk1, ctk2, tk1, tk2, m1, m2, ncontrols, qubits, nqubits,
        t1, t2);
  }
};

template <typename T>
__device__ void apply_swap(T* state, long i, long tk1, long tk2) {
  const auto buffer = state[i + tk1];
  state[i + tk1] = state[i + tk2];
  state[i + tk2] = buffer;
}

template <typename T>
__global__ void ApplySwapKernel(T* state, const T* gate, long ctk1, long ctk2,
                                long tk1, long tk2, int m1, int m2) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  auto i = ((long)((long)g >> m1) << (m1 + 1)) + (g & (ctk1 - 1));
  i = ((long)((long)i >> m2) << (m2 + 1)) + (i & (ctk2 - 1));
  apply_swap(state, i, tk1, tk2);
}

template <typename T>
__global__ void ApplySwapMultiControlKernel(T* state, const T* gate, long ctk1,
                                            long ctk2, long tk1, long tk2,
                                            int m1, int m2, int ncontrols,
                                            const int* qubits, int nqubits,
                                            int t1, int t2) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  auto i = g;
  for (auto iq = 0; iq < ncontrols + 2; iq++) {
    const auto m = qubits[iq];
    long k = (long)1 << m;
    i = ((long)((long)i >> m) << (m + 1)) + (i & (k - 1)) + k;
  }
  apply_swap(state, i - ctk1 - ctk2, tk1, tk2);
}

// Apply SWAP gate
template <typename T>
struct ApplySwapFunctor<GPUDevice, T> : BaseTwoQubitGateFunctor<GPUDevice, T> {
  inline void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                            T* state, const T* gate, long ctk1, long ctk2,
                            long tk1, long tk2, int m1, int m2) const override {
    ApplySwapKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, ctk1, ctk2, tk1, tk2, m1, m2);
  }

  inline void multicontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                               T* state, const T* gate, long ctk1, long ctk2,
                               long tk1, long tk2, int m1, int m2,
                               int ncontrols, const int* qubits, int nqubits,
                               int t1, int t2) const override {
    ApplySwapMultiControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, ctk1, ctk2, tk1, tk2, m1, m2, ncontrols, qubits, nqubits,
        t1, t2);
  }
};


template <typename T>
__device__ void zero_state(T* state, long g, long h, int ntargets) {
  long i = g;
  for (auto iq = 0; iq < ntargets; iq++) {
    const auto n = qubits[iq];
    long k = (long)1 << n;
    i = ((long)((long)i >> n) << (n + 1)) + (i & (k - 1));
    i += ((long)((int)(h >> iq) % 2) * k);
  }
  state[i] = 0;
}

template <typename T>
__global__ void CollapseStateKernel(T* state, const int* qubits,
                                    long result, long nsubstates,
                                    int ntargets) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  for (auto h = 0; h < result; h++) {
    zero_state(state, g, h, ntargets);
  }
  //norm += CalcNorm(state[GetIndex(g, result)]);
  for (auto h = result + 1; h < nsubstates; h++) {
    zero_state(state, g, h, ntargets);
  }
}

// Collapse state gate
template <typename T>
struct CollapseStateFunctor<GPUDevice, T> {
  void operator()(const OpKernelContext* context, const GPUDevice& d, T* state,
                  int nqubits, int ntargets, const int32* qubits,
                  const int64* result) const {
    int64 nstates = (int64)1 << (nqubits - ntargets);
    int64 nsubstates = (int64)1 << ntargets;

    int blockSize = DEFAULT_BLOCK_SIZE;
    int numBlocks = (nstates + blockSize - 1) / blockSize;
    if (nstates < blockSize) {
      numBlocks = 1;
      blockSize = nstates;
    }

    CollapseStateKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, qubits, result[0], nsubstates, ntargets);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
#define REGISTER_TEMPLATE(FUNCTOR)               \
  template struct FUNCTOR<GPUDevice, complex64>; \
  template struct FUNCTOR<GPUDevice, complex128>;

REGISTER_TEMPLATE(BaseOneQubitGateFunctor);
REGISTER_TEMPLATE(ApplyGateFunctor);
REGISTER_TEMPLATE(ApplyXFunctor);
REGISTER_TEMPLATE(ApplyYFunctor);
REGISTER_TEMPLATE(ApplyZFunctor);
REGISTER_TEMPLATE(ApplyZPowFunctor);
REGISTER_TEMPLATE(BaseTwoQubitGateFunctor);
REGISTER_TEMPLATE(ApplyTwoQubitGateFunctor);
REGISTER_TEMPLATE(ApplyFsimFunctor);
REGISTER_TEMPLATE(ApplySwapFunctor);
REGISTER_TEMPLATE(CollapseStateFunctor);
}  // end namespace functor
}  // end namespace tensorflow
#endif  // GOOGLE_CUDA
