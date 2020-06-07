#if GOOGLE_CUDA
#define EIGEN_USE_GPU

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
                             T* state, const T* gate, long tk) const {}

  virtual void singlecontrolwork(const GPUDevice& d, int numBlocks,
                                 int blockSize, T* state, const T* gate,
                                 long tk, long tk_reduced, int c) const {}

  virtual void multicontrolwork(const GPUDevice& d, int numBlocks,
                                int blockSize, T* state, const T* gate, long tk,
                                long tk_reduced, int ncontrols,
                                const int* controls, int nqubits) const {}

  void operator()(const OpKernelContext* context, const GPUDevice& d, T* state,
                  int nqubits, int target, int ncontrols, const int32* controls,
                  const int32* tensor_controls, const T* gate = NULL) const {
    const int64 tk = (int64)1 << (nqubits - target - 1);
    const int64 nstates = (int64)1 << (nqubits - ncontrols);
    int target_eff = target;
    for (int i = 0; i < ncontrols; i++) {
      if (controls[i] < target) {
        target_eff--;
      }
    }
    const int64 tk_reduced = (int64)1 << (nqubits - target_eff - ncontrols - 1);

    int64 nreps = nstates;
    if (nreps % (2 * tk_reduced)) {
      nreps = 2 * tk_reduced;
    }
    int blockSize = 1024;
    int numBlocks = (nreps / 2 + blockSize - 1) / blockSize;
    if (nreps / 2 < blockSize) {
      numBlocks = 1;
      blockSize = nreps / 2;
    }

    if (ncontrols == 0) {
      nocontrolwork(d, numBlocks, blockSize, state, gate, tk);
    } else if (ncontrols == 1) {
      singlecontrolwork(d, numBlocks, blockSize, state, gate, tk, tk_reduced,
                        nqubits - controls[0] - 1);
    } else {
      multicontrolwork(d, numBlocks, blockSize, state, gate, tk, tk_reduced,
                       ncontrols, tensor_controls, nqubits);
    }
  }
};

template <typename T>
__global__ void ApplyGateKernel(T* state, const T* gate, long tk) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % tk + 2 * tk * int(index / tk);
  const auto state1 = state[i];
  const auto state2 = state[i + tk];
  const auto buffer = state1;
  state[i] = cadd(cmult(gate[0], state1), cmult(gate[1], state2));
  state[i + tk] = cadd(cmult(gate[2], buffer), cmult(gate[3], state2));
}

template <typename T>
__global__ void ApplyGateSingleControlKernel(T* state, const T* gate, long tk,
                                             long tk_reduced, int c) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % tk_reduced + 2 * tk_reduced * int(index / tk_reduced);
  const long ck = (long)1 << c;
  const long i1 = ((long)((long)i >> c) << (c + 1)) + (i & (ck - 1)) + ck;
  const auto state1 = state[i1];
  const auto state2 = state[i1 + tk];
  const auto buffer = state1;
  state[i1] = cadd(cmult(gate[0], state1), cmult(gate[1], state2));
  state[i1 + tk] = cadd(cmult(gate[2], buffer), cmult(gate[3], state2));
}

template <typename T>
__global__ void ApplyGateMultiControlKernel(T* state, const T* gate, long tk,
                                            long tk_reduced, int ncontrols,
                                            const int* controls, int nqubits) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % tk_reduced + 2 * tk_reduced * int(index / tk_reduced);
  auto i1 = i;
  for (auto ic = 0; ic < ncontrols; ic++) {
    const long c = nqubits - controls[ic] - 1;
    const long ck = (long)1 << c;
    i1 = ((long)((long)i1 >> c) << (c + 1)) + (i1 & (ck - 1)) + ck;
  }
  const auto state1 = state[i1];
  const auto state2 = state[i1 + tk];
  const auto buffer = state1;
  state[i1] = cadd(cmult(gate[0], state1), cmult(gate[1], state2));
  state[i1 + tk] = cadd(cmult(gate[2], buffer), cmult(gate[3], state2));
}

// Apply general one-qubit gate via gate matrix
template <typename T>
struct ApplyGateFunctor<GPUDevice, T> : BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                            T* state, const T* gate, long tk) const override {
    ApplyGateKernel<T>
        <<<numBlocks, blockSize, 0, d.stream()>>>(state, gate, tk);
  }

  inline void singlecontrolwork(const GPUDevice& d, int numBlocks,
                                int blockSize, T* state, const T* gate, long tk,
                                long tk_reduced, int c) const override {
    ApplyGateSingleControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, tk, tk_reduced, c);
  }

  inline void multicontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                               T* state, const T* gate, long tk,
                               long tk_reduced, int ncontrols,
                               const int* controls,
                               int nqubits) const override {
    ApplyGateMultiControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, tk, tk_reduced, ncontrols, controls, nqubits);
  }
};

template <typename T>
__global__ void ApplyXKernel(T* state, const T* gate, long tk) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % tk + 2 * tk * int(index / tk);
  const auto buffer = state[i];
  state[i] = state[i + tk];
  state[i + tk] = buffer;
}

template <typename T>
__global__ void ApplyXSingleControlKernel(T* state, const T* gate, long tk,
                                          long tk_reduced, int c) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % tk_reduced + 2 * tk_reduced * int(index / tk_reduced);
  const long ck = (long)1 << c;
  const long i1 = ((long)((long)i >> c) << (c + 1)) + (i & (ck - 1)) + ck;
  const auto buffer = state[i1];
  state[i1] = state[i1 + tk];
  state[i1 + tk] = buffer;
}

template <typename T>
__global__ void ApplyXMultiControlKernel(T* state, const T* gate, long tk,
                                         long tk_reduced, int ncontrols,
                                         const int* controls, int nqubits) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % tk_reduced + 2 * tk_reduced * int(index / tk_reduced);
  auto i1 = i;
  for (auto ic = 0; ic < ncontrols; ic++) {
    const long c = nqubits - controls[ic] - 1;
    const long ck = (long)1 << c;
    i1 = ((long)((long)i1 >> c) << (c + 1)) + (i1 & (ck - 1)) + ck;
  }
  const auto buffer = state[i1];
  state[i1] = state[i1 + tk];
  state[i1 + tk] = buffer;
}

// Apply X gate via swap
template <typename T>
struct ApplyXFunctor<GPUDevice, T> : BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                            T* state, const T* gate, long tk) const override {
    ApplyXKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(state, gate, tk);
  }

  inline void singlecontrolwork(const GPUDevice& d, int numBlocks,
                                int blockSize, T* state, const T* gate, long tk,
                                long tk_reduced, int c) const override {
    ApplyXSingleControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, tk, tk_reduced, c);
  }

  inline void multicontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                               T* state, const T* gate, long tk,
                               long tk_reduced, int ncontrols,
                               const int* controls,
                               int nqubits) const override {
    ApplyXMultiControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, tk, tk_reduced, ncontrols, controls, nqubits);
  }
};

template <typename T>
__global__ void ApplyYKernel(T* state, const T* gate, long tk) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % tk + 2 * tk * int(index / tk);
  state[i] = cmult(state[i], T(0, 1));
  state[i + tk] = cmult(state[i + tk], T(0, -1));
  const auto buffer = state[i];
  state[i] = state[i + tk];
  state[i + tk] = buffer;
}

template <typename T>
__global__ void ApplyYSingleControlKernel(T* state, const T* gate, long tk,
                                          long tk_reduced, int c) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % tk_reduced + 2 * tk_reduced * int(index / tk_reduced);
  const long ck = (long)1 << c;
  const long i1 = ((long)((long)i >> c) << (c + 1)) + (i & (ck - 1)) + ck;
  state[i1] = cmult(state[i1], T(0, 1));
  state[i1 + tk] = cmult(state[i1 + tk], T(0, -1));
  const auto buffer = state[i1];
  state[i1] = state[i1 + tk];
  state[i1 + tk] = buffer;
}

template <typename T>
__global__ void ApplyYMultiControlKernel(T* state, const T* gate, long tk,
                                         long tk_reduced, int ncontrols,
                                         const int* controls, int nqubits) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % tk_reduced + 2 * tk_reduced * int(index / tk_reduced);
  auto i1 = i;
  for (auto ic = 0; ic < ncontrols; ic++) {
    const long c = nqubits - controls[ic] - 1;
    const long ck = (long)1 << c;
    i1 = ((long)((long)i1 >> c) << (c + 1)) + (i1 & (ck - 1)) + ck;
  }
  state[i1] = cmult(state[i1], T(0, 1));
  state[i1 + tk] = cmult(state[i1 + tk], T(0, -1));
  const auto buffer = state[i1];
  state[i1] = state[i1 + tk];
  state[i1 + tk] = buffer;
}

// Apply Y gate via swap
template <typename T>
struct ApplyYFunctor<GPUDevice, T> : BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                            T* state, const T* gate, long tk) const override {
    ApplyYKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(state, gate, tk);
  }

  inline void singlecontrolwork(const GPUDevice& d, int numBlocks,
                                int blockSize, T* state, const T* gate, long tk,
                                long tk_reduced, int c) const override {
    ApplyYSingleControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, tk, tk_reduced, c);
  }

  inline void multicontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                               T* state, const T* gate, long tk,
                               long tk_reduced, int ncontrols,
                               const int* controls,
                               int nqubits) const override {
    ApplyYMultiControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, tk, tk_reduced, ncontrols, controls, nqubits);
  }
};

template <typename T>
__global__ void ApplyZKernel(T* state, const T* gate, long tk) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % tk + 2 * tk * int(index / tk);
  state[i + tk] = cmult(state[i + tk], T(-1));
}

template <typename T>
__global__ void ApplyZSingleControlKernel(T* state, const T* gate, long tk,
                                          long tk_reduced, int c) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % tk_reduced + 2 * tk_reduced * int(index / tk_reduced);
  const long ck = (long)1 << c;
  const long i1 = ((long)((long)i >> c) << (c + 1)) + (i & (ck - 1)) + ck;
  state[i1 + tk] = cmult(state[i1 + tk], T(-1));
}

template <typename T>
__global__ void ApplyZMultiControlKernel(T* state, const T* gate, long tk,
                                         long tk_reduced, int ncontrols,
                                         const int* controls, int nqubits) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % tk_reduced + 2 * tk_reduced * int(index / tk_reduced);
  auto i1 = i;
  for (auto ic = 0; ic < ncontrols; ic++) {
    const long c = nqubits - controls[ic] - 1;
    const long ck = (long)1 << c;
    i1 = ((long)((long)i1 >> c) << (c + 1)) + (i1 & (ck - 1)) + ck;
  }
  state[i1 + tk] = cmult(state[i1 + tk], T(-1));
}

// Apply Z gate
template <typename T>
struct ApplyZFunctor<GPUDevice, T> : BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                            T* state, const T* gate, long tk) const override {
    ApplyZKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(state, gate, tk);
  }

  inline void singlecontrolwork(const GPUDevice& d, int numBlocks,
                                int blockSize, T* state, const T* gate, long tk,
                                long tk_reduced, int c) const override {
    ApplyZSingleControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, tk, tk_reduced, c);
  }

  inline void multicontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                               T* state, const T* gate, long tk,
                               long tk_reduced, int ncontrols,
                               const int* controls,
                               int nqubits) const override {
    ApplyZMultiControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, tk, tk_reduced, ncontrols, controls, nqubits);
  }
};

template <typename T>
__global__ void ApplyZPowKernel(T* state, const T* gate, long tk) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % tk + 2 * tk * int(index / tk);
  state[i + tk] = cmult(state[i + tk], gate[0]);
}

template <typename T>
__global__ void ApplyZPowSingleControlKernel(T* state, const T* gate, long tk,
                                             long tk_reduced, int c) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % tk_reduced + 2 * tk_reduced * int(index / tk_reduced);
  const long ck = (long)1 << c;
  const long i1 = ((long)((long)i >> c) << (c + 1)) + (i & (ck - 1)) + ck;
  state[i1 + tk] = cmult(state[i1 + tk], gate[0]);
}

template <typename T>
__global__ void ApplyZPowMultiControlKernel(T* state, const T* gate, long tk,
                                            long tk_reduced, int ncontrols,
                                            const int* controls, int nqubits) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % tk_reduced + 2 * tk_reduced * int(index / tk_reduced);
  auto i1 = i;
  for (auto ic = 0; ic < ncontrols; ic++) {
    const long c = nqubits - controls[ic] - 1;
    const long ck = (long)1 << c;
    i1 = ((long)((long)i1 >> c) << (c + 1)) + (i1 & (ck - 1)) + ck;
  }
  state[i1 + tk] = cmult(state[i1 + tk], gate[0]);
}

// Apply ZPow gate
template <typename T>
struct ApplyZPowFunctor<GPUDevice, T> : BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                            T* state, const T* gate, long tk) const override {
    ApplyZPowKernel<T>
        <<<numBlocks, blockSize, 0, d.stream()>>>(state, gate, tk);
  }

  inline void singlecontrolwork(const GPUDevice& d, int numBlocks,
                                int blockSize, T* state, const T* gate, long tk,
                                long tk_reduced, int c) const override {
    ApplyZPowSingleControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, tk, tk_reduced, c);
  }

  inline void multicontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                               T* state, const T* gate, long tk,
                               long tk_reduced, int ncontrols,
                               const int* controls,
                               int nqubits) const override {
    ApplyZPowMultiControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, tk, tk_reduced, ncontrols, controls, nqubits);
  }
};

template <typename T>
struct BaseTwoQubitGateFunctor<GPUDevice, T> {
  virtual void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                             T* state, const T* gate, long tk1, long tk2,
                             int m1, int m2) const {}

  virtual void multicontrolwork(const GPUDevice& d, int numBlocks,
                                int blockSize, T* state, const T* gate,
                                long ctk1, long ctk2, long tk1, long tk2,
                                int m1, int m2, int ncontrols,
                                const int* controls, int nqubits, int t1,
                                int t2) const {}

  void operator()(const OpKernelContext* context, const GPUDevice& d, T* state,
                  int nqubits, int target1, int target2, int ncontrols,
                  const int32* controls, const int32* tensor_controls,
                  const T* gate = NULL) const {
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

    int64 nreps = nstates;
    int blockSize = 1024;
    int numBlocks = (nreps + blockSize - 1) / blockSize;
    if (nreps < blockSize) {
      numBlocks = 1;
      blockSize = nreps;
    }

    if (ncontrols == 0) {
      nocontrolwork(d, numBlocks, blockSize, state, gate, targetk1, targetk2,
                    m1, m2);
    } else {
      multicontrolwork(d, numBlocks, blockSize, state, gate, tk1, tk2, targetk1,
                       targetk2, m1, m2, ncontrols, tensor_controls, nqubits,
                       t1, t2);
    }
  };
};

template <typename T>
__global__ void ApplyTwoQubitGateKernel(T* state, const T* gate, long tk1,
                                        long tk2, int m1, int m2) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  auto i = ((long)((long)g >> m1) << (m1 + 1)) + (g & (tk1 - 1));
  i = ((long)((long)i >> m2) << (m2 + 1)) + (i & (tk2 - 1));

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
__global__ void ApplyTwoQubitGateMultiControlKernel(
    T* state, const T* gate, long ctk1, long ctk2, long tk1, long tk2, int m1,
    int m2, int ncontrols, const int* controls, int nqubits, int t1, int t2) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  auto i = g;

  int* qubits = new int[ncontrols + 2];
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
  } else if (q == 1) {
    qubits[ncontrols + 1] = m2;
  }

  for (auto iq = 0; iq < ncontrols + 2; iq++) {
    const auto m = qubits[iq];
    long k = (long)1 << m;
    i = ((long)((long)i >> m) << (m + 1)) + (i & (k - 1)) + k;
  }
  delete[] qubits;

  i = i - ctk1 - ctk2;
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

// Apply general one-qubit gate via gate matrix
template <typename T>
struct ApplyTwoQubitGateFunctor<GPUDevice, T>
    : BaseTwoQubitGateFunctor<GPUDevice, T> {
  inline void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                            T* state, const T* gate, long tk1, long tk2, int m1,
                            int m2) const override {
    ApplyTwoQubitGateKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, tk1, tk2, m1, m2);
  }

  inline void multicontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                               T* state, const T* gate, long ctk1, long ctk2,
                               long tk1, long tk2, int m1, int m2,
                               int ncontrols, const int* controls, int nqubits,
                               int t1, int t2) const override {
    ApplyTwoQubitGateMultiControlKernel<T>
        <<<numBlocks, blockSize, 0, d.stream()>>>(state, gate, ctk1, ctk2, tk1,
                                                  tk2, m1, m2, ncontrols,
                                                  controls, nqubits, t1, t2);
  }
};

template <typename T>
__global__ void ApplyFsimKernel(T* state, const T* gate, long tk1, long tk2,
                                int m1, int m2) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  auto i = ((long)((long)g >> m1) << (m1 + 1)) + (g & (tk1 - 1));
  i = ((long)((long)i >> m2) << (m2 + 1)) + (i & (tk2 - 1));

  const auto i1 = i + tk1;
  const auto i2 = i + tk2;
  const auto i3 = i1 + tk2;
  const auto buffer = state[i1];
  state[i1] = cadd(cmult(gate[0], state[i1]), cmult(gate[1], state[i2]));
  state[i2] = cadd(cmult(gate[2], buffer), cmult(gate[3], state[i2]));
  state[i3] = cmult(gate[4], state[i3]);
}

template <typename T>
__global__ void ApplyFsimMultiControlKernel(T* state, const T* gate, long ctk1,
                                            long ctk2, long tk1, long tk2,
                                            int m1, int m2, int ncontrols,
                                            const int* controls, int nqubits,
                                            int t1, int t2) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  auto i = g;

  int* qubits = new int[ncontrols + 2];
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
  } else if (q == 1) {
    qubits[ncontrols + 1] = m2;
  }

  for (auto iq = 0; iq < ncontrols + 2; iq++) {
    const auto m = qubits[iq];
    long k = (long)1 << m;
    i = ((long)((long)i >> m) << (m + 1)) + (i & (k - 1)) + k;
  }
  delete[] qubits;

  i = i - ctk1 - ctk2;
  const auto i1 = i + tk1;
  const auto i2 = i + tk2;
  const auto i3 = i1 + tk2;
  const auto buffer = state[i1];
  state[i1] = cadd(cmult(gate[0], state[i1]), cmult(gate[1], state[i2]));
  state[i2] = cadd(cmult(gate[2], buffer), cmult(gate[3], state[i2]));
  state[i3] = cmult(gate[4], state[i3]);
}

template <typename T>
struct ApplyFsimFunctor<GPUDevice, T> : BaseTwoQubitGateFunctor<GPUDevice, T> {
  inline void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                            T* state, const T* gate, long tk1, long tk2, int m1,
                            int m2) const override {
    ApplyFsimKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, tk1, tk2, m1, m2);
  }

  inline void multicontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                               T* state, const T* gate, long ctk1, long ctk2,
                               long tk1, long tk2, int m1, int m2,
                               int ncontrols, const int* controls, int nqubits,
                               int t1, int t2) const override {
    ApplyFsimMultiControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, ctk1, ctk2, tk1, tk2, m1, m2, ncontrols, controls, nqubits,
        t1, t2);
  }
};

template <typename T>
__global__ void ApplySwapKernel(T* state, const T* gate, long tk1, long tk2,
                                int m1, int m2) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  auto i = ((long)((long)g >> m1) << (m1 + 1)) + (g & (tk1 - 1));
  i = ((long)((long)i >> m2) << (m2 + 1)) + (i & (tk2 - 1));

  const auto buffer = state[i + tk1];
  state[i + tk1] = state[i + tk2];
  state[i + tk2] = buffer;
}

template <typename T>
__global__ void ApplySwapMultiControlKernel(T* state, const T* gate, long ctk1,
                                            long ctk2, long tk1, long tk2,
                                            int m1, int m2, int ncontrols,
                                            const int* controls, int nqubits,
                                            int t1, int t2) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  auto i = g;

  int* qubits = new int[ncontrols + 2];
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
  } else if (q == 1) {
    qubits[ncontrols + 1] = m2;
  }

  for (auto iq = 0; iq < ncontrols + 2; iq++) {
    const auto m = qubits[iq];
    long k = (long)1 << m;
    i = ((long)((long)i >> m) << (m + 1)) + (i & (k - 1)) + k;
  }
  delete[] qubits;

  i = i - ctk1 - ctk2;
  const auto buffer = state[i + tk1];
  state[i + tk1] = state[i + tk2];
  state[i + tk2] = buffer;
}

// Apply SWAP gate
template <typename T>
struct ApplySwapFunctor<GPUDevice, T> : BaseTwoQubitGateFunctor<GPUDevice, T> {
  inline void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                            T* state, const T* gate, long tk1, long tk2, int m1,
                            int m2) const override {
    ApplySwapKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, tk1, tk2, m1, m2);
  }

  inline void multicontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                               T* state, const T* gate, long ctk1, long ctk2,
                               long tk1, long tk2, int m1, int m2,
                               int ncontrols, const int* controls, int nqubits,
                               int t1, int t2) const override {
    ApplySwapMultiControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(
        state, gate, ctk1, ctk2, tk1, tk2, m1, m2, ncontrols, controls, nqubits,
        t1, t2);
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
}  // end namespace functor
}  // end namespace tensorflow
#endif  // GOOGLE_CUDA