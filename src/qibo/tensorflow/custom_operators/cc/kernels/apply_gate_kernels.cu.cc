#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "apply_gate.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;


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

template <typename T>
struct BaseOneQubitGateFunctor<GPUDevice, T> {

  virtual void apply_cuda(const GPUDevice& d, T* state, int nqubits, int target, int ncontrols,
                          const int32* controls, const int32* tensor_controls, const T* gate = NULL) const {}

  virtual void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                             T* state, const T* gate, long tk) const {}

  virtual void singlecontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                                 T* state, const T* gate, long tk, long tk_reduced,
                                 int c) const {}

  virtual void multicontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                                 T* state, const T* gate, long tk, long tk_reduced,
                                 int ncontrols, const int * controls, int nqubits) const {}

  void operator()(const OpKernelContext* context, const GPUDevice& d, T* state, int nqubits,
                  int target, int ncontrols, const int32* controls, const int32* tensor_controls,
                  const T* gate = NULL) const {
    const int64 tk = (int64) 1 << (nqubits - target - 1);
    const int64 nstates = (int64) 1 << (nqubits - ncontrols);
    int target_eff = target;
    for (int i = 0; i < ncontrols; i++) {
      if (controls[i] < target) {
        target_eff--;
      }
    }
    const int64 tk_reduced = (int64) 1 << (nqubits - target_eff - ncontrols - 1);

    int64 nreps = nstates;
    if (nreps % (2 * tk_reduced)) {
      nreps = 2 * tk_reduced;
    }
    int blockSize = 1024;
    int numBlocks = (nreps / 2 + blockSize - 1) / blockSize;
    if (nreps / 2 < blockSize)
    {
      numBlocks = 1;
      blockSize = nreps / 2;
    }

    if (ncontrols == 0) {
      nocontrolwork(d, numBlocks, blockSize, state, gate, tk);
    }
    else if (ncontrols == 1) {
      singlecontrolwork(d, numBlocks, blockSize, state, gate, tk,
                        tk_reduced, nqubits - controls[0] - 1);
    }
    else {
      multicontrolwork(d, numBlocks, blockSize, state, gate, tk,
                       tk_reduced, ncontrols, tensor_controls, nqubits);
    }
  }
};


template<typename T>
__global__ void ApplyGateKernel(T* state, const T* gate, long tk) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % tk + 2 * tk * int(index / tk);
  const auto state1 = state[i];
  const auto state2 = state[i + tk];
  const auto buffer = state1;
  state[i]      = cadd(cmult(gate[0], state1), cmult(gate[1], state2));
  state[i + tk] = cadd(cmult(gate[2], buffer), cmult(gate[3], state2));
}

template<typename T>
__global__ void ApplyGateSingleControlKernel(T* state, const T* gate, long tk, long tk_reduced, int c) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % tk_reduced + 2 * tk_reduced * int(index / tk_reduced);
  const long ck = (long) 1 << c;
  const long i1 = ((long) ((long) i >> c) << (c + 1)) + (i & (ck - 1)) + ck;
  const auto state1 = state[i1];
  const auto state2 = state[i1 + tk];
  const auto buffer = state1;
  state[i1]      = cadd(cmult(gate[0], state1), cmult(gate[1], state2));
  state[i1 + tk] = cadd(cmult(gate[2], buffer), cmult(gate[3], state2));
}

template<typename T>
__global__ void ApplyGateMultiControlKernel(T* state, const T* gate, long tk, long tk_reduced,
                                            int ncontrols, const int * controls, int nqubits) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % tk_reduced + 2 * tk_reduced * int(index / tk_reduced);
  auto i1 = i;
  for (auto ic = 0; ic < ncontrols; ic++) {
    const long c = nqubits - controls[ic] - 1;
    const long ck = (long) 1 << c;
    i1 = ((long) ((long) i1 >> c) << (c + 1)) + (i1 & (ck - 1)) + ck;
  }
  const auto state1 = state[i1];
  const auto state2 = state[i1 + tk];
  const auto buffer = state1;
  state[i1]      = cadd(cmult(gate[0], state1), cmult(gate[1], state2));
  state[i1 + tk] = cadd(cmult(gate[2], buffer), cmult(gate[3], state2));
}

// Apply general one-qubit gate via gate matrix
template <typename T>
struct ApplyGateFunctor<GPUDevice, T>: BaseOneQubitGateFunctor<GPUDevice, T> {

  inline void nocontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                            T* state, const T* gate, long tk) const override {
    ApplyGateKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(state, gate, tk);
  }

  inline void singlecontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                                T* state, const T* gate, long tk, long tk_reduced,
                                int c) const override {
    ApplyGateSingleControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(state, gate, tk, tk_reduced, c);
  }

  inline void multicontrolwork(const GPUDevice& d, int numBlocks, int blockSize,
                                 T* state, const T* gate, long tk, long tk_reduced,
                                 int ncontrols, const int * controls, int nqubits) const override {
    ApplyGateMultiControlKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(state, gate, tk, tk_reduced, ncontrols, controls, nqubits);
  }
};


template <typename T>
__global__ void ApplyXWork(const int size, const int k, T* state, const T* gate) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % k + 2 * k * int(index / k);
  const auto buffer = state[i];
  state[i] = state[i + k];
  state[i + k] = buffer;
}

// Apply X gate via swap
template <typename T>
struct ApplyXFunctor<GPUDevice, T>: BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void apply_cuda(const GPUDevice& d, T* state, int nqubits, int target, int ncontrols,
                         const int32* controls, const int32* tensor_controls,
                         const T* gate = NULL) const override {
    const int64 tk = (int64) 1 << (nqubits - target - 1);
    const int64 nstates = (int64) 1 << (nqubits - ncontrols);
    int64 nreps = nstates;
    int blockSize = 1024;
    int numBlocks = (nreps / 2 + blockSize - 1) / blockSize;
    if (nreps / 2 < blockSize)
    {
      numBlocks = 1;
      blockSize = nreps / 2;
    }

    if (ncontrols == 0) {
      ApplyXWork<T>
        <<<numBlocks, blockSize, 0, d.stream()>>>(nreps, tk, state, gate);
    }
  }
};


template <typename T>
__global__ void ApplyYWork(const int size, const int k, T* state, const T* gate) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % k + 2 * k * int(index / k);
  state[i] = cmult(state[i], T(0, 1));
  state[i + k] = cmult(state[i + k], T(0, -1));
  const auto buffer = state[i];
  state[i] = state[i + k];
  state[i + k] = buffer;
}

// Apply Y gate via swap
template <typename T>
struct ApplyYFunctor<GPUDevice, T>: BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void apply_cuda(const GPUDevice& d, T* state, int nqubits, int target, int ncontrols,
                         const int32* controls, const int32* tensor_controls,
                         const T* gate = NULL) const override {
    const int64 tk = (int64) 1 << (nqubits - target - 1);
    const int64 nstates = (int64) 1 << (nqubits - ncontrols);
    int64 nreps = nstates;
    int blockSize = 1024;
    int numBlocks = (nreps / 2 + blockSize - 1) / blockSize;
    if (nreps / 2 < blockSize)
    {
      numBlocks = 1;
      blockSize = nreps / 2;
    }

    if (ncontrols == 0) {
      ApplyYWork<T>
        <<<numBlocks, blockSize, 0, d.stream()>>>(nreps, tk, state, gate);
    }
  }
};


template <typename T>
__global__ void ApplyZWork(const int size, const int k, T* state, const T* gate) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % k + 2 * k * int(index / k);
  state[i + k] = cmult(state[i + k], T(-1));
}

// Apply Z gate
template <typename T>
struct ApplyZFunctor<GPUDevice, T>: BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void apply_cuda(const GPUDevice& d, T* state, int nqubits, int target, int ncontrols,
                         const int32* controls, const int32* tensor_controls,
                         const T* gate = NULL) const override {
    const int64 tk = (int64) 1 << (nqubits - target - 1);
    const int64 nstates = (int64) 1 << (nqubits - ncontrols);
    int64 nreps = nstates;
    int blockSize = 1024;
    int numBlocks = (nreps / 2 + blockSize - 1) / blockSize;
    if (nreps / 2 < blockSize)
    {
      numBlocks = 1;
      blockSize = nreps / 2;
    }

    if (ncontrols == 0) {
      ApplyZWork<T>
        <<<numBlocks, blockSize, 0, d.stream()>>>(nreps, tk, state, gate);
    }
  }
};


template <typename T>
__global__ void ApplyZPowWork(const int size, const int k, T* state, const T* gate) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto i = index % k + 2 * k * int(index / k);
  state[i + k] = cmult(state[i + k], gate[0]);
}

// Apply ZPow gate
template <typename T>
struct ApplyZPowFunctor<GPUDevice, T>: BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void apply_cuda(const GPUDevice& d, T* state, int nqubits, int target, int ncontrols,
                   const int32* controls, const int32* tensor_controls,
                   const T* gate = NULL) const override {
    const int64 tk = (int64) 1 << (nqubits - target - 1);
    const int64 nstates = (int64) 1 << (nqubits - ncontrols);
    int64 nreps = nstates;
    int blockSize = 1024;
    int numBlocks = (nreps / 2 + blockSize - 1) / blockSize;
    if (nreps / 2 < blockSize)
    {
      numBlocks = 1;
      blockSize = nreps / 2;
    }

    if (ncontrols == 0) {
      ApplyZPowWork<T>
        <<<numBlocks, blockSize, 0, d.stream()>>>(nreps, tk, state, gate);
    }
  }
};


template <typename T>
struct BaseTwoQubitGateFunctor<GPUDevice, T> {

  virtual void apply_cuda(const GPUDevice& d, T* state, int nqubits, int target1, int target2,
                          int ncontrols, const int32* controls, const T* gate = NULL) const {}

  void operator()(const OpKernelContext* context, const GPUDevice& d, T* state,
                  int nqubits,
                  int target1,
                  int target2,
                  int ncontrols,
                  const int32* controls,
                  const T* gate = NULL) const
                  {
                    std::cout << "BaseTwoQubitGateFunctor" << std::endl;
                    apply_cuda(d, state, nqubits, target1, target2, ncontrols, controls, gate);
                  };
};


// Apply general one-qubit gate via gate matrix
template <typename T>
struct ApplyTwoQubitGateFunctor<GPUDevice, T>: BaseTwoQubitGateFunctor<GPUDevice, T> {
  inline void apply_cuda(const GPUDevice& d, T* state, int nqubits, int target1, int target2,
                         int ncontrols, const int32* controls, const T* gate = NULL) const override
                         {
                          std::cout << "ApplyTwoQubitGateFunctor" << std::endl;
                         }
};


template <typename T>
struct ApplyFsimFunctor<GPUDevice, T>: BaseTwoQubitGateFunctor<GPUDevice, T> {
  inline void apply_cuda(const GPUDevice& d, T* state, int nqubits, int target1, int target2,
                         int ncontrols, const int32* controls, const T* gate = NULL) const override
                         {
                          std::cout << "ApplyFsimFunctor" << std::endl;
                         }
};


// Apply SWAP gate
template <typename T>
struct ApplySwapFunctor<GPUDevice, T>: BaseTwoQubitGateFunctor<GPUDevice, T> {
  inline void apply_cuda(const GPUDevice& d, T* state, int nqubits, int target1, int target2,
                         int ncontrols, const int32* controls, const T* gate = NULL) const override
                         {
                          std::cout << "ApplySwapFunctor" << std::endl;
                         }
};

// Explicitly instantiate functors for the types of OpKernels registered.
#define REGISTER_TEMPLATE(FUNCTOR)                 \
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
} // end namespace functor
} // end namespace tensorflow
#endif // GOOGLE_CUDA