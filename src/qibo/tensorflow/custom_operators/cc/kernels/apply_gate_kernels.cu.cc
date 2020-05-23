#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "apply_gate.h"

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

// cuda kernel
template <typename T>
__global__ void ApplyGateCudaKernel(const int size, const int k, T* state, const T* gate) {
  auto g = threadIdx.x * 2 * k;
  for (auto i = g; i < g + k; i++) {
    const auto buffer = state[i];
    state[i] = cadd(cmult(gate[0], state[i]), cmult(gate[1], state[i + k]));
    state[i + k] = cadd(cmult(gate[2], buffer), cmult(gate[3], state[i + k]));
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct ApplyGateFunctor<GPUDevice, T> {
  void operator()(const OpKernelContext* context, const GPUDevice& d, T* state,
                  const T* gate, int nqubits, int target) {
    const int64 nstates = std::pow(2, nqubits);
    const int64 k = std::pow(2, nqubits - target - 1);
    int threads = nstates / (2 * k);
    int blocks = (nstates / (2 * k) + threads -1) / threads;
    std::cout << threads << " " << blocks << std::endl;
    ApplyGateCudaKernel<T>
        <<<blocks, threads, 0, d.stream()>>>(nstates, k, state, gate);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct ApplyGateFunctor<GPUDevice, complex64>;
template struct ApplyGateFunctor<GPUDevice, complex128>;
} // end namespace functor
} // end namespace tensorflow
#endif // GOOGLE_CUDA