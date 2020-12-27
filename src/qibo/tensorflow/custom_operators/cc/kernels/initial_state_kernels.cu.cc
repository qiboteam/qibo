#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#define DEFAULT_BLOCK_SIZE 1024  // default number of threads

#include "initial_state.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// cuda kernel
template <typename T>
__global__ void SetFirstEntryToZero(T* out) {
  out[0] = T(1, 0);
}

template <typename T>
__global__ void InitializeToZero(T* out) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  out[g] = T(0, 0);
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct InitialStateFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, T* out, int64 shape, int nthreads) {

    int blockSize = DEFAULT_BLOCK_SIZE;
    int numBlocks = (shape + blockSize - 1) / blockSize;
    if (shape < blockSize) {
      numBlocks = 1;
      blockSize = shape;
    }

    InitializeToZero<T><<<numBlocks, blockSize, 0, d.stream()>>>(out);
    SetFirstEntryToZero<T><<<1, 1, 0, d.stream()>>>(out);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct InitialStateFunctor<GPUDevice, complex64>;
template struct InitialStateFunctor<GPUDevice, complex128>;
}  // end namespace functor
}  // end namespace tensorflow
#endif  // GOOGLE_CUDA
