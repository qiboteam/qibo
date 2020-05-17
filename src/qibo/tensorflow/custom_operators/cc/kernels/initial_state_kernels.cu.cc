#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "initial_state.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// cuda kernel
template <typename T>
__global__ void InitialStateCudaKernel(T* inout) {
    out[0] = T(1, 0);
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct InitialStateFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, T* inout) {
    InitialStateCudaKernel<T>
        <<<1, 1, 0, d.stream()>>>(inout);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct InitialStateFunctor<GPUDevice, complex64>;
template struct InitialStateFunctor<GPUDevice, complex128>;
} // end namespace functor
} // end namespace tensorflow
#endif // GOOGLE_CUDA