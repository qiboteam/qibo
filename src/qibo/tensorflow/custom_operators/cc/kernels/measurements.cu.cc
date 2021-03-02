#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#define DEFAULT_BLOCK_SIZE 1024  // default number of threads

#include "measurements.h"
#include "tensorflow/core/framework/op_kernel.h"

// Cuda includes
#include <curand.h>
#include <curand_kernel.h>

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// cuda kernel
template <typename Tint, typename Tfloat>
__global__ void Measure(Tint* frequencies, const Tfloat* probs,
                        int64 nshots, int nqubits, int user_seed)
{
  int64 block_id = blockIdx.x;
  int64 thread_id = threadIdx.x;
  int64 block_size = blockDim.x;

  int64 index = block_id*block_size + thread_id;
  int64 grid_dim = gridDim.x;
  int64 stride = block_size * grid_dim;

  // seed-sequence-offset
  curandState_t state;
  curand_init(index + user_seed, 0, 0, &state);

  // Initial bitstring is the one with the maximum probability
  int64 nstates = 1 << nqubits;
  int64 initial_shot = 0;
  for (int64 i = 0; i < nstates; i++) {
      if (probs[i] > probs[initial_shot]) {
          initial_shot = i;
      }
  }

  int64 shot = initial_shot;
  for (auto j = index; j < nshots; j+= stride) {
    // Generate random index to flip its bit
    int flip_index = ((int) curand(&state) % nqubits);
    // Flip the corresponding bit
    int current_value = ((int64) shot >> flip_index) % 2;
    int64 new_shot = shot + ((int64)(1 - 2 * current_value)) * ((int64) 1 << flip_index);
    // Accept or reject move
    Tfloat ratio = probs[new_shot] / probs[shot];
    if (ratio > ((Tfloat) curand(&state) / RAND_MAX)) {
      shot = new_shot;
    }
    // Update frequencies

  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename Tint, typename Tfloat>
struct MeasureFrequenciesFunctor<GPUDevice, Tint, Tfloat> {
  void operator()(const GPUDevice &d, Tint* frequencies, const Tfloat* probs,
                  int64 nshots, int nqubits, int user_seed = 1234)
  {
    int64 blockSize = DEFAULT_BLOCK_SIZE;
    int64 numBlocks = (nshots + blockSize - 1) / blockSize;
    if (nshots < blockSize) {
      numBlocks = 1;
      blockSize = nshots;
    }

    Measure<Tint, Tfloat><<<numBlocks, blockSize, 0, d.stream()>>>(frequencies, probs, nshots, nqubits, user_seed);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct MeasureFrequenciesFunctor<GPUDevice, int32, float>;
template struct MeasureFrequenciesFunctor<GPUDevice, int64, float>;
template struct MeasureFrequenciesFunctor<GPUDevice, int32, double>;
template struct MeasureFrequenciesFunctor<GPUDevice, int64, double>;
}  // end namespace functor
}  // end namespace tensorflow
#endif  // GOOGLE_CUDA
