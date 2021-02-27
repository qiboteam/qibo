#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "measurements.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// CPU specialization
template <typename Tint, typename Tfloat>
struct MeasureFrequenciesFunctor<CPUDevice, Tint, Tfloat> {
  void operator()(const CPUDevice &d, Tint* frequencies, const Tfloat* probs,
                  Tint nshots, int nqubits, int user_seed = 1234)
  {
    int64 nstates = 1 << nqubits;
    srand(user_seed);
    unsigned thread_seed[omp_get_max_threads()];
    for (auto i = 0; i < omp_get_max_threads(); i++) {
      thread_seed[i] = rand();
    }
    // Initial bitstring is the one with the maximum probability
    const int64 size_ratio = ((int64) sizeof(probs) / sizeof(Tfloat));
    int64 initial_shot = std::distance(probs, std::max_element(probs, probs + size_ratio));
    #pragma omp parallel
    {
        std::unordered_map<int64, int64> frequencies_private;
        unsigned seed = thread_seed[omp_get_thread_num()];
        int64 shot = initial_shot;
        #pragma omp for
        for (int64 i = 0; i < nshots; i++) {
          // Generate random index to flip its bit
          int flip_index = ((int) rand_r(&seed) % nqubits);
          // Flip the corresponding bit
          int current_value = ((int64) shot >> flip_index) % 2;
          int64 new_shot = shot + ((int64)(1 - 2 * current_value)) * ((int64) 1 << flip_index);
          // Accept or reject move
          Tfloat ratio = probs[new_shot] / probs[shot];
          if (ratio > ((Tfloat) rand_r(&seed) / RAND_MAX)) {
            shot = new_shot;
          }
          // Update frequencies
          if (frequencies_private.find(shot) == frequencies_private.end()) {
              frequencies_private[shot] = 1;
          } else {
              frequencies_private[shot]++;
          }
        }
        #pragma omp critical
        {
            for(const auto& entry : frequencies_private) {
                frequencies[entry.first] += entry.second;
            }
        }
    }
  }
};

template <typename Device, typename Tint, typename Tfloat>
class MeasureFrequenciesOp : public OpKernel {
 public:
  explicit MeasureFrequenciesOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("nqubits", &nqubits_));
    OP_REQUIRES_OK(context, context->GetAttr("omp_num_threads", &threads_));
    OP_REQUIRES_OK(context, context->GetAttr("seed", &seed_));
    omp_set_num_threads(threads_);
  }

  void Compute(OpKernelContext *context) override {
    // grab the input tensor
    Tensor frequencies = context->input(0);
    const Tensor& cumprobs = context->input(1);
    const Tensor& nshots = context->input(2);

    // call the implementation
    MeasureFrequenciesFunctor<Device, Tint, Tfloat>()
      (context->eigen_device<Device>(), frequencies.flat<Tint>().data(),
       cumprobs.flat<Tfloat>().data(), nshots.flat<Tint>().data()[0],
       nqubits_, seed_);
    context->set_output(0, frequencies);
  }

 private:
  int nqubits_;
  int threads_;
  int seed_;
};

// Register the CPU kernels.
#define REGISTER_CPU(Tint, Tfloat)                                     \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("MeasureFrequencies").Device(DEVICE_CPU)                    \
      .TypeConstraint<Tint>("Tint").TypeConstraint<Tfloat>("Tfloat"),  \
      MeasureFrequenciesOp<CPUDevice, Tint, Tfloat>);
REGISTER_CPU(int32, float);
REGISTER_CPU(int64, float);
REGISTER_CPU(int32, double);
REGISTER_CPU(int64, double);

//#ifdef GOOGLE_CUDA
// Register the GPU kernels.
//#define REGISTER_GPU(T)                                               \
  extern template struct InitialStateFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("InitialState").Device(DEVICE_GPU).TypeConstraint<T>("dtype"), \
      InitialStateOp<GPUDevice, T>);
//REGISTER_GPU(complex64);
//REGISTER_GPU(complex128);
//#endif
}  // namespace functor
}  // namespace tensorflow
