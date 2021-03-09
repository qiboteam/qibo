#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "measurements.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace functor {

// CPU specialization
template <typename Tint, typename Tfloat>
struct MeasureFrequenciesFunctor<CPUDevice, Tint, Tfloat> {
  void operator()(const CPUDevice &d, Tint* frequencies, const Tfloat* probs,
                  int64 nshots, int nqubits, int user_seed = 1234)
  {
    int64 nstates = 1 << nqubits;
    srand(user_seed);
    // Create vector of seeds for each thread
    std::vector<unsigned> thread_seed;
    for (auto i = 0; i < omp_get_max_threads(); i++) {
      thread_seed.push_back(rand());
    }
    // Initial bitstring is the one with the maximum probability
    int64 initial_shot = 0;
    for (int64 i = 0; i < nstates; i++) {
        if (probs[i] > probs[initial_shot]) {
            initial_shot = i;
        }
    }
    #pragma omp parallel
    {
        std::vector<int64> frequencies_private(nstates, 0);
        unsigned seed = thread_seed[omp_get_thread_num()];
        int64 shot = initial_shot;
        #pragma omp for
        for (int64 i = 0; i < nshots; i++) {
          int64 new_shot = (shot + ((int64) rand_r(&seed) % nstates)) % nstates;
          // Accept or reject move
          Tfloat ratio = probs[new_shot] / probs[shot];
          if (ratio > ((Tfloat) rand_r(&seed) / RAND_MAX)) {
            shot = new_shot;
          }
          // Update frequencies
          frequencies_private[shot]++;
        }
        #pragma omp critical
        {
            for(int64 i = 0; i < nstates; i++) {
                frequencies[i] += frequencies_private[i];
            }
        }
    }
  }
};

template <typename Device, typename Tint, typename Tfloat>
class MeasureFrequenciesOp : public OpKernel {
 public:
  explicit MeasureFrequenciesOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("nshots", &nshots_));
    OP_REQUIRES_OK(context, context->GetAttr("nqubits", &nqubits_));
    OP_REQUIRES_OK(context, context->GetAttr("omp_num_threads", &threads_));
    OP_REQUIRES_OK(context, context->GetAttr("seed", &seed_));
    omp_set_num_threads(threads_);
  }

  void Compute(OpKernelContext *context) override {
    // grab the input tensor
    Tensor frequencies = context->input(0);
    const Tensor& probs = context->input(1);

    // call the implementation
    MeasureFrequenciesFunctor<Device, Tint, Tfloat>()
      (context->eigen_device<Device>(), frequencies.flat<Tint>().data(),
       probs.flat<Tfloat>().data(), (int64) nshots_, nqubits_, seed_);
    context->set_output(0, frequencies);
  }

 private:
  int nqubits_;
  int threads_;
  int seed_;
  float nshots_;
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

}  // namespace functor
}  // namespace tensorflow
