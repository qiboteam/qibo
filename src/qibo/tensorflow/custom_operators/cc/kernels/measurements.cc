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
  void operator()(const CPUDevice &d, Tint* frequencies, const Tfloat* cumprobs,
                  Tint nshots, int nqubits)
  {
    int64 nstates = 1 << nqubits;
    #pragma omp parallel shared(cumprobs)
    {
        std::unordered_map<int64, int64> frequencies_private;
        unsigned seed = 12345 + 17 * omp_get_thread_num();
        #pragma omp for
        for (auto i = 0; i < nshots; i++) {
          Tfloat random_number = ((Tfloat) rand_r(&seed) / RAND_MAX);
          for (auto j = 0; j < nstates; j++) {
            if (random_number < cumprobs[j]) {
                if (frequencies_private.find(j) == frequencies_private.end()) {
                    frequencies_private[j] = 1;
                } else {
                    frequencies_private[j]++;
                }
                break;
            }
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
       cumprobs.flat<Tfloat>().data(), nshots.flat<Tint>().data()[0], nqubits_);
    context->set_output(0, frequencies);
  }

 private:
  int nqubits_;
  int threads_;
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
