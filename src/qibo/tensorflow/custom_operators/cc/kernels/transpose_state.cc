#include "transpose_state.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

using thread::ThreadPool;

template <typename T>
struct TransposeStateFunctor<CPUDevice, T> {
  void operator()(const OpKernelContext* context, const CPUDevice &d,
                  const std::vector<T*> state, T* transposed_state,
                  int nqubits, int ndevices, const int* qubit_order) {
    const int64 nstates = (int64) 1 << nqubits;
    const int64 npiece = (int64) nstates / ndevices;
    std::vector<int64> qubit_exponents(nqubits);
    for (int q = 0; q < nqubits; q++) {
      qubit_exponents[q] = (int64) 1 << (nqubits - qubit_order[nqubits - q - 1] - 1);
    }

    // Set multi-threading
    auto thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    const int ncores = (int) thread_pool->NumThreads();
    int64 nreps;
    if (ncores > 1) {
      nreps = (int64) nstates / ncores;
    }
    else {
      nreps = 1;
    }
    const ThreadPool::SchedulingParams p(
        ThreadPool::SchedulingStrategy::kFixedBlockSize, absl::nullopt,
        nreps);

    auto DoWork = [&](int64 t, int64 w) {
      for (auto g = t; g < w; g++) {
        int64 k = 0;
        for (int q = 0; q < nqubits; q++) {
          if ((g >> q) % 2) k += qubit_exponents[q];
        }
        transposed_state[g] = state[(int64) k / npiece][(int64) k % npiece];
      }
    };
    thread_pool->ParallelFor(nstates, p, DoWork);
  };
};


template <typename Device, typename T>
class TransposeStateOp : public OpKernel {
 public:
  explicit TransposeStateOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("nqubits", &nqubits_));
    OP_REQUIRES_OK(context, context->GetAttr("ndevices", &ndevices_));
    OP_REQUIRES_OK(context, context->GetAttr("qubit_order", &qubit_order_));
  }

  void Compute(OpKernelContext *context) override {
    // grabe the input tensor
    std::vector<T*> state(ndevices_);
    for (int i = 0; i < ndevices_; i++) {
      state[i] = (T*) context->input(i).flat<T>().data();
    }
    Tensor transposed_state = context->input(ndevices_);

    // prevent running on GPU
    OP_REQUIRES(
        context, (std::is_same<Device, CPUDevice>::value == true),
        errors::Unimplemented("TransposeStateOp operator not implemented for GPU."));

    // call the implementation
    TransposeStateFunctor<Device, T>()(context, context->eigen_device<Device>(),
                                       state, transposed_state.flat<T>().data(),
                                       nqubits_, ndevices_, qubit_order_.data());
    context->set_output(0, transposed_state);
  }
  private:
   int nqubits_;
   int ndevices_;
   std::vector<int> qubit_order_;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                             \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("TransposeState").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      TransposeStateOp<CPUDevice, T>);
REGISTER_CPU(complex64);
REGISTER_CPU(complex128);

// Register the GPU kernels.
#define REGISTER_GPU(T)                                             \
  extern template struct TransposeStateFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("TransposeState").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      TransposeStateOp<GPUDevice, T>);
REGISTER_GPU(complex64);
REGISTER_GPU(complex128);
}  // namespace functor
}  // namespace tensorflow
