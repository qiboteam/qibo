#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "apply_gate.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;


template <typename T>
struct BaseOneQubitGateFunctor<GPUDevice, T> {

  virtual void apply_cuda(const GPUDevice& d, T* state, int nqubits, int target, int ncontrols,
                          const int32* controls, const T* gate = NULL) const {}

  void operator()(const OpKernelContext* context, const GPUDevice& d, T* state, int nqubits,
                  int target, int ncontrols, const int32* controls, const T* gate = NULL) const {
                    apply_cuda(d, state, nqubits, target, ncontrols, controls, gate);
                  };
};


// Apply general one-qubit gate via gate matrix
template <typename T>
struct ApplyGateFunctor<GPUDevice, T>: BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void apply_cuda(const GPUDevice& d, T* state, int nqubits, int target, int ncontrols,
                         const int32* controls, const T* gate = NULL) const override
                         {
                           std::cout << "ApplyGateFunctor" << std::endl;
                         }
};


// Apply X gate via swap
template <typename T>
struct ApplyXFunctor<GPUDevice, T>: BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void apply_cuda(const GPUDevice& d, T* state, int nqubits, int target, int ncontrols,
                         const int32* controls, const T* gate = NULL) const override
                         {
                           std::cout << "ApplyXFunctor" << std::endl;
                         }
};


// Apply Y gate via swap
template <typename T>
struct ApplyYFunctor<GPUDevice, T>: BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void apply_cuda(const GPUDevice& d, T* state, int nqubits, int target, int ncontrols,
                         const int32* controls, const T* gate = NULL) const override
                         {
                           std::cout << "ApplyYFunctor" << std::endl;
                         }
};


// Apply Z gate
template <typename T>
struct ApplyZFunctor<GPUDevice, T>: BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void apply_cuda(const GPUDevice& d, T* state, int nqubits, int target, int ncontrols,
                         const int32* controls, const T* gate = NULL) const override
                        {
                          std::cout << "ApplyZFunctor" << std::endl;
                        }
};


// Apply ZPow gate
template <typename T>
struct ApplyZPowFunctor<GPUDevice, T>: BaseOneQubitGateFunctor<GPUDevice, T> {
  inline void apply_cuda(const GPUDevice& d, T* state, int nqubits, int target, int ncontrols,
                   const int32* controls, const T* gate = NULL) const override
                   {
                     std::cout << "ApplyZPowFunctor" << std::endl;
                   }
};


template <typename T>
struct BaseTwoQubitGateFunctor<GPUDevice, T> {

  virtual void apply_cuda(const GPUDevice& d, T* state, int nqubits, int target1, int target2,
                          int ncontrols, const int32* controls, const T* gate = NULL) const {};

  void operator()(const OpKernelContext* context, const GPUDevice& d, T* state,
                  int nqubits,
                  int target1,
                  int target2,
                  int ncontrols,
                  const int32* controls,
                  const T* gate = NULL) const
                  {
                    apply_cuda(d, state, nqubits, target1, target2, ncontrols, controls, gate);
                  };
};


// Apply general one-qubit gate via gate matrix
template <typename T>
struct ApplyTwoQubitGateFunctor<GPUDevice, T>: BaseTwoQubitGateFunctor<GPUDevice, T> {
  inline void apply_cuda(const GPUDevice& d, T* state, int nqubits, int target1, int target2,
                         int ncontrols, const int32* controls, const T* gate = NULL) const override
                         {
                         }
};


template <typename T>
struct ApplyFsimFunctor<GPUDevice, T>: BaseTwoQubitGateFunctor<GPUDevice, T> {
  inline void apply_cuda(const GPUDevice& d, T* state, int nqubits, int target1, int target2,
                         int ncontrols, const int32* controls, const T* gate = NULL) const override
                         {
                         }
};


// Apply SWAP gate
template <typename T>
struct ApplySwapFunctor<GPUDevice, T>: BaseTwoQubitGateFunctor<GPUDevice, T> {
  inline void apply_cuda(const GPUDevice& d, T* state, int nqubits, int target1, int target2,
                         int ncontrols, const int32* controls, const T* gate = NULL) const override
                         {
                         }
};

// Explicitly instantiate functors for the types of OpKernels registered.
#define REGISTER_TEMPLATE(FUNCTOR)                 \
    template struct FUNCTOR<GPUDevice, complex64>; \
    template struct FUNCTOR<GPUDevice, complex128>;

REGISTER_TEMPLATE(BaseOneQubitGateFunctor);
REGISTER_TEMPLATE(BaseTwoQubitGateFunctor);
REGISTER_TEMPLATE(ApplyGateFunctor);
REGISTER_TEMPLATE(ApplyXFunctor);
REGISTER_TEMPLATE(ApplyYFunctor);
REGISTER_TEMPLATE(ApplyZFunctor);
REGISTER_TEMPLATE(ApplyZPowFunctor);
REGISTER_TEMPLATE(ApplyTwoQubitGateFunctor);
REGISTER_TEMPLATE(ApplyFsimFunctor);
REGISTER_TEMPLATE(ApplySwapFunctor);
} // end namespace functor
} // end namespace tensorflow
#endif // GOOGLE_CUDA