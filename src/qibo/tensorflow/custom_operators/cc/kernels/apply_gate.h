/************************************************
 * Functors for gate kernels that apply gates.
 *
 * The following kernels are defined:
 *    - @struct ApplyGateFunctor
 *        Applies gates using standard matrix multiplication.
 *    - @struct ApplyXFunctor
 *        Applies the Pauli X gate by swapping the |...0...> subspace
 *        amplitudes with the |...1...> subspace.
 *    - @struct ApplyYFunctor
 *        Applies the Pauli Y gate by amplitude swapping and multiplying each
 *        piece with the \f$\pm i\f$ phase.
 *    - @struct ApplyZFunctor
 *        Applies the Pauli Z gate by multiplying |...1...> with -1.
 *    - @struct ApplyZPowFunctor
 *        Applies the ZPow gate by multiplying |...1...> with \f$e^{if\theta}\f$.
 *        The phase is passed using the first component of the \p gate parameter.
 *    - @struct ApplySwapFunctor
 *        Applies the SWAP gate by swapping the |...0...1...> subspace with
 *        the |...1...0...> subspace.
 *
 * All functors (except @struct ApplySwapFunctor) inherit the
 * @struct BaseApplyGateFunctor which defines BaseApplyGateFunctor::operator.
 * Each specialized functor definces the corresponding
 * BaseApplyGateFunctor::apply method that handles how the gate acts on states.
 *
 * Gates applied by kernels defined here support a single target qubit but can
 * also be controlled to an arbitrary number of qubits. When a gate is controlled
 * it is applied only to the components that belong to |11...1> subspace of
 * control. The indices of elements that belong to this subpsace are generated
 * as follows: For \f$n_q\f$ total qubits and \f$n_c\f$ control qubits we generate
 * the indices of the target qubit 0- and 1-subspace assuming we have \f$n_q-n_c\f$
 * qubits. These are numbers whose binary representation has length \f$n_q-n_c\f$.
 * Each of these numbers is then transformed to one with binary representation
 * of length \f$n_q\f$ by adding ones in the positions of control qubits. This
 * is done using C++ binary operators.
 ***********************************************/
#ifndef KERNEL_APPLY_GATE_H_
#define KERNEL_APPLY_GATE_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct BaseOneQubitGateFunctor {

  virtual void apply(T& state1, T& state2, const T* gate = NULL) const;

  virtual void nocontrolwork(const Device& d, int numBlocks, int blockSize,
                             T* state, const T* gate, long tk) const;

  virtual void singlecontrolwork(const Device& d, int numBlocks, int blockSize,
                                 T* state, const T* gate, long tk, long tk_reduced,
                                 int c) const;

  virtual void multicontrolwork(const Device& d, int numBlocks, int blockSize,
                                 T* state, const T* gate, long tk, long tk_reduced,
                                 int ncontrols, const int * controls, int nqubits) const;

  void operator()(const OpKernelContext* context,
                  const Device& d,
                  T* state,       //!< Total state vector.
                  int nqubits,    //!< Total number of qubits in the state.
                  int target,     //!< Target qubit id.
                  int ncontrols,  //!< Number of qubits that the gate is controlled on.
                  const int32* controls,  //!< List of control qubits ids sorted in decreasing order.
                  const int32* tensor_controls, //!< List of control qubits ids sorted in decreasing order and stored on Device.
                  const T* gate = NULL    //!< Gate matrix (used only by)
                  ) const;
};

template <typename Device, typename T>
struct ApplyGateFunctor: BaseOneQubitGateFunctor<Device, T> {};

template <typename Device, typename T>
struct ApplyXFunctor: BaseOneQubitGateFunctor<Device, T> {};

template <typename Device, typename T>
struct ApplyYFunctor: BaseOneQubitGateFunctor<Device, T> {};

template <typename Device, typename T>
struct ApplyZFunctor: BaseOneQubitGateFunctor<Device, T> {};

template <typename Device, typename T>
struct ApplyZPowFunctor: BaseOneQubitGateFunctor<Device, T> {};

template <typename Device, typename T>
struct BaseTwoQubitGateFunctor {

  virtual void apply(T* state, int64 i, int64 tk1, int64 tk2, const T* gate = NULL) const;

  virtual void apply_cuda(const Device& d, T* state, int nqubits, int target1, int target2,
                          int ncontrols, const int32* controls, const T* gate = NULL) const;

  void operator()(const OpKernelContext* context,
                  const Device& d,
                  T* state,
                  int nqubits,
                  int target1,
                  int target2,
                  int ncontrols,
                  const int32* controls,
                  const T* gate = NULL) const;
};

template <typename Device, typename T>
struct ApplyTwoQubitGateFunctor: BaseTwoQubitGateFunctor<Device, T> {};

template <typename Device, typename T>
struct ApplyFsimFunctor: BaseTwoQubitGateFunctor<Device, T> {};

template <typename Device, typename T>
struct ApplySwapFunctor: BaseTwoQubitGateFunctor<Device, T> {};

}  // namespace functor

}  // namespace tensorflow

#endif  // KERNEL_APPLY_GATE_H_
