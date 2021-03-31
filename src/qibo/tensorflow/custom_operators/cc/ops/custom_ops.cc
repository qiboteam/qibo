#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

// Register op that generates initial state
REGISTER_OP("InitialState")
    .Attr("nqubits: int")
    .Attr("dtype: {complex64, complex128}")
    .Attr("is_matrix: bool")
    .Attr("omp_num_threads: int")
    .Output("out: dtype");


// Register op that changes qubit order for multi-GPU
REGISTER_OP("TransposeState")
    .Attr("T: {complex64, complex128}")
    .Attr("ndevices: int")
    .Input("state: ndevices * T")
    .Input("transposed_state: T")
    .Attr("nqubits: int")
    .Attr("qubit_order: list(int)")
    .Attr("omp_num_threads: int")
    .Output("out: T")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);


// Register op that swap state pieces for multi-GPU
REGISTER_OP("SwapPieces")
    .Attr("T: {complex64, complex128}")
    .Input("piece0: T")
    .Input("piece1: T")
    .Attr("target: int")
    .Attr("nqubits: int")
    .Attr("omp_num_threads: int")
    .Output("out0: T")
    .Output("out1: T")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);


// Register op that calculates measurement frequencies
REGISTER_OP("MeasureFrequencies")
    .Attr("Tfloat: {float32, float64}")
    .Attr("Tint: {int32, int64}")
    .Input("frequencies: Tint")
    .Input("probs: Tfloat")
    .Attr("nshots: float")
    .Attr("nqubits: int")
    .Attr("seed: int")
    .Attr("omp_num_threads: int")
    .Output("out: Tint")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);


// Register op that collapses state vector according to measured bit string
REGISTER_OP("CollapseState")            \
    .Attr("T: {complex64, complex128}") \
    .Input("state: T")                  \
    .Input("qubits: int32")             \
    .Input("result: int64")             \
    .Attr("nqubits: int")               \
    .Attr("normalize: bool")            \
    .Attr("omp_num_threads: int")       \
    .Output("out: T")                   \
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);


// Register one-qubit gate op with gate matrix
#define REGISTER_GATE1_OP(NAME)           \
  REGISTER_OP(NAME)                       \
      .Attr("T: {complex64, complex128}") \
      .Input("state: T")                  \
      .Input("gate: T")                   \
      .Input("qubits: int32")             \
      .Attr("nqubits: int")               \
      .Attr("target: int")                \
      .Attr("omp_num_threads: int")       \
      .Output("out: T")                   \
      .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

// Register one-qubit gate op without gate matrix
#define REGISTER_GATE1_NOMATRIX_OP(NAME)  \
  REGISTER_OP(NAME)                       \
      .Attr("T: {complex64, complex128}") \
      .Input("state: T")                  \
      .Input("qubits: int32")             \
      .Attr("nqubits: int")               \
      .Attr("target: int")                \
      .Attr("omp_num_threads: int")       \
      .Output("out: T")                   \
      .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

// Register two-qubit gate op with gate matrix
#define REGISTER_GATE2_OP(NAME)           \
  REGISTER_OP(NAME)                       \
      .Attr("T: {complex64, complex128}") \
      .Input("state: T")                  \
      .Input("gate: T")                   \
      .Input("qubits: int32")             \
      .Attr("nqubits: int")               \
      .Attr("target1: int")               \
      .Attr("target2: int")               \
      .Attr("omp_num_threads: int")       \
      .Output("out: T")                   \
      .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

// Register two-qubit gate op without gate matrix
#define REGISTER_GATE2_NOMATRIX_OP(NAME)  \
  REGISTER_OP(NAME)                       \
      .Attr("T: {complex64, complex128}") \
      .Input("state: T")                  \
      .Input("qubits: int32")             \
      .Attr("nqubits: int")               \
      .Attr("target1: int")               \
      .Attr("target2: int")               \
      .Attr("omp_num_threads: int")       \
      .Output("out: T")                   \
      .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

REGISTER_GATE1_OP("ApplyGate")
REGISTER_GATE1_OP("ApplyZPow")
REGISTER_GATE1_NOMATRIX_OP("ApplyX")
REGISTER_GATE1_NOMATRIX_OP("ApplyY")
REGISTER_GATE1_NOMATRIX_OP("ApplyZ")

REGISTER_GATE2_OP("ApplyTwoQubitGate")
REGISTER_GATE2_OP("ApplyFsim")
REGISTER_GATE2_NOMATRIX_OP("ApplySwap")
