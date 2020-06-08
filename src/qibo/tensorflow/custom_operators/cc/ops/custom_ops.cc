#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

// Register op that generates initial state
REGISTER_OP("InitialState")
    .Attr("T: {complex64, complex128}")
    .Input("in: T")
    .Output("out: T")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);


// Register op that splits state to pieces for multi-GPU
REGISTER_OP("SplitState")
    .Attr("T: {complex64, complex128}")
    .Input("state: T")
    .Input("pieces: T")
    .Attr("nqubits: int")
    .Attr("global_qubits: list(int)")
    .Output("out: T")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);


// Register one-qubit gate op with gate matrix
#define REGISTER_GATE1_OP(NAME)           \
  REGISTER_OP(NAME)                       \
      .Attr("T: {complex64, complex128}") \
      .Input("state: T")                  \
      .Input("gate: T")                   \
      .Input("tensor_controls: int32")    \
      .Attr("controls: list(int)")        \
      .Attr("nqubits: int")               \
      .Attr("target: int")                \
      .Output("out: T")                   \
      .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

// Register one-qubit gate op without gate matrix
#define REGISTER_GATE1_NOMATRIX_OP(NAME)  \
  REGISTER_OP(NAME)                       \
      .Attr("T: {complex64, complex128}") \
      .Input("state: T")                  \
      .Input("tensor_controls: int32")    \
      .Attr("controls: list(int)")        \
      .Attr("nqubits: int")               \
      .Attr("target: int")                \
      .Output("out: T")                   \
      .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

// Register two-qubit gate op with gate matrix
#define REGISTER_GATE2_OP(NAME)           \
  REGISTER_OP(NAME)                       \
      .Attr("T: {complex64, complex128}") \
      .Input("state: T")                  \
      .Input("gate: T")                   \
      .Input("tensor_controls: int32")    \
      .Attr("controls: list(int)")        \
      .Attr("nqubits: int")               \
      .Attr("target1: int")               \
      .Attr("target2: int")               \
      .Output("out: T")                   \
      .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

// Register two-qubit gate op without gate matrix
#define REGISTER_GATE2_NOMATRIX_OP(NAME)  \
  REGISTER_OP(NAME)                       \
      .Attr("T: {complex64, complex128}") \
      .Input("state: T")                  \
      .Input("tensor_controls: int32")    \
      .Attr("controls: list(int)")        \
      .Attr("nqubits: int")               \
      .Attr("target1: int")               \
      .Attr("target2: int")               \
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
