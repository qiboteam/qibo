#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

REGISTER_OP("InitialState")
    .Attr("T: {complex64, complex128}")
    .Input("in: T")
    .Output("out: T")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

#define REGISTER_GATE_OP(NAME)                                                \
  REGISTER_OP(NAME)                                                           \
      .Attr("T: {complex64, complex128}")                                     \
      .Input("state: T")                                                      \
      .Input("gate: T")                                                       \
      .Input("controls: int32")                                               \
      .Attr("nqubits: int")                                                   \
      .Attr("target: int")                                                    \
      .Output("out: T")                                                       \
      .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

#define REGISTER_GATE_NOMATRIX_OP(NAME)                                       \
  REGISTER_OP(NAME)                                                           \
      .Attr("T: {complex64, complex128}")                                     \
      .Input("state: T")                                                      \
      .Input("controls: int32")                                               \
      .Attr("nqubits: int")                                                   \
      .Attr("target: int")                                                    \
      .Output("out: T")                                                       \
      .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

REGISTER_GATE_OP("ApplyGate")
REGISTER_GATE_OP("ApplyZPow")
REGISTER_GATE_NOMATRIX_OP("ApplyX")
REGISTER_GATE_NOMATRIX_OP("ApplyY")
REGISTER_GATE_NOMATRIX_OP("ApplyZ")

REGISTER_OP("ApplySwap")
    .Attr("T: {complex64, complex128}")
    .Input("state: T")
    .Input("controls: int32")
    .Attr("nqubits: int")
    .Attr("target1: int")
    .Attr("target2: int")
    .Output("out: T")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);
