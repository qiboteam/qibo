#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("InitialState")
    .Attr("T: {complex64, complex128}")
    .Input("in: T")
    .Output("out: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

#define REGISTER_GATE_OP(NAME)                                                \
  REGISTER_OP(NAME)                                                           \
      .Attr("T: {complex64, complex128}")                                     \
      .Input("state: T")                                                      \
      .Input("gate: T")                                                       \
      .Input("controls: int32")                                               \
      .Attr("nqubits: int")                                                   \
      .Attr("target: int")                                                    \
      .Output("out: T")                                                       \
      .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {    \
        c->set_output(0, c->input(0));                                        \
        return Status::OK();                                                  \
      });

#define REGISTER_NOGATE_OP(NAME)                                              \
  REGISTER_OP(NAME)                                                           \
      .Attr("T: {complex64, complex128}")                                     \
      .Input("state: T")                                                      \
      .Input("controls: int32")                                               \
      .Attr("nqubits: int")                                                   \
      .Attr("target: int")                                                    \
      .Output("out: T")                                                       \
      .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {    \
        c->set_output(0, c->input(0));                                        \
        return Status::OK();                                                  \
      });

REGISTER_GATE_OP("ApplyGate")
REGISTER_NOGATE_OP("ApplyX")
REGISTER_NOGATE_OP("ApplyY")
REGISTER_NOGATE_OP("ApplyZ")
REGISTER_GATE_OP("ApplyZPow")

REGISTER_OP("ApplySwap")
    .Attr("T: {complex64, complex128}")
    .Input("state: T")
    .Input("controls: int32")
    .Attr("nqubits: int")
    .Attr("target1: int")
    .Attr("target2: int")
    .Output("out: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
