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

REGISTER_OP("ApplyGate")
    .Attr("T: {complex64, complex128}")
    .Input("state: T")
    .Input("gate: T")
    .Input("controls: int32")
    .Attr("nqubits: int")
    .Attr("target: int")
    .Output("out: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
