#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

REGISTER_OP("InitialState")
      .Attr("T: {complex64, complex128}")
      .Input("in: T")
      .Output("out: T");