#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include<cmath>

using namespace tensorflow;

REGISTER_OP("QuantBinary")
    .Attr("marginal: float")
    .Attr("T: {float, double}")
    .Input("to_reshape: T")
    .Output("reshaped: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

template<typename T>
class QuantOp : public OpKernel {
 public:
  explicit QuantOp(OpKernelConstruction* context) : OpKernel(context) {
        // Get Attributes
    OP_REQUIRES_OK(context,
                   context->GetAttr("marginal", &marginal));
    // Check Attributes
    OP_REQUIRES(context, marginal >= 0,
                errors::InvalidArgument("marginal needs to be positive, got ",
                                        marginal));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->flat<T>();

    // set elements >=0 to marginal, or -marginal else
    const int N = input.size();
    for (int i = 0; i < N; i++) {
	    output(i) = input(i)>=0 ? marginal : -marginal;
    }
  }

  private:
    float marginal;
};

REGISTER_KERNEL_BUILDER(
    Name("QuantBinary")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    QuantOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("QuantBinary")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    QuantOp<double>);
