#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include<cmath>

using namespace tensorflow;

REGISTER_OP("QuantSparse")
    .Attr("threshold: float")
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
                   context->GetAttr("threshold", &threshold));
    // Check Attributes
    OP_REQUIRES(context, threshold >= 0,
                errors::InvalidArgument("threshold needs to be positive, got ",
                                        threshold));
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

    // every element whose magnitude is below the threshold is set to 0.
    const int N = input.size();
    for (int i = 0; i < N; i++) {
	    output(i) = std::abs(input(i))>threshold ? input(i) : 0;
    }
  }

  private:
    float threshold;
};

REGISTER_KERNEL_BUILDER(
    Name("QuantSparse")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    QuantOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("QuantSparse")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    QuantOp<double>);
