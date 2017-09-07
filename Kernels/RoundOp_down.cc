#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include<cmath>

using namespace tensorflow;

REGISTER_OP("RoundDown")
    .Attr("fixed_size: int")
    .Attr("fixed_prec: int")
    .Attr("T: {float, double}")
    .Input("to_reshape: T")
    .Output("reshaped: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

template<typename T>
class RoundOp : public OpKernel {
 public:
  explicit RoundOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get Attributes
    OP_REQUIRES_OK(context,
                   context->GetAttr("fixed_size", &fixed_size));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fixed_prec", &fixed_prec));
    // Check Attributes
    OP_REQUIRES(context, fixed_size > 0,
                errors::InvalidArgument("fixed_size needs to be bigger than 0, got ",
                                        fixed_size));
    OP_REQUIRES(context, fixed_prec >= 0 && fixed_prec < fixed_size,
                errors::InvalidArgument("fixed_prec needs to be between 0 and fixed_size, got ",
                                        fixed_prec));

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

    // truncate every element of the tensor to fixed point
    const int N = input.size();
    const T fixed_max_signed = ((T)(1UL<<(fixed_size-1))-1)/(1UL<<fixed_prec);
    const T fixed_min_signed = -(1L<<(fixed_size-fixed_prec-1));

    for (int i = 0; i < N; i++) {

    // rounds towards negative (e.g. -0.001 -> -0.5 )
        T fixed_number = floor(input(i)*(1UL<<fixed_prec)) / (1UL<<fixed_prec);
	    fixed_number = std::max(std::min(fixed_number,fixed_max_signed), fixed_min_signed);
	    output(i) = fixed_number;
    }
  }

 private:
    int fixed_size;
    int fixed_prec;
};

REGISTER_KERNEL_BUILDER(
    Name("RoundDown")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    RoundOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("RoundDown")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    RoundOp<double>);
