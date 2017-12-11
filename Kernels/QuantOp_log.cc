#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include<cmath>

using namespace tensorflow;

#define EPS (0.000000001)

REGISTER_OP("QuantLog")
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

    // change every element to the nearest value of the form 2^i.
    const int N = input.size();
    for (int i = 0; i < N; i++) {
        int exp=std::ilogb(std::abs(input(i))+EPS);
        float number = exp>0 ? 1<<exp : 1.0/(1<<std::abs(exp));
	    output(i) = std::signbit(input(i)) ? -1*number : number;
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("QuantLog")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    QuantOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("QuantLog")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    QuantOp<double>);
