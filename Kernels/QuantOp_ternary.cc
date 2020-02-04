#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include <iostream>

using namespace tensorflow;

REGISTER_OP("QuantTernary")
    .Attr("marginal: float")
    .Attr("auto_threshold: bool=true")    
    .Attr("threshold: float=0.5")
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
    OP_REQUIRES_OK(context,
                   context->GetAttr("auto_threshold", &auto_threshold));
    OP_REQUIRES_OK(context,
                   context->GetAttr("threshold", &threshold));
    // Check Attributes
    OP_REQUIRES(context, marginal >= 0,
                errors::InvalidArgument("marginal needs to be positive, got ",
                                        marginal));
    OP_REQUIRES(context, threshold >= 0,
                errors::InvalidArgument("threshold needs to be positive, got ",
                                        threshold));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();
    //Eigen::Tensor<T,1> input = input_tensor.flat<T>();
    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->flat<T>();

    // set elements >threshold to marginal, <threshold to -marginal and else to 0
    // auto threshold based on https://arxiv.org/pdf/1605.04711.pdf
    const int N = input.size();
    if(auto_threshold) {
        const auto abs_input = Eigen::Map< const Eigen::Array<
                   T, Eigen::Dynamic, 1> >
                   ( input.data(), input.size() );

        float abs_sum = abs_input.abs().sum();
        
        threshold = 0.7 * abs_sum / N;
        //std::cout<< threshold << std::endl;
    }
    for (int i = 0; i < N; i++) {
        if(input(i)>threshold) {
            output(i)=marginal;
        }
        else if(input(i)<-threshold) {
            output(i)=-marginal;
        }        
        else {
            output(i)=0;
        }
    }
  }

  private:
    float marginal;
    bool auto_threshold;
    float threshold;
};

REGISTER_KERNEL_BUILDER(
    Name("QuantTernary")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    QuantOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("QuantTernary")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    QuantOp<double>);
