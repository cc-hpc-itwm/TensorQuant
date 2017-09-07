#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include<cmath>
#include<iostream>

// Determines if negative numbers should be rounded towards zero when truncated
// #define NEGATIVE_ROUND

using namespace tensorflow;

REGISTER_OP("FixedConv")
    .Attr("fixed_size: int")
    .Attr("fixed_prec: int")
    .Attr("T: {float, double}")
    .Input("input: T")
    .Input("kernel: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle in = c->input(0);
      c->ReplaceDim(c->input(0),3,c->Dim(c->input(1),-1),&in);
      c->set_output(0,in);
      return Status::OK();
    });

template<typename T>
class FixedConvOp : public OpKernel {
 public:
  explicit FixedConvOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get Attributes
    OP_REQUIRES_OK(context,
                   context->GetAttr("fixed_size", &fixed_size));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fixed_prec", &fixed_prec));
    // Check Attributes
    OP_REQUIRES(context, fixed_size > 0,
                errors::InvalidArgument("fixed_size needs to be bigger than 0, got ",
                                        fixed_size));
    OP_REQUIRES(context, fixed_prec > 0 && fixed_prec < fixed_size,
                errors::InvalidArgument("fixed_prec needs to be between 0 and fixed_size, got ",
                                        fixed_prec));

    }

  T fixedPoint(T in) {
    const T fixed_max_signed = ((T)(1UL<<(fixed_size-1))-1)/(1UL<<fixed_prec);
    const T fixed_min_signed = -(1L<<(fixed_size-fixed_prec-1));
    T fixed_number;
#ifdef NEGATIVE_ROUND
    // rounds towards zero (e.g. -0.001 -> 0 )
	fixed_number = floor(std::abs(in)*(1UL<<fixed_prec)) / (1UL<<fixed_prec) *(in<0?(-1):1);
#else
    // rounds towards negative (e.g. -0.001 -> -0.5 )
    fixed_number = floor(in*(1UL<<fixed_prec)) / (1UL<<fixed_prec);
#endif
	fixed_number = std::max(std::min(fixed_number,fixed_max_signed), fixed_min_signed);
    return fixed_number;
    }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& kernel_tensor = context->input(1);
    auto input = input_tensor.flat<T>();
    auto kernel = kernel_tensor.flat_inner_dims<T,2>();
    
    // Create an output tensor
    Tensor* output_tensor = NULL;
    TensorShape out_shape = input_tensor.shape();
    out_shape.set_dim(3,kernel_tensor.dim_size(3));
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                &output_tensor));
    auto output = output_tensor->flat_inner_dims<T,2>();

    // create temporary tensor buffer
    Tensor tmp_tensor;
    OP_REQUIRES_OK( context, context->allocate_temp(input_tensor.dtype(), input_tensor.shape(),
                                                &tmp_tensor) );
    auto tmp= tmp_tensor.flat<T>();
    
    // computation
    const int N = input.size();
    T fixed_number;
    int kernel_size = kernel_tensor.dim_size(0)*kernel_tensor.dim_size(1);
    for (int out_iter = 0; out_iter < output_tensor->dim_size(3); ++out_iter ) {
        for (int i = 0; i < N; i++) {
            fixed_number = input(i) * kernel(i%input_tensor.dim_size(3),out_iter);
            fixed_number= fixedPoint(fixed_number);
            tmp(i)=fixed_number;
        }
        int64 reduction_runs = input_tensor.dim_size(3);
        for (int k = 0; k<(input_tensor.dim_size(1)*input_tensor.dim_size(2)); k++) { 
            T sum = 0;
            for (int i = k*reduction_runs; i< (k+1)*reduction_runs; i++) {
                sum += tmp(i);
            }
            output(k,out_iter)= fixedPoint(sum);
        }
    }
  }

 private:
    int fixed_size;
    int fixed_prec;
};

REGISTER_KERNEL_BUILDER(
    Name("FixedConv")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    FixedConvOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("FixedConv")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    FixedConvOp<double>);
