#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include<cmath>

using namespace tensorflow;

REGISTER_OP("QuantHalffp")
    .Attr("T: {float}")
    .Input("to_reshape: T")
    .Output("reshaped: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

template<typename T>
class QuantOp : public OpKernel {
 private:
    static int basetable[512];
    static unsigned int masktable[512];
 public:
  explicit QuantOp(OpKernelConstruction* context) : OpKernel(context) {
    
    // basetable and masktable for half-precision floating point rounding
    

    for(unsigned int i=0; i<256; ++i){
        int e=i-127;
        if(e<-24){
         // Very small numbers map to zero
        basetable[i|0x000]=0x00000000;
        basetable[i|0x100]=0x80000000;
        masktable[i|0x000]=0x00000000;
        masktable[i|0x100]=0x00000000;
        }
        
        else if(e<-14){ // TODO: denorms in FP16 are cut at the end, but normalized
         // Small numbers map to denorms
        basetable[i|0x000]=((e+127)<<23);
        basetable[i|0x100]=((e+127)<<23) | 0x80000000;
        masktable[i|0x000]=(1<<23)-(1<<(-e-14+13));
        masktable[i|0x100]=(1<<23)-(1<<(-e-14+13));
        }
        else if(e<=15){ // TODO: cut normalized numbers
         // Normal numbers just lose precision
        basetable[i|0x000]=((e+127)<<23);
        basetable[i|0x100]=((e+127)<<23) | 0x80000000;
        masktable[i|0x000]=(1<<23)-(1<<(13));
        masktable[i|0x100]=(1<<23)-(1<<(13));
        }
        else if(e<128){ // TODO: FP32 infinity
         // Large numbers map to Infinity
        basetable[i|0x000]=0x7F800000;
        basetable[i|0x100]=0xFF800000;
        masktable[i|0x000]=0x00000000;
        masktable[i|0x100]=0x00000000;
        }
        else{ // TODO: do nothing
         // Infinity and NaN's stay Infinity and NaN's
        basetable[i|0x000]=0x7F800000;
        basetable[i|0x100]=0xFF800000;
        masktable[i|0x000]=(1<<23)-(1<<(13));
        masktable[i|0x100]=(1<<23)-(1<<(13));
        }
    }
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

    // change every element to half-precision floating point
    const int N = input.size();
    unsigned int f, index, h;
    T result;
    for (int i = 0; i < N; i++) {
        f= *(unsigned int*)&input(i); //converter.i;
        index = (f>>23)&0x1ff;
        h=basetable[index]+(f & masktable[index]);
        T result = *(T*)&h;
	    output(i) = result;
    }
  }
};
template<typename T>
int QuantOp<T>::basetable[512];
template<typename T>
unsigned int QuantOp<T>::masktable[512];

REGISTER_KERNEL_BUILDER(
    Name("QuantHalffp")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    QuantOp<float>);
/*
REGISTER_KERNEL_BUILDER(
    Name("QuantHalffp")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    QuantOp<double>);*/
