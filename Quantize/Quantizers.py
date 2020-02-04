from TensorQuant.Quantize import FixedPoint
from TensorQuant.Quantize import QuantKernelWrapper as Wrapped
import tensorflow as tf

class Quantizer_if():
    """Interface for quantizer classes"""
    def __str__(self):
        return self.__class__.__name__
    def quantize(self,tensor):
        raise NotImplementedError
    def __call__(self, tensor):
        return self.quantize(tensor)

###############################
### Fixed Point
###############################

class FixedPointQuantizer_zero(Quantizer_if):
    """Fixed point quantization with fixed_size bits and fixed_prec fractional bits.
       Uses c-kernel for quantization.
    """
    def __init__(self, fixed_size, fixed_prec):
        self.fixed_size=fixed_size
        self.fixed_prec=fixed_prec
        self.fixed_max_signed =  (1<<(fixed_size-1)-1)/(1<<fixed_prec)
        self.fixed_min_signed = -(1<<(fixed_size-fixed_prec-1))
    def C_quantize(self,tensor):
        return FixedPoint.round_zero(tensor,self.fixed_size,self.fixed_prec)
    def quantize(self,tensor):
        @tf.custom_gradient
        def op(tensor):
            def grad(dy):
                return dy
            out =  tf.math.floor( tf.math.abs(tensor)*(1<<self.fixed_prec) ) / (1<<self.fixed_prec) * tf.math.sign(tensor)
            # handle overflow (saturate number towards maximum or minimum)
            out = tf.math.maximum( tf.math.minimum( out, self.fixed_max_signed ), self.fixed_min_signed)
            # tag output
            out = tf.identity(out, name=str(self)+"_output")
            return out, grad
        return op(tensor)

class FixedPointQuantizer_down(Quantizer_if):
    """Fixed point quantization with fixed_size bits and fixed_prec fractional bits.
    """
    def __init__(self, fixed_size, fixed_prec):
        self.fixed_size=fixed_size
        self.fixed_prec=fixed_prec
        self.fixed_max_signed =  (1<<(fixed_size-1)-1)/(1<<fixed_prec)
        self.fixed_min_signed = -(1<<(fixed_size-fixed_prec-1))

    def C_quantize(self,tensor):
        return FixedPoint.round_down(tensor,self.fixed_size,self.fixed_prec)

    def quantize(self,tensor):
        @tf.custom_gradient
        def op(tensor):
            def grad(dy):
                return dy
            out =  tf.math.floor( tensor*(1<<self.fixed_prec) ) / (1<<self.fixed_prec)
            # handle overflow (saturate number towards maximum or minimum)
            out = tf.math.maximum( tf.math.minimum( out, self.fixed_max_signed ), self.fixed_min_signed)
            # tag output
            out = tf.identity(out, name=str(self)+"_output")
            return out, grad
        return op(tensor)


class FixedPointQuantizer_nearest(Quantizer_if):
    """Fixed point quantization with fixed_size bits and fixed_prec fractional bits.
    """
    def __init__(self, fixed_size, fixed_prec):
        self.fixed_size=fixed_size
        self.fixed_prec=fixed_prec
        self.fixed_max_signed =  (1<<(fixed_size-1)-1)/(1<<fixed_prec)
        self.fixed_min_signed = -(1<<(fixed_size-fixed_prec-1))

    def C_quantize(self,tensor):
        return FixedPoint.round_nearest(tensor,self.fixed_size,self.fixed_prec)

    def quantize(self,tensor):
        @tf.custom_gradient
        def op(tensor):
            def grad(dy):
                return dy
            out =  tf.math.floor( tf.math.abs(tensor)*(1<<self.fixed_prec)+0.5) /(1<<self.fixed_prec) * tf.math.sign(tensor)
            # handle overflow (saturate number towards maximum or minimum)
            out = tf.math.maximum( tf.math.minimum( out, self.fixed_max_signed ), self.fixed_min_signed)
            # tag output
            out = tf.identity(out, name=str(self)+"_output")
            return out, grad
        return op(tensor)


class FixedPointQuantizer_stochastic(Quantizer_if):
    """Fixed point quantization with fixed_size bits and fixed_prec fractional bits.
    """
    def __init__(self, fixed_size, fixed_prec):
        self.fixed_size=fixed_size
        self.fixed_prec=fixed_prec
        self.fixed_max_signed =  (1<<(fixed_size-1)-1)/(1<<fixed_prec)
        self.fixed_min_signed = -(1<<(fixed_size-fixed_prec-1))
    def C_quantize(self,tensor):
        return FixedPoint.round_stochastic(tensor,self.fixed_size,self.fixed_prec)
    def quantize(self,tensor):
        @tf.custom_gradient
        def op(tensor):
            def grad(dy):
                return dy
            randn = tf.random.uniform(tensor.shape, minval=0, maxval=1 )
            out_up = tf.math.ceil( tensor*(1<<self.fixed_prec) ) / (1<<self.fixed_prec)
            out_down = tf.math.floor( tensor*(1<<self.fixed_prec) ) / (1<<self.fixed_prec)
            out_mask = tf.less_equal( (tensor-tf.math.floor(tensor))*(1<<self.fixed_prec) ,randn )
            out = out_down * tf.dtypes.cast(out_mask, tensor.dtype) + out_up * tf.dtypes.cast(tf.math.logical_not(out_mask), tensor.dtype)
            # handle overflow (saturate number towards maximum or minimum)
            out = tf.math.maximum( tf.math.minimum( out, self.fixed_max_signed ), self.fixed_min_signed)
            # tag output
            out = tf.identity(out, name=str(self)+"_output")
            return out, grad
        return op(tensor)


###############################
### Logarithmic
###############################
class LogarithmicQuantizer(Quantizer_if):
    """Log2 quantization. Transforms the values into the closest of the form +/- 2^i.
    """
    def __init__(self):
        pass
    def quantize(self,tensor):
        return Wrapped.quant_log(tensor)
    def P_quantize(self, tensor):
        @tf.custom_gradient
        def op(tensor):
            def grad(dy):
                return dy
            #randn = tf.random.uniform(tensor.shape, minval=0, maxval=1 )
            #mask = tf.dtypes.cast(tf.less(tensor,randn), tensor.dtype)
            out= tf.math.floor(tf.math.log(tf.math.abs(tensor))/tf.math.log(tf.constant(2,dtype=tensor.dtype))) #+ mask
            out= tf.math.pow(2*tf.ones_like(tensor),out)
            out= out*tf.sign(tensor)
            out = tf.identity(out, name=str(self)+"_output")
            return out, grad
        return op(tensor)


###############################
### Sparse
###############################
class SparseQuantizer(Quantizer_if):
    """Every element whose magnitude is below the threshold is set to 0.
       Uses c-kernel for quantization.
    """
    def __init__(self, threshold):
        self.threshold = threshold
    def C_quantize(self,tensor):
        return Wrapped.quant_sparse(tensor, self.threshold)
    def quantize(self, tensor):
        @tf.custom_gradient
        def op(tensor):
            def grad(dy):
                return dy
            out_mask= tf.greater_equal(tf.math.abs(tensor),self.threshold)
            out = tensor * tf.dtypes.cast(out_mask, tensor.dtype)
            out = tf.identity(out, name=str(self)+"_output")
            return out, grad
        return op(tensor)

###############################
### Half-precision floating point
###############################
class HalffpQuantizer(Quantizer_if):
    """FP16 quantization. Rounds a floating point number into a half-precision one.
       Uses c-kernel for quantization.
    """
    def __init__(self):
        pass
    def C_quantize(self,tensor):
        return Wrapped.quant_halffp(tensor)
    def quantize(self, tensor):
        @tf.custom_gradient
        def op(tensor):
            def grad(dy):
                return dy
            out=tf.dtypes.cast(tf.dtypes.cast(tensor,tf.float16),tensor.dtype)
            out = tf.identity(out, name=str(self)+"_output")
            return out, grad
        return op(tensor)

###############################
### Binary
###############################
class BinaryQuantizer(Quantizer_if):
    """Binary quantization. Rounds to +marginal if input is >=0, or -marginal if input <0.
    """
    def __init__(self, marginal):
        self.marginal=marginal
    def C_quantize(self,tensor):
        return Wrapped.quant_binary(tensor, self.marginal)
    def quantize(self, tensor):
        @tf.custom_gradient
        def op(tensor):
            def grad(dy):
                return dy
            out = (tf.dtypes.cast(tf.greater_equal(tensor,0),tensor.dtype)*2-1)*self.marginal
            out = tf.identity(out, name=str(self)+"_output")
            return out, grad
        return op(tensor)

###############################
### Ternary
###############################
class TernaryQuantizer(Quantizer_if):
    """Ternary quantization. Rounds to +marginal if input is >threshold, -marginal if input <-threshold or 0.

    """
    def __init__(self, marginal, auto_threshold=True, threshold=0.5):
        self.marginal=marginal
        self.auto_threshold=auto_threshold
        self.threshold=threshold
    def C_quantize(self,tensor):
        return Wrapped.quant_ternary(tensor, self.marginal, self.auto_threshold, self.threshold)
    def quantize(self, tensor):
        @tf.custom_gradient
        def op(tensor):
            def grad(dy):
                return dy
            if self.auto_threshold:
                threshold = -0.7 * tf.math.reduce_sum(tf.math.abs(tensor))/tf.dtypes.cast(tf.size(tensor),tensor.dtype)
            else:
                threshold = self.threshold
            out = tf.ones_like(tensor)*-1
            out += tf.dtypes.cast(tf.greater(tensor, -threshold),tensor.dtype)
            out += tf.dtypes.cast(tf.greater(tensor, threshold),tensor.dtype)
            out *= self.marginal
            out = tf.identity(out, name=str(self)+"_output")
            return out, grad
        return op(tensor)



###############################
### Other
###############################
class NoQuantizer(Quantizer_if):
    """Applies no quantization to the tensor"""
    def quantize(self,tensor):
        return tensor
