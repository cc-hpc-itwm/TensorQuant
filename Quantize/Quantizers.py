from Quantize import FixedPoint
from Quantize import QuantKernelWrapper as Wrapped

class Quantizer_if():
    """Interface for quantizer classes"""
    def quantize(self,tensor):
        raise NotImplementedError

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
    def quantize(self,tensor):
        return FixedPoint.round_zero(tensor,self.fixed_size,self.fixed_prec)

class FixedPointQuantizer_down(Quantizer_if):
    """Fixed point quantization with fixed_size bits and fixed_prec fractional bits.
       Uses c-kernel for quantization. 
    """
    def __init__(self, fixed_size, fixed_prec):
        self.fixed_size=fixed_size
        self.fixed_prec=fixed_prec
    def quantize(self,tensor):
        return FixedPoint.round_down(tensor,self.fixed_size,self.fixed_prec)

class FixedPointQuantizer_nearest(Quantizer_if):
    """Fixed point quantization with fixed_size bits and fixed_prec fractional bits.
       Uses c-kernel for quantization. 
    """
    def __init__(self, fixed_size, fixed_prec):
        self.fixed_size=fixed_size
        self.fixed_prec=fixed_prec
    def quantize(self,tensor):
        return FixedPoint.round_nearest(tensor,self.fixed_size,self.fixed_prec)

class FixedPointQuantizer_stochastic(Quantizer_if):
    """Fixed point quantization with fixed_size bits and fixed_prec fractional bits.
       Uses c-kernel for quantization. 
    """
    def __init__(self, fixed_size, fixed_prec):
        self.fixed_size=fixed_size
        self.fixed_prec=fixed_prec
    def quantize(self,tensor):
        return FixedPoint.round_stochastic(tensor,self.fixed_size,self.fixed_prec)

###############################
### Logarithmic 
###############################
class LogarithmicQuantizer(Quantizer_if):
    """Log2 quantization. Transforms the values into the closest of the form +/- 2^i.
       Uses c-kernel for quantization. 
    """
    def __init__(self):
        pass
    def quantize(self,tensor):
        return Wrapped.quant_log(tensor)

###############################
### Sparse
###############################
class SparseQuantizer(Quantizer_if):
    """Every element whose magnitude is below the threshold is set to 0.
       Uses c-kernel for quantization. 
    """
    def __init__(self, threshold):
        self.threshold = threshold
    def quantize(self,tensor):
        return Wrapped.quant_sparse(tensor, self.threshold)

###############################
### Other
###############################
class NoQuantizer(Quantizer_if):
    """Applies no quantization to the tensor"""
    def quantize(self,tensor):
        return tensor
