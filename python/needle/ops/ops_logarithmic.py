from typing import Optional, Any, Union

from numpy._core.fromnumeric import reshape
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        z_max = Z.max(axis=(1,), keepdims=True)
        # Broadcast z_max to match Z's shape
        z_max_broadcast = z_max.broadcast_to(Z.shape)
        Z_shifted = Z - z_max_broadcast
        exp_z = Z_shifted.exp()
        exp_sum = exp_z.sum(axis=1, keepdims=True)
        log_sum_exp = exp_sum.log()
        # Broadcast log_sum_exp to match Z's shape
        log_sum_exp_broadcast = log_sum_exp.broadcast_to(Z.shape)

        return Z_shifted - log_sum_exp_broadcast
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        softmax = exp(node)  # node is logsoftmax(z), so exp(node) = softmax(z)

        # sum upstream grad over axis=1, keep dims for broadcasting
        gsum = summation(out_grad, axes=(1,))                 # shape (N,)
        gsum = reshape(gsum, (gsum.shape[0], 1))              # shape (N,1)
        gsum = broadcast_to(gsum, out_grad.shape)             # shape (N,C)

        # grad w.r.t. z
        return out_grad - softmax * gsum
        # # TODO: come back to this to resolve it
        # Z = node.inputs[0]
    
        # # Convert to numpy array
        # Z_numpy = Z.numpy()
        
        # # Compute softmax (same as compute function)
        # z_max = array_api.max(Z_numpy, axis=1, keepdims=True)
        # z_shifted = Z_numpy - z_max
        
        # exp_z = array_api.exp(z_shifted)
        # exp_sum = array_api.sum(exp_z, axis=1, keepdims=True)
        # softmax_z = exp_z / exp_sum
        
        # # Convert back to Tensor
        # softmax_tensor = Tensor(softmax_z, dtype=Z.dtype)
        # ones = Tensor(array_api.ones_like(softmax_z), dtype=Z.dtype)
        
        # # Gradient = out_grad * (1 - softmax)
        # return out_grad * (ones - softmax_tensor)
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        z_max = Z.max(axis=self.axes, keepdims=True)
        
        # Handle broadcasting: ensure z_max has the same number of dimensions as Z
        # This is needed for axes=None case where z_max might be scalar or have fewer dims
        if len(z_max.shape) < len(Z.shape):
            # Reshape to match Z's dimensionality (all dimensions of size 1)
            z_max = z_max.reshape((1,) * len(Z.shape))
        
        # Broadcast z_max to match Z's shape for subtraction
        z_max_broadcast = z_max.broadcast_to(Z.shape)
        z_shifted = Z - z_max_broadcast
        exp_z = z_shifted.exp()
        exp_sum = exp_z.sum(axis=self.axes, keepdims=True)
        
        # Handle exp_sum having fewer dimensions than Z
        if len(exp_sum.shape) < len(Z.shape):
            exp_sum = exp_sum.reshape((1,) * len(Z.shape))
        
        log_sum_exp = exp_sum.log()
        # Broadcast z_max to match log_sum_exp's shape for addition  
        res = log_sum_exp + z_max
        
        # Remove keepdims: squeeze out dimensions of size 1 that were reduced
        # res currently has shape with keepdims=True, so dimensions that were reduced are size 1
        if self.axes is not None:
            z_shape_len = len(Z.shape)
            # Normalize axes to positive indices
            if isinstance(self.axes, int):
                axes_normalized = (self.axes % z_shape_len if self.axes < 0 else self.axes,)
            else:
                axes_normalized = tuple(a % z_shape_len if a < 0 else a for a in self.axes)
            
            # Build the desired shape by squeezing out reduced dimensions
            desired_shape = []
            for i in range(len(res.shape)):
                # If this dimension corresponds to a reduced axis, skip it (squeeze it out)
                # Otherwise keep it
                if i < z_shape_len:
                    orig_axis = i
                else:
                    # This shouldn't happen with keepdims=True
                    desired_shape.append(int(res.shape[i]))
                    continue
                
                if orig_axis not in axes_normalized:
                    desired_shape.append(int(res.shape[i]))
                # else: skip this dimension (it was reduced)
            
            if desired_shape:
                res = res.reshape(tuple(desired_shape))
            else:
                # All dimensions were reduced, result is scalar
                res = res.reshape(())
        else:
            # Handle axes=None - all dimensions reduced, result is scalar
            res = res.reshape(())

        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z, = node.inputs
        # Use Tensor operations - realize cached data to get NDArray, compute, wrap back
        Z_data = Z.realize_cached_data()
        z_max_data = Z_data.max(axis=self.axes, keepdims=True)
        
        # Ensure z_max_data has same number of dimensions as Z_data for broadcasting
        if len(z_max_data.shape) < len(Z_data.shape):
            z_max_data = z_max_data.reshape((1,) * len(Z_data.shape))
        z_max_broadcast = z_max_data.broadcast_to(Z_data.shape)
        
        z_shifted_data = Z_data - z_max_broadcast
        z_exp_data = z_shifted_data.exp()
        z_sum_data = z_exp_data.sum(axis=self.axes, keepdims=True)
        
        # Ensure z_sum_data has same number of dimensions as z_exp_data for broadcasting
        if len(z_sum_data.shape) < len(z_exp_data.shape):
            z_sum_data = z_sum_data.reshape((1,) * len(z_exp_data.shape))
        z_sum_broadcast = z_sum_data.broadcast_to(z_exp_data.shape)
        
        dfdz_data = z_exp_data / z_sum_broadcast
        dfdz = Tensor(dfdz_data, device=Z.device)
        
        # Reshape and broadcast out_grad to match Z's shape
        res_shape = list(Z.shape)
        grad = out_grad
        if self.axes is not None:
            # Normalize axes
            if isinstance(self.axes, int):
                axes_normalized = (self.axes % len(Z.shape) if self.axes < 0 else self.axes,)
            else:
                axes_normalized = tuple(a % len(Z.shape) if a < 0 else a for a in self.axes)
            for val in axes_normalized:
                res_shape[val] = 1
            grad = reshape(out_grad, tuple(res_shape))
        
        return broadcast_to(grad, Z.shape) * dfdz
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)