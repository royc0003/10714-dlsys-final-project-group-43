"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy
import needle.init as init

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND, NDArray as BackendNDArray
from .ops_tuple import *


BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return numpy.power(a, b)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        left = out_grad * (b * power(a, b-1))
        right = out_grad * ( power(a,b) * log(a))
        return left, right
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # there should only be 1 input here
        a, = node.inputs
        diff = self.scalar * (a ** (self.scalar - 1))
        return out_grad * diff
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        left = out_grad / b
        right = -out_grad * a  / b / b
        ### END YOUR SOLUTION
        return left, right


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        ndim = len(a.shape)
        if ndim < 2:
            # For 1D or 0D arrays, transpose is a no-op
            return a
        
        if not self.axes:
            axis_1, axis_2 = -1, -2
        else:
            axis_1, axis_2 = self.axes[0], self.axes[1]
        
        # Convert negative indices to positive
        axis_1 = axis_1 % ndim
        axis_2 = axis_2 % ndim
        
        # Create permutation: swap axis_1 and axis_2
        perm = list(range(ndim))
        perm[axis_1], perm[axis_2] = perm[axis_2], perm[axis_1]
        return a.permute(tuple(perm))
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if not self.axes:
          axis_1, axis_2 = -1, -2
        elif self.axes and len(self.axes) == 2:
          axis_1, axis_2 = self.axes[0], self.axes[1]
        return transpose(out_grad,axes=(axis_1, axis_2))
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if not a.is_compact():
            a = a.compact()
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs

        return reshape(out_grad, a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.broadcast_to(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Sum all of the axes that were changed 
        a, = node.inputs
        original_shape = a.shape
        # pre-populate with the index
        # e.g. [1, 2, 3, 4]
        index_added = [i for i in range(len(self.shape))]
        rev_orig, rev_shape = original_shape[::-1], self.shape[::-1]
        i, j = 0, 0
        # keep only index we want
        while i < len(rev_shape) and j < len(rev_orig):
          # if matched we can ignore them
          if rev_shape[i] == rev_orig[j]:
            index_added[len(self.shape) - i - 1] = -1
          i += 1
          j += 1
        # print("original shape:")
        # print(original_shape)
        # print("expected shape")
        # print(self.shape)
        # print("testing index_added")
        # print(index_added)
        new_index_added = []
        for val in index_added:
          if val == -1:
            continue
          new_index_added.append(val)
        result = out_grad
        if new_index_added:
          for axis in sorted(new_index_added, reverse=True):
            result = summation(result, axes=axis)
        
        return result.reshape(original_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Convert single axis to tuple if needed
        axes = self.axes
        if axes is not None and isinstance(axes, int):
            axes = (axes,)
        return a.sum(axis=axes, keepdims=False)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Note to self, this is a reverse of broadcast
        original, = node.inputs
        original_shape = original.shape

        # normalize axes
        axes = self.axes
        if axes is None:
            axes = tuple(range(len(original_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        # handle negatives
        axes = tuple(a % len(original_shape) for a in axes)

        new_shape = list(original_shape)
        for ax in axes:
            new_shape[ax] = 1

        # reshape then broadcast back
        return out_grad.reshape(tuple(new_shape)).broadcast_to(original_shape)
        # original, = node.inputs
        # original_shape = original.shape
        # # print('this is original shape')
        # # print(original_shape)
        # new_shape = [i for i in original_shape]
        # # flatten the values
        # axes = self.axes if self.axes else range(len(original_shape))
        # for axe in axes:
        #   new_shape[axe] = 1
        # return out_grad.reshape(new_shape).broadcast_to(original_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        left, right = node.inputs
        # print("BEFORE-----------")
        # print("This is left shape \n")
        # print(left.shape)
        # print("This is right shape \n")
        # print(right.shape)
        # print("this is outgrad shape")
        # print(out_grad.shape)
        left_grad = matmul(out_grad, transpose(right, axes=(-1, -2)))
        right_grad = matmul(transpose(left, axes=(-1,-2)), out_grad)
        left_length, right_length, out_length = len(left.shape), len(right.shape), len(out_grad.shape)
        # print("After-----------")
        # print("This is left-grad shape \n")
        # print(left_grad.shape)
        # print("This is right-grad shape \n")
        # print(right_grad.shape)
        # print("this is outgrad shape")
        # print(out_grad.shape)
        if out_length > left_length:
          left_grad = summation(left_grad, axes=tuple(range(out_length - left_length)))
        if out_length > right_length:
          right_grad = summation(right_grad, axes=tuple(range(out_length - right_length)))
        # print("SOLUTION:__---")
        # print("left shape")
        # print(left_grad.shape)
        # print("Right shape")
        # print(right_grad.shape)

        return  left_grad , right_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -1 * a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * -1
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return numpy.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        return out_grad / a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return numpy.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        return out_grad * exp(a)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.maximum(0.0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.realize_cached_data()
        # Create mask: 1 where a > 0, 0 otherwise
        mask = (a > 0.0)
        return out_grad * Tensor(mask, device=a.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        # d/dx tanh(x) = 1 - tanh^2(x)
        tanh_a = tanh(a)
        ones = Tensor(init.ones_like(a))
        return out_grad * (ones - tanh_a * tanh_a)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # args should be a tuple of NDArrays (realized cached_data from TensorTuple)
        # Convert to list for easier handling
        if isinstance(args, tuple):
            arrays = list(args)
        else:
            # If it's iterable, convert to list
            arrays = list(args)
        
        if not arrays:
            raise ValueError("Stack requires at least one array")
        
        # Convert to numpy for stacking
        np_arrays = [arr.numpy() for arr in arrays]
        # Stack along the specified axis
        stacked_np = numpy.stack(np_arrays, axis=self.axis)
        
        # Convert back to NDArray using the same device as the first array
        device = arrays[0].device
        return BackendNDArray(stacked_np, device=device)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Gradient of stack is split
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        # A is an NDArray, split it along axis
        np_array = A.numpy()
        # Split along the axis - each element along that axis becomes a separate array
        axis_size = A.shape[self.axis]
        
        # Use indexing to get each slice along the axis
        splits = []
        device = A.device
        for i in range(axis_size):
            # Create index tuple to get slice at position i along axis
            indices = [slice(None)] * len(A.shape)
            indices[self.axis] = i
            np_slice = np_array[tuple(indices)]
            splits.append(BackendNDArray(np_slice, device=device))
        
        return tuple(splits)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Gradient of split is stack
        # out_grad is a TensorTuple, convert to list of tensors
        return stack([out_grad[i] for i in range(len(out_grad))], axis=self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Flipping is its own inverse, so gradient is just flipping the output gradient
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Calculate new shape: for dilated axes, new_size = old_size * (1 + dilation)
        new_shape = []
        for i in range(len(a.shape)):
            if i in self.axes:
                new_shape.append(a.shape[i] * (1 + self.dilation))
            else:
                new_shape.append(a.shape[i])
        
        # Create output array filled with zeros
        device = a.device
        out = BackendNDArray.make(tuple(new_shape), device=device)
        out.fill(0.0)
        
        # Copy original elements to positions: for dilated axis i, position = i * (1 + dilation)
        # Build slices for placing original data
        slices = []
        for i in range(len(a.shape)):
            if i in self.axes:
                # Along this axis, we place elements at positions 0, (1+d), 2*(1+d), ...
                # Use slice(start, stop, step)
                slices.append(slice(0, None, 1 + self.dilation))
            else:
                slices.append(slice(None))
        
        # Copy data using NDArray's __setitem__
        out[tuple(slices)] = a.compact()
        
        return out.compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Gradient of dilate is undilate
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Calculate the original shape (before dilation): for dilated axes, divide by (1 + dilation)
        original_shape = []
        for i in range(len(a.shape)):
            if i in self.axes:
                # Original size = current size / (1 + dilation)
                original_shape.append(a.shape[i] // (1 + self.dilation))
            else:
                original_shape.append(a.shape[i])
        
        # Extract elements at positions 0, (1+d), 2*(1+d), ... along dilated axes
        slices = []
        for i in range(len(a.shape)):
            if i in self.axes:
                # Extract elements at step (1 + dilation)
                slices.append(slice(0, None, 1 + self.dilation))
            else:
                slices.append(slice(None))
        
        # Extract the data using slicing
        extracted = a[tuple(slices)]
        
        # Reshape to original shape if needed (should already be correct, but ensure it)
        if extracted.shape != tuple(original_shape):
            extracted = extracted.reshape(tuple(original_shape))
        
        return extracted.compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Gradient of undilate is dilate (the inverse operation)
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # A: (N, H, W, C_in) in NHWC format
        # B: (K, K, C_in, C_out) - weights
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        
        # Apply padding to spatial dimensions (axes 1 and 2) if padding > 0
        if self.padding > 0:
            # Pad with zeros: (padding, padding) on left and right for H and W
            # Padding format: ((left, right), ...) for each dimension
            # For NHWC: axis 0=batch (0,0), axis 1=H (padding, padding), axis 2=W (padding, padding), axis 3=channels (0,0)
            padding_axes = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
            A = A.pad(padding_axes)
            # Update H and W after padding
            H = H + 2 * self.padding
            W = W + 2 * self.padding
        
        # Calculate output dimensions
        # Output H = (H - K) // stride + 1
        # Output W = (W - K) // stride + 1
        out_H = (H - K) // self.stride + 1
        out_W = (W - K) // self.stride + 1
        
        # Get strides of A (after padding)
        Ns, Hs, Ws, Cs = A.strides
        
        # Create im2col using as_strided
        # Shape: (N, out_H, out_W, K, K, C_in)
        # For stride > 1, we need to sample every stride-th position
        # Strides: batch stride, H stride (scaled by self.stride), W stride (scaled by self.stride),
        #          then repeat H and W strides for the K x K window, then channel stride
        im2col_shape = (N, out_H, out_W, K, K, C_in)
        im2col_strides = (Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)
        
        # Create strided view
        A_strided = A.as_strided(im2col_shape, im2col_strides)
        
        # Flatten to (N * out_H * out_W) x (K * K * C_in)
        inner_dim = K * K * C_in
        num_patches = N * out_H * out_W
        A_flat = A_strided.compact().reshape((num_patches, inner_dim))
        
        # Reshape weight to (K * K * C_in) x C_out
        B_flat = B.compact().reshape((inner_dim, C_out))
        
        # Matrix multiplication: (N * out_H * out_W) x C_out
        out_flat = A_flat @ B_flat
        
        # Reshape back to (N, out_H, out_W, C_out)
        out = out_flat.reshape((N, out_H, out_W, C_out))
        
        return out.compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_tensor, weight_tensor = node.inputs
        batch_size, input_height, input_width, input_channels = input_tensor.shape
        kernel_size, _, _, output_channels = weight_tensor.shape
        
        dilated_out_grad = dilate(out_grad, (1, 2), self.stride - 1) if self.stride > 1 else out_grad
        
        input_data = input_tensor.realize_cached_data()
        weight_data = weight_tensor.realize_cached_data()
        
        permuted_input = input_data.permute((3, 1, 2, 0))
        flipped_weight = weight_data.flip((0, 1))
        flipped_transposed_weight = flipped_weight.permute((0, 1, 3, 2))
        
        output_height = ((input_height + 2 * self.padding - kernel_size) // self.stride + 1) * self.stride
        input_padding = (input_height + kernel_size - output_height - 1) // 2
        weight_padding = (kernel_size + output_height - input_height - 1) // 2
        
        input_grad = conv(
            dilated_out_grad,
            Tensor(flipped_transposed_weight, dtype=out_grad.dtype, device=out_grad.device),
            stride=1,
            padding=input_padding
        )
        
        transposed_out_grad = dilated_out_grad.transpose((0, 2)).transpose((0, 1))
        weight_grad_intermediate = conv(
            Tensor(permuted_input, dtype=out_grad.dtype, device=out_grad.device),
            transposed_out_grad,
            stride=1,
            padding=weight_padding
        )
        weight_grad = weight_grad_intermediate.transpose((0, 2)).transpose((0, 1))

        return input_grad, weight_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


