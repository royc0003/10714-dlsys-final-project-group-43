"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        receptive_field_size = self.kernel_size * self.kernel_size
        fan_in = self.in_channels * receptive_field_size
        fan_out = self.out_channels * receptive_field_size
        
        weight_shape = (self.kernel_size, self.kernel_size, self.in_channels, self.out_channels)
        self.weight = Parameter(
            init.kaiming_uniform(fan_in, fan_out, shape=weight_shape, device=device, dtype=dtype)
        )
        
        if bias:
            bias_bound = 1.0 / (fan_in ** 0.5)
            self.bias = Parameter(
                init.rand(self.out_channels, low=-bias_bound, high=bias_bound, device=device, dtype=dtype)
            )
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        padding = self.kernel_size // 2
        
        x_nhwc = ops.transpose(ops.transpose(x, axes=(1, 2)), axes=(2, 3))
        
        conv_out = ops.conv(x_nhwc, self.weight, stride=self.stride, padding=padding)
        
        out_nchw = ops.transpose(ops.transpose(conv_out, axes=(2, 3)), axes=(1, 2))
        
        if self.bias is not None:
            out_nchw = out_nchw + self.bias.reshape((1, self.out_channels, 1, 1)).broadcast_to(out_nchw.shape)
        
        return out_nchw
        ### END YOUR SOLUTION