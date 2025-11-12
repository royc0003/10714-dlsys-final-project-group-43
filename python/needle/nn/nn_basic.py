"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__() # This sets the training = true
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        # need to broadcast here
        if bias:
      
          b = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)
          # print("Current bias shape:")
          # print(b.shape)
          # Need to be reshaped because
          # the output_shape is (1, out_features); but current bias shape is
          # (out_features, 1)
          expected_shape = (1, out_features)
          b = b.reshape((expected_shape))
          # print("After reshaping bias")
          # print(b.shape)
          self.bias = Parameter(b)
        else:
          self.bias = None
        
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Linear Transformation:: y = xA.T + b
        input_shape = X.shape
        in_features = self.in_features

        if input_shape[-1] != in_features:
            raise ValueError("Input feature dimension mismatch in Linear layer")

        if len(input_shape) == 2:
            output = ops.matmul(X, self.weight)
        else:
            batch = 1
            for dim in input_shape[:-1]:
                batch *= dim
            X_flat = ops.reshape(X, (batch, in_features))
            output_flat = ops.matmul(X_flat, self.weight)
            output = ops.reshape(output_flat, (*input_shape[:-1], self.out_features))

        if self.bias is not None:
            bias_shape = (1,) * (len(output.shape) - 1) + (self.out_features,)
            bias = ops.reshape(self.bias, bias_shape)
            bias = ops.broadcast_to(bias, output.shape)
            output = output + bias

        return output
        
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # print(X.shape)
        # print(X)
        # this is the batch 
        # e.g. (2,2,4,3)
        # batch_size = 2
        # non-batch size = 2, 4, 3
        # flattened = (2, 2 * 4 *3)
        batch_size = X.shape[0]
        
        flattened_size = 1
        for i in range(1, len(X.shape)):
            flattened_size *= X.shape[i]
        
        # Reshape to (batch_size, flattened_size)
        return X.reshape((batch_size, flattened_size))

        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
          x = module(x)
        ### END YOUR SOLUTION
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # print("logits shape")
        # print(logits.shape)
        # print(logits.shape[-1])
        # print(y.shape)
        no_of_categories = logits.shape[-1]
        y_one_hot_encoded = init.one_hot(no_of_categories, y, device=y.device)
        # print("y_one_hot shape")
        # print(y_one_hot_encoded.shape)
        # print("before one hot")
        # print(y)
        # print("After one hot")
        # print(y_one_hot_encoded)
        # print("logits")
        # print(logits)
        z_y = ops.summation(ops.multiply(logits, y_one_hot_encoded), axes=(-1,))
        # print("z-y shape")
        # print(z_y.shape)
        # print(z_y)
        z_i = ops.logsumexp(logits, axes=(-1,))
        pre_res = ops.summation(z_i - z_y)
        softmax_loss = ops.divide_scalar(pre_res, logits.shape[0])
        return softmax_loss
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        # shared across batch
        w = init.ones(dim)
        self.weight = Parameter(w, device=device, dtype=dtype)
        # shared across batch
        b = init.zeros(dim)
        self.bias = Parameter(b, device=device, dtype=dtype)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)        
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        B, D = x.shape
        if self.training:
          # per-feature mean across the batch (axes=0)
          mean = ops.summation(x, axes=0) / B                       # (D,)
          mean_b = ops.broadcast_to(ops.reshape(mean, (1, D)), x.shape)
          # update running_mean (exponential moving average on NDArray)
          self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        else:
          mean_b = ops.broadcast_to(ops.reshape(self.running_mean, (1, D)), x.shape)

        x_zero = x - mean_b

        if self.training:
            # per-feature var across the batch (biased, divide by B)
            var = ops.summation(x_zero * x_zero, axes=0) / B          # (D,)
            var_b = ops.broadcast_to(ops.reshape(var, (1, D)), x.shape)
            # update running_var
            self.running_var = (1 - self.momentum) * self.running_var + \
                               self.momentum * var.data
        else:
            var_b = ops.broadcast_to(ops.reshape(self.running_var, (1, D)), x.shape)

        inv_std = ops.power_scalar(var_b + self.eps, 0.5)
        x_hat = x_zero / inv_std

        # affine: broadcast (D,) -> (1,D) -> (B,D)
        w = ops.broadcast_to(ops.reshape(self.weight, (1, D)), x.shape)
        b = ops.broadcast_to(ops.reshape(self.bias,   (1, D)), x.shape)
        return x_hat * w + b
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))

class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        w = init.ones(dim)
        # cast to Parameter
        self.w = Parameter(w, device=device, dtype=dtype)
        b = init.zeros(dim)
        self.b = Parameter(b, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        feature_dim = x.shape[-1]

        mean = ops.summation(x, axes=-1) / feature_dim
        mean = ops.reshape(mean, (*x.shape[:-1], 1))
        mean = ops.broadcast_to(mean, x.shape)

        centered = x - mean

        variance = ops.summation(centered * centered, axes=-1) / feature_dim
        variance = ops.reshape(variance, (*x.shape[:-1], 1))
        variance = ops.broadcast_to(variance, x.shape)

        std = ops.power_scalar(variance + self.eps, 0.5)
        normalized = centered / std

        expand_shape = (1,) * (len(x.shape) - 1) + (feature_dim,)
        weight = ops.reshape(self.w, expand_shape)
        weight = ops.broadcast_to(weight, x.shape)
        bias = ops.reshape(self.b, expand_shape)
        bias = ops.broadcast_to(bias, x.shape)

        return normalized * weight + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Note: base on paper, this is to prevent over-fitting
        # by randomly turning off some neurons
        if not self.training:
            return x
        
        # Generate random binary mask with probability (1-p) of keeping elements
        prob = 1 - self.p
        mask = init.randb(*x.shape, p=prob, device=x.device, dtype=x.dtype)
        
        dropped = x * mask # zeroing out elements
        scaled = dropped / prob # scaling by probability
        return scaled
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
