"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay
        self.params = params
        for param in params:
          self.u[hash(param)] = 0.0

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
          if not param.grad:
            continue
          g = param.grad.data + self.weight_decay * param.data
          u = self.momentum * self.u[hash(param)] + (1 - self.momentum)*g
          self.u[hash(param)] = u
          new_data = param.data - self.lr * u
          param.data = ndl.Tensor(new_data.numpy().astype(param.dtype), dtype=param.dtype)
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        pass
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}
        # initialization
        for param in self.params:
          self.m[hash(param)] = 0
          self.v[hash(param)] = 0

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t = self.t + 1
        for param in self.params:
          if not param.grad:
            continue
          param_key = hash(param)
          # FIX: Correct weight decay formula
          gradient = param.grad.data + self.weight_decay * param.data
          # FIX: Correct variable assignments
          m = self.beta1 * self.m[param_key] + (1 - self.beta1) * gradient
          self.m[param_key] = m
          v = self.beta2 * self.v[param_key] + (1 - self.beta2) * gradient * gradient
          self.v[param_key] = v

          # bias correction
          m_hat = m / (1 - self.beta1**self.t)
          v_hat = v / (1 - self.beta2**self.t)

          new_data = param.data - self.lr * ndl.ops.divide(m_hat, ndl.ops.power_scalar(v_hat, 0.5) + self.eps)
          param.data = ndl.Tensor(new_data.numpy().astype(param.dtype), dtype=param.dtype)
        ### END YOUR SOLUTION
