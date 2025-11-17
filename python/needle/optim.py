"""Optimization module with registry utilities for config-driven instantiation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Sequence, Type
import copy

import needle as ndl
import numpy as np


@dataclass(frozen=True, slots=True)
class _OptimizerSpec:
    """Internal metadata describing an optimizer."""

    name: str
    cls: Type["Optimizer"]
    defaults: Dict[str, Any] = field(default_factory=dict)
    supports_sparse: bool = False
    aliases: Sequence[str] = field(default_factory=tuple)


_OPTIMIZER_REGISTRY: Dict[str, _OptimizerSpec] = {}


def register_optimizer(
    name: str,
    cls: Type["Optimizer"],
    *,
    defaults: Dict[str, Any] | None = None,
    aliases: Sequence[str] | None = None,
    supports_sparse: bool = False,
) -> None:
    """Register an optimizer so it can be constructed from configuration.

    Args:
        name: Primary identifier used in config files.
        cls: Optimizer class implementing the Needle optimizer interface.
        defaults: Baseline hyperparameters applied when not overridden.
        aliases: Optional alternative identifiers resolving to the same optimizer.
        supports_sparse: True if the optimizer has a sparse update variant.
    """
    if not name:
        raise ValueError("Optimizer name must be a non-empty string.")
    key = name.lower()
    if key in _OPTIMIZER_REGISTRY:
        raise ValueError(f"Optimizer '{name}' already registered.")
    spec = _OptimizerSpec(
        name=name,
        cls=cls,
        defaults=dict(defaults or {}),
        supports_sparse=supports_sparse,
        aliases=tuple(aliases or ()),
    )
    _OPTIMIZER_REGISTRY[key] = spec
    for alias in spec.aliases:
        alias_key = alias.lower()
        if alias_key in _OPTIMIZER_REGISTRY:
            raise ValueError(f"Alias '{alias}' already registered.")
        _OPTIMIZER_REGISTRY[alias_key] = spec


def available_optimizers() -> Sequence[str]:
    """Return the list of distinct optimizer identifiers."""
    seen = set()
    names = []
    for spec in _OPTIMIZER_REGISTRY.values():
        if spec.name in seen:
            continue
        seen.add(spec.name)
        names.append(spec.name)
    return tuple(sorted(names))


def optimizer_spec(name: str) -> _OptimizerSpec:
    """Retrieve the spec for a previously registered optimizer."""
    try:
        return _OPTIMIZER_REGISTRY[name.lower()]
    except KeyError as exc:
        registered = ", ".join(sorted(set(spec.name for spec in _OPTIMIZER_REGISTRY.values())))
        raise KeyError(f"Unknown optimizer '{name}'. Available: {registered}") from exc


def build_optimizer_from_config(
    params: Iterable[ndl.Tensor],
    config: Dict[str, Any] | str,
) -> "Optimizer":
    """Instantiate an optimizer using a configuration dictionary or identifier.

    Args:
        params: Iterable of parameters to optimize.
        config: Either a string optimizer name or a dictionary containing:
            - name (str): registered optimizer name (required if dict).
            - Any additional keyword arguments forwarded to the optimizer.
            - hyperparams (dict): optional nested dict merged after defaults.

    Returns:
        An initialized optimizer instance.
    """
    if isinstance(config, str):
        name = config
        cfg_dict: Dict[str, Any] = {}
    else:
        if "name" not in config:
            raise KeyError("Optimizer config dictionary must contain a 'name' entry.")
        name = config["name"]
        cfg_dict = dict(config)

    spec = optimizer_spec(name)

    hyperparams = copy.deepcopy(spec.defaults)
    cfg_dict.pop("name", None)
    nested_hparams = cfg_dict.pop("hyperparams", {})
    if nested_hparams:
        if not isinstance(nested_hparams, dict):
            raise TypeError("'hyperparams' entry must be a dictionary.")
        hyperparams.update(nested_hparams)
    hyperparams.update(cfg_dict)

    return spec.cls(list(params), **hyperparams)


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


class RMSProp(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay

        self.v = {}
        # initialization
        for param in self.params:
          self.v[hash(param)] = 0

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
          if not param.grad:
            continue
          param_key = hash(param)
          # Apply weight decay to gradient
          gradient = param.grad.data + self.weight_decay * param.data
          # Update running average of squared gradients
          v = self.alpha * self.v[param_key] + (1 - self.alpha) * gradient * gradient
          self.v[param_key] = v
          # Update parameters: param = param - lr * g / (sqrt(v) + eps)
          new_data = param.data - self.lr * ndl.ops.divide(gradient, ndl.ops.power_scalar(v, 0.5) + self.eps)
          param.data = ndl.Tensor(new_data.numpy().astype(param.dtype), dtype=param.dtype)
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


class Adagrad(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay

        self.G = {}
        # Initialize accumulator for squared gradients
        for param in self.params:
            param_key = hash(param)
            self.G[param_key] = ndl.zeros_like(param.data)

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad is None:
                continue

            param_key = hash(param)
            if param_key not in self.G:
                self.G[param_key] = ndl.zeros_like(param.data)

            # Apply weight decay to gradient
            gradient = param.grad.data + self.weight_decay * param.data

            # Accumulate squared gradients: G = G + g²
            self.G[param_key] = self.G[param_key] + gradient * gradient

            # Update parameters: param = param - lr * g / (sqrt(G) + eps)
            denom = ndl.ops.power_scalar(self.G[param_key], 0.5) + self.eps
            new_data = param.data - self.lr * ndl.ops.divide(gradient, denom)
            param.data = ndl.Tensor(new_data.numpy().astype(param.dtype), dtype=param.dtype)
        ### END YOUR SOLUTION


class Adagrad(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay

        self.G = {}
        # Initialize accumulator for squared gradients
        for param in self.params:
            param_key = hash(param)
            self.G[param_key] = ndl.zeros_like(param.data)

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad is None:
                continue

            param_key = hash(param)
            if param_key not in self.G:
                self.G[param_key] = ndl.zeros_like(param.data)

            # Apply weight decay to gradient
            gradient = param.grad.data + self.weight_decay * param.data

            # Accumulate squared gradients: G = G + g²
            self.G[param_key] = self.G[param_key] + gradient * gradient

            # Update parameters: param = param - lr * g / (sqrt(G) + eps)
            denom = ndl.ops.power_scalar(self.G[param_key], 0.5) + self.eps
            new_data = param.data - self.lr * ndl.ops.divide(gradient, denom)
            param.data = ndl.Tensor(new_data.numpy().astype(param.dtype), dtype=param.dtype)
        ### END YOUR SOLUTION


class AdamW(Optimizer):
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
        for param in self.params:
            param_key = hash(param)
            self.m[param_key] = ndl.zeros_like(param.data)
            self.v[param_key] = ndl.zeros_like(param.data)

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t

        for param in self.params:
            if param.grad is None:
                continue

            param_key = hash(param)
            if param_key not in self.m:
                self.m[param_key] = ndl.zeros_like(param.data)
                self.v[param_key] = ndl.zeros_like(param.data)

            grad = param.grad.data
            m_prev = self.m[param_key]
            v_prev = self.v[param_key]

            m = self.beta1 * m_prev + (1 - self.beta1) * grad
            v = self.beta2 * v_prev + (1 - self.beta2) * (grad * grad)
            self.m[param_key] = m
            self.v[param_key] = v

            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            denom = ndl.ops.power_scalar(v_hat, 0.5) + self.eps
            adam_step = ndl.ops.divide(m_hat, denom)

            if self.weight_decay != 0.0:
                adam_step = adam_step + self.weight_decay * param.data

            new_data = param.data - self.lr * adam_step
            param.data = ndl.Tensor(new_data.numpy().astype(param.dtype), dtype=param.dtype)
        ### END YOUR SOLUTION


# Register built-in optimizers for config-driven workflows.
register_optimizer(
    "sgd",
    SGD,
    defaults={"lr": 0.01, "momentum": 0.0, "weight_decay": 0.0},
    aliases=("stochastic_gradient_descent",),
)

register_optimizer(
    "rmsprop",
    RMSProp,
    defaults={"lr": 0.01, "alpha": 0.99, "eps": 1e-8, "weight_decay": 0.0},
)

register_optimizer(
    "adagrad",
    Adagrad,
    defaults={"lr": 0.01, "eps": 1e-8, "weight_decay": 0.0},
)

register_optimizer(
    "adam",
    Adam,
    defaults={"lr": 0.01, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.0},
)

register_optimizer(
    "adamw",
    AdamW,
    defaults={"lr": 0.01, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.0},
)
