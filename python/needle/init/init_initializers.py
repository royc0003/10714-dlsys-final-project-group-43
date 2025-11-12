import math
from .init_basic import *
from typing import Any


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, shape=None, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    a = gain * (6 / (fan_in + fan_out))**0.5
    if shape is not None:
        return rand(*shape, low=-1, high=1, **kwargs) * a
    return rand(fan_in, fan_out, low=-1, high=1, **kwargs) * a
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, shape=None, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    std = gain * ((2 / (fan_in + fan_out)) ** 0.5)
    if shape is not None:
        return randn(*shape, **kwargs) * std
    return randn(fan_in, fan_out, **kwargs) * std
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", shape=None, **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = 2**0.5
    bound = gain * ((3 / fan_in)**0.5)
    if shape is not None:
        return rand(*shape, low=-1, high=1, **kwargs) * bound
    return rand(fan_in, fan_out, low=-1, high=1, **kwargs) * bound
    ### END YOUR SOLUTION



def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", shape=None, **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = 2**0.5
    std = gain / (fan_in)**0.5
    if shape is not None:
        return randn(*shape, **kwargs) * std
    return randn(fan_in, fan_out, **kwargs) * std
    ### END YOUR SOLUTION