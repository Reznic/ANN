"""Utilities module."""
import numpy as np
from collections import namedtuple

DFunc = namedtuple("DFunc", "func deriv")

# Todo: sigmoid derivative could be optimized: recalc of sig(x).
SIGMOID = DFunc(func=lambda x: 1 / (1 + np.exp(-x)),
                deriv=lambda x: SIGMOID.func(x) * (1 - SIGMOID.func(x)))

ERROR_FUNCTION = DFunc(func=lambda y, o: 0.5 * ((y - o) ** 2),
                       deriv=lambda y, o: -(y - o))

def gradient_descent(x, deriv, step):
    """Gradient descent on a function.
    
    Args:
        x (number): the current x value guess for minimum.
        deriv (func): the derivative of the target function.
        step (number): the climb step rate hyperparameter.
    """
    return x - step * deriv(x)

