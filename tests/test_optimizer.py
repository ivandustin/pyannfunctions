from annfunctions import optimizer
from jax.numpy import array

def test():
    theta    = 1.0
    alpha    = 2.0
    gradient = 3.0
    expected = -5.0
    actual   = optimizer(theta, alpha, gradient)
    assert actual == expected
