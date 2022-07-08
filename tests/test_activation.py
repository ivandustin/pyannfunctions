from jax.numpy import array, inf, array_equal
from annfunctions import activation

def test():
    input    = array([-inf, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, inf])
    expected = array([ 0.0,  0.0,  0.0, 0.0, 0.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0])
    actual   = activation(input)
    assert array_equal(actual, expected)
