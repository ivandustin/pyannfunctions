from jax.numpy import array, inf, array_equal, ones_like, sum
from jax import grad
from annfunctions import activation


def test():
    input = array([-inf, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, inf])
    expected = array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0])
    actual = activation(input)
    assert array_equal(actual, expected)


def test_gradient():
    input = array([-inf, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, inf])
    expected = ones_like(input)
    actual = grad(lambda input: sum(activation(input)))(input)
    assert array_equal(actual, expected)
