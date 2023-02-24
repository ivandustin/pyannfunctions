from jax.numpy import array, nan, inf, all
from annfunctions import prune


def test():
    input = array([nan, -inf, -1.0, 0.0, 1.0, inf, nan])
    expected = array([0.0, -inf, -1.0, 0.0, 1.0, inf, 0.0])
    actual = prune(input)
    assert all(actual == expected)
