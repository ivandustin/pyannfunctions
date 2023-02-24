from jax.numpy import array, round
from annfunctions import loss


def test():
    observed = array([34, 37, 44, 47, 48, 48, 46, 43, 32, 27, 26, 24])
    predicted = array([37, 40, 46, 44, 46, 50, 45, 44, 34, 30, 22, 23])
    expected = 5.91667
    actual = loss(observed, predicted)
    actual = round(actual, decimals=5)
    assert actual == expected
