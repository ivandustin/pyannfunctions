from annfunctions import optimize


def test():
    theta = 1.0
    alpha = 2.0
    gradient = 3.0
    expected = -5.0
    actual = optimize(theta, gradient, alpha)
    assert actual == expected
