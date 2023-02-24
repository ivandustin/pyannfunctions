from jax.numpy import mean, square


def loss(observed, predicted):
    return mean(square(observed - predicted))
