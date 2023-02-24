from jax.numpy import where, clip
from jax import custom_jvp


@custom_jvp
def activation(x):
    return where(x >= 1, clip(x, 1, 2), 0)


@activation.defjvp
def activation_jvp(primals, tangents):
    x,  = primals
    dy, = tangents
    return activation(x), dy
