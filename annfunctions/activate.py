from jax.numpy import where, clip
from jax import custom_jvp


@custom_jvp
def activate(x):
    return where(x >= 1, clip(x, 1, 2), 0)


@activate.defjvp
def activate_jvp(primals, tangents):
    x,  = primals
    dy, = tangents
    return activate(x), dy
