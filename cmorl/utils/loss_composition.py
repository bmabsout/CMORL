import tensorflow as tf
import numpy as np


@tf.function
def geo(l, axis=0):
    return tf.exp(tf.reduce_mean(tf.math.log(l), axis=axis))


# @tf.function
# def p_mean(l, p, slack=0.0, axis=1):
#     slacked = l + slack
#     if len(slacked.shape) == 1:  # enforce having batches
#         slacked = tf.expand_dims(slacked, axis=0)
#     batch_size = slacked.shape[0]
#     zeros = tf.zeros(batch_size, l.dtype)
#     ones = tf.ones(batch_size, l.dtype)
#     handle_zeros = (
#         tf.reduce_all(slacked > 1e-20, axis=axis)
#         if p <= 1e-20
#         else tf.fill((batch_size,), True)
#     )
#     escape_from_nan = tf.where(
#         tf.expand_dims(handle_zeros, axis=axis), slacked, slacked * 0.0 + 1.0
#     )
#     handled = (
#         geo(escape_from_nan, axis=axis)
#         if p == 0
#         else tf.reduce_mean(escape_from_nan**p, axis=axis) ** (1.0 / p)
#     ) - slack
#     res = tf.where(handle_zeros, handled, zeros)
#     return res


@tf.function
def tf_pop(tensor, axis):
    return tf.concat([tf.slice(tensor, [0], [axis]), tf.slice(tensor, [axis+1], [-1])], 0)


@tf.function
def p_mean(l, p: float, slack=1e-15, default_val=0.0, axis=None):
    """
    The Generalized mean
    l: a tensor of elements we would like to compute the p_mean with respect to, elements must be > 0.0
    p: the value of the generalized mean, p = -1 is the harmonic mean, p = 1 is the regular mean, p=inf is the max function ...
    slack: allows elements to be at 0.0 with p < 0.0 without collapsing the pmean to 0.0 fully allowing useful gradient information to leak
    axis: axis or axese to collapse the pmean with respect to, None would collapse all
    https://www.wolframcloud.com/obj/26a59837-536e-4e9e-8ed1-b1f7e6b58377
    """
    l = tf.convert_to_tensor(l)
    p = tf.cast(p, l.dtype)
    slack = tf.cast(slack, l.dtype)
    default_val = tf.cast(default_val, l.dtype)
    p = tf.where(tf.abs(p) < 1e-3, -1e-3 if p < 0.0 else 1e-3, p)

    return tf.cond(tf.reduce_prod(tf.shape(l)) == 0 # condition if an empty array is fed in
        , lambda: tf.broadcast_to(default_val, tf_pop(tf.shape(l), axis)) if axis else default_val
        , lambda: tf.reduce_mean((l + slack)**p, axis=axis))**(1.0/p) - slack
        # tf.debugging.assert_greater_equal(l, 0.0)


@tf.function
def p_to_min(l, p=0, q=0):
    deformator = p_mean(1.0 - l, q)
    return p_mean(l, p) * deformator + (1.0 - deformator) * tf.reduce_min(l)


# @tf.function
# def with_mixer(actions): #batch dimension is 0
#     return actions-tf.reduce_min(actions,axis=1)

# def mixer_diff_dfl(a1,a2):
#     return tf.abs(with_mixer(a1)-with_mixer(a2))/2.0


@tf.function
def laplace_smoothing(weaken_me, weaken_by):
    return (weaken_me + weaken_by) / (1.0 + weaken_by)


@tf.custom_gradient
def scale_gradient(x, scale):
    def grad(dy):
        return (dy * scale, None)

    return x, grad


@tf.custom_gradient
def move_toward_zero(x):
    # tweaked to be a good activity regularizer for tanh within the dfl framework
    def grad(dy):
        return -dy * x * x * x * 5.0

    return tf.sigmoid(-tf.abs(x) + 5), grad


@tf.function
def sigmoid_regularizer(x):
    return tf.where(tf.abs(x) > 3, tf.abs(x), 0)


@tf.custom_gradient
def move_towards_range(x, min, max):
    min = tf.cast(min, x.dtype)
    max = tf.cast(max, x.dtype)
    normalized = 2.0*(x-min)/(max-min)-1.0
    in_range = tf.abs(normalized) <= 1.0
    def grad(dy):
        return -dy*tf.where(in_range, 0.0, normalized-tf.sign(normalized)), None, None

    return 1.0 / tf.where(in_range, 1.0, tf.abs(normalized)**0.5), grad

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    with tf.GradientTape() as gt:
        x = tf.Variable(tf.linspace(-2.0, 2.0, 100))
        y = move_towards_range(x, 0.0, 1.0)

    plt.plot(x, gt.gradient(y,x))
    plt.plot(x, y)
    plt.show()