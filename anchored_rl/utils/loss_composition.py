import tensorflow as tf


@tf.function
def geo(l, axis=0):
    return tf.exp(tf.reduce_mean(tf.math.log(l), axis=axis))


@tf.function
def p_mean(l, p, slack=0.0, axis=1):
    slacked = l + slack
    if(len(slacked.shape) == 1):  # enforce having batches
        slacked = tf.expand_dims(slacked, axis=0)
    batch_size = slacked.shape[0]
    zeros = tf.zeros(batch_size, l.dtype)
    ones = tf.ones(batch_size, l.dtype)
    handle_zeros = tf.reduce_all(
        slacked > 1e-20, axis=axis) if p <= 1e-20 else tf.fill((batch_size,), True)
    escape_from_nan = tf.where(tf.expand_dims(
        handle_zeros, axis=axis), slacked, slacked*0.0 + 1.0)
    handled = (
        geo(escape_from_nan, axis=axis)
        if p == 0 else
        tf.reduce_mean(escape_from_nan**p, axis=axis)**(1.0/p)
    ) - slack
    res = tf.where(handle_zeros, handled, zeros)
    return res


@tf.function
def p_to_min(l, p=0, q=0):
    deformator = p_mean(1.0-l, q)
    return p_mean(l, p)*deformator + (1.0-deformator)*tf.reduce_min(l)

# @tf.function
# def with_mixer(actions): #batch dimension is 0
#     return actions-tf.reduce_min(actions,axis=1)

# def mixer_diff_dfl(a1,a2):
#     return tf.abs(with_mixer(a1)-with_mixer(a2))/2.0


@tf.function
def laplace_smoothing(weaken_me, weaken_by):
    return (weaken_me + weaken_by)/(1.0 + weaken_by)


@tf.custom_gradient
def scale_gradient(x, scale):
  def grad(dy): return (dy * scale, None)
  return x, grad


@tf.custom_gradient
def move_toward_zero(x):
    #tweaked to be a good activity regularizer for tanh within the dfl framework
    def grad(dy):
        return -dy*x*x*x*5.0
    return tf.sigmoid(-tf.abs(x)+5), grad