import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Lambda, Activation
import gymnasium as gym
from gymnasium import spaces


def mlp_functional(
    inputs,
    hidden_sizes=(32,),
    activation="relu",
    use_bias=True,
    output_activation="sigmoid",
):
    layer = inputs
    for hidden_size in hidden_sizes[:-1]:
        # glorot_limit = np.sqrt(6 / hidden_size*10.0 + layer.shape[1])*0.02
        layer = Dense(
            units=hidden_size,
            activation=activation,
        )(layer)
    outputs = Dense(
        units=hidden_sizes[-1],
        activation=output_activation,
        name="output",
        use_bias=use_bias,
    )(layer)

    return outputs


def scale_by_space(scale_me, space):  # scale_me: [0,1.0]
    return scale_me * (space.high - space.low) + space.low


def unscale_by_space(unscale_me, space):  # outputs [-0.5, 0.5]
    return (unscale_me - space.low) / (space.high - space.low) - 0.5


"""
Actor-Critics
"""


def actor(obs_space: spaces.Box, act_space: spaces.Box, hidden_sizes, obs_normalizer):
    inputs = tf.keras.Input((obs_space.shape[0],))
    normalized_input = Lambda(lambda t: t / obs_normalizer)(inputs)
    # unscaled = unscale_by_space(inputs, obs_space)
    linear_output = mlp_functional(
        normalized_input,
        hidden_sizes + (act_space.shape[0],),
        use_bias=True,
        output_activation=None,
        activation="relu",
    )
    tanhed = Activation("tanh")(linear_output)
    # clipped = Lambda(lambda t: tf.clip_by_value(
    #     t, -1.0, 1.0))(normed)
    # scaled = scale_by_space(normed, act_space)
    model = tf.keras.Model(inputs, tanhed)
    model.compile()
    return model


def critic(obs_space: spaces.Box, act_space: spaces.Box, hidden_sizes, obs_normalizer):
    concated_normalizer = np.concatenate([obs_normalizer, np.ones(act_space.shape[0])])
    inputs = tf.keras.Input((obs_space.shape[0] + act_space.shape[0],))
    normalized_input = Lambda(lambda t: t / concated_normalizer)(inputs)
    outputs = mlp_functional(
        normalized_input, hidden_sizes + (1,), output_activation=None
    )

    # name the layer before sigmoid
    before_sigmoid = Lambda(lambda t: t * 0.1 - 0.5, name="before_sigmoid")(outputs)

    biased_normed = Activation("sigmoid")(before_sigmoid)
    model = tf.keras.Model(inputs, biased_normed)
    model.compile()
    return model


def mlp_actor_critic(
    obs_space: spaces.Box,
    act_space: spaces.Box,
    obs_normalizer=None,
    actor_hidden_sizes=(32, 32),
    critic_hidden_sizes=(256, 256),
) -> tuple[Model, Model]:
    if obs_normalizer is None:
        obs_normalizer = obs_space.high * 0.0 + 1.0
    obs_normalizer = np.array(obs_normalizer)
    return (
        actor(obs_space, act_space, actor_hidden_sizes, obs_normalizer),
        critic(obs_space, act_space, critic_hidden_sizes, obs_normalizer),
    )
