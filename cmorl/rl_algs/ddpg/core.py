import numpy as np
from keras import Model # type: ignore
from keras.layers import Dense, Input, Rescaling, Activation # type: ignore
from gymnasium import spaces
import keras
import tensorflow as tf # type: ignore



def mlp_functional(
    inputs,
    hidden_sizes=(32,),
    activation="relu",
    use_bias=True,
    output_activation="sigmoid",
    seed=42
):
    keras.utils.set_random_seed(seed)
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

@keras.saving.register_keras_serializable(package="MyLayers")
class RescalingFixed(Rescaling):
    def __init__(self, scale, offset=0.0, **kwargs):
        if type(scale) is dict:
            scale = np.array(scale['config']['value'])
        if type(scale) is dict:
            scale = np.array(scale['config']['value'])
        super().__init__(scale, offset, **kwargs)


class ClipLayer(Activation):
    def __init__(self, min, max, **kwargs):
        self.min = min
        self.max = max
        if "activation" in kwargs:
            del kwargs["activation"]
        super().__init__(activation=lambda x: tf.clip_by_value(x, min, max), **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "min": self.min,
            "max": self.max,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def actor(obs_space: spaces.Box, act_space: spaces.Box, hidden_sizes, obs_normalizer, seed=42):
    inputs = Input((obs_space.shape[0],))
    normalized_input = RescalingFixed(1./obs_normalizer)(inputs)
    # unscaled = unscale_by_space(inputs, obs_space)
    linear_output = mlp_functional(
        normalized_input,
        hidden_sizes + (act_space.shape[0],),
        use_bias=True,
        output_activation=None,
        activation="relu",
        seed=seed
    )
    # tanhed = Activation("tanh")(linear_output)
    clipped = ClipLayer(-1.0, 1.0)(linear_output)
    # scaled = scale_by_space(normed, act_space)
    model = keras.Model(inputs, clipped)
    model.compile()
    return model


def critic(
    obs_space: spaces.Box,
    act_space: spaces.Box,
    hidden_sizes,
    obs_normalizer,
    rwds_dim=1,
    seed=42,
):
    concated_normalizer = np.concatenate([obs_normalizer, np.ones(act_space.shape[0])])
    inputs = Input((obs_space.shape[0] + act_space.shape[0],))
    normalized_input = RescalingFixed(1. / concated_normalizer)(inputs)
    outputs = mlp_functional(
        normalized_input, hidden_sizes + (rwds_dim,), output_activation=None, seed=seed
    )

    # name the layer before sigmoid
    before_clip = RescalingFixed(0.1, offset=0.3, name="before_clip")(outputs)

    biased_normed = ClipLayer(0.0, 1.0)(before_clip)
    model = keras.Model(inputs, biased_normed)
    model.compile()
    return model


def mlp_actor_critic(
    obs_space: spaces.Box,
    act_space: spaces.Box,
    rwds_dim: int,
    obs_normalizer=None,
    actor_hidden_sizes=(32, 32),
    critic_hidden_sizes=(256, 256),
    seed=42
) -> tuple[Model, Model]:
    if obs_normalizer is None:
        obs_normalizer = np.zeros_like(obs_space.high) + 1.0
    obs_normalizer = np.array(obs_normalizer)
    return (
        actor(obs_space, act_space, actor_hidden_sizes, obs_normalizer, seed=seed),
        critic(obs_space, act_space, critic_hidden_sizes, obs_normalizer, rwds_dim, seed=seed),
    )
