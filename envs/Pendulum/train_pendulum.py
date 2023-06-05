import spinup.algos.ddpg.ddpg as rl_alg
import Pendulum
import numpy as np
import time
import pickle
import tensorflow as tf

def on_save(actor, q_network, epoch, replay_buffer):
    actor.save("pendulum/actor")
    q_network.save("pendulum/critic")
    with open( "pendulum/replay.p", "wb" ) as replay_file:
            pickle.dump( replay_buffer, replay_file)

def existing_actor_critic(*args, **kwargs):
    return tf.keras.models.load_model("pendulum/actor"), tf.keras.models.load_model("pendulum/critic")

rl_alg.ddpg(lambda: Pendulum.PendulumEnv(g=10.0, setpoint=0.0)
    , hp = rl_alg.HyperParams(
        seed=int(time.time()* 1e5) % int(1e6),
        steps_per_epoch=1000,
        ac_kwargs={
            "actor_hidden_sizes":(32,32),
            "critic_hidden_sizes":(256,256),
            "obs_normalizer": np.array([1.0, 1.0, 8.0])
        },
        pi_bar_variance=[0.0,0.0,0.0],
        start_steps=1000,
        replay_size=int(1e5),
        gamma=0.9,
        polyak=0.995,
        pi_lr=tf.optimizers.schedules.PolynomialDecay(3e-3, 1e6, end_learning_rate=1e-5),
        q_lr=tf.optimizers.schedules.PolynomialDecay(3e-3, 1e6, end_learning_rate=1e-5),
        batch_size=200,
        act_noise=0.1,
        max_ep_len=200,
        epochs=50,
        train_every=50,
        train_steps=30,
    )
    , on_save=on_save
    # , anchor_q=tf.keras.models.load_model("pendulum/critic")
    # , actor_critic=existing_actor_critic
)