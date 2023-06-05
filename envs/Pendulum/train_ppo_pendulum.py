import spinup.algos.ppo.ppo as rl_alg
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
    return tf.keras.models.load_model("right_leaning_pendulum/actor"), tf.keras.models.load_model("right_leaning_pendulum/critic")

rl_alg.ppo(lambda: Pendulum.PendulumEnv(g=10.0, setpoint=0.0), seed=0, 
        steps_per_epoch=1000, epochs=100, gamma=0.9, clip_ratio=0.2, pi_lr=1e-3,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=200,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10)