import spinup.algos.sac as rl_alg
import Pendulum
import numpy as np
import time
import pickle
from collections import namedtuple
import tensorflow as tf

def on_save(act_crit, epoch, replay_buffer):
    act_crit.pi.save("sac_anchored_pendulum/actor")
    act_crit.q1.save("sac_anchored_pendulum/critic")
    # with open( "pendulum/replay.p", "wb" ) as replay_file:
    #         pickle.dump( replay_buffer, replay_file)

def existing_actor_critic(*args, **kwargs):
    saved_actor = rl_alg.core.Mlp_Gaussian_Actor.load("sac_left_pendulum/actor")
    saved_critic1 = tf.keras.models.load_model("sac_left_pendulum/critic")
    saved_critic2 = tf.keras.models.load_model("sac_left_pendulum/critic")

    ActorCritic = namedtuple('ActorCritic', 'pi q1 q2')
    return ActorCritic(saved_actor, saved_critic1, saved_critic2)

rl_alg.sac.sac(lambda: Pendulum.PendulumEnv(g=10.0, setpoint=-np.pi/5.0), seed=0, actor_critic=existing_actor_critic, anchor_q=tf.keras.models.load_model("sac_left_pendulum/critic"),
        steps_per_epoch=1000, epochs=200, replay_size=int(1e5), gamma=0.9, 
        polyak=0.995, lr=1e-3, alpha=0.01, batch_size=100, start_steps=1000, 
        update_after=900, update_every=50, num_test_episodes=4, num_opt_steps=50, max_ep_len=200, save_freq=1, on_save=on_save)