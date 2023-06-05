from spinup import td3
import Pendulum
import numpy as np
import time
import pickle
import tensorflow as tf

def on_save(actor, q_network, epoch, replay_buffer):
    actor.save("td3_pendulum_anchored/actor")
    q_network.save("td3_pendulum_anchored/critic")
    # with open( "td3_pendulum_left/replay.p", "wb" ) as replay_file:
    #         pickle.dump( replay_buffer, replay_file)

def existing_actor_critic(*args, **kwargs):
    return (tf.keras.models.load_model("td3_pendulum_left/actor")
    	, tf.keras.models.load_model("td3_pendulum_left/critic")
    	, tf.keras.models.load_model("td3_pendulum_left/critic"))

td3(lambda: Pendulum.PendulumEnv(g=10.0, setpoint=-np.pi/5), actor_critic=existing_actor_critic
	, seed=0, steps_per_epoch=1000, epochs=200, replay_size=int(1e5), gamma=0.9, 
        polyak=0.995, batch_size=200, start_steps=1000, max_ep_len=200, save_freq=1, on_save=on_save,anchor_q=tf.keras.models.load_model("td3_pendulum_left/critic"))