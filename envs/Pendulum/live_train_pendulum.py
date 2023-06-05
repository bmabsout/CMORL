import transmitter
import receiver
import time
import serial
from threading import Thread
import xbee
import datetime
import live_ddpg
import copy
from gym import spaces
import numpy as np
import pickle
import os
from multiprocessing import Process, Queue
import tensorflow as tf
import obs_utils

import Pendulum

def on_save(actor, q_network, epoch):
    actor.save("left_leaning_actor")
    q_network.save("left_leaning_q")

def existing_actor_critic(*args, **kwargs):
    return tf.keras.models.load_model("right_leaning_actor"), tf.keras.models.load_model("right_leaning_q")

def clear_queue(q):
    while not q.empty():
        q.get()

def existing_actor_critic(*args, **kwargs):
    actor = tf.keras.models.load_model("actor")
    critic = tf.keras.models.load_model("critic")
    return actor, critic

def live_training(env, nn_queue, obs_queue):
    def on_save(actor, critic, epoch):
        save_path = os.path.join(".", f"ckpt_{epoch}")
        critic.save(os.path.join(save_path, "critic"))
        actor.save(os.path.join(save_path, "actor"))
        nn_queue.put(converted)

    live_ddpg.live_ddpg(
        obs_queue,
        env.observation_space,
        env.action_space,
        actor_critic=existing_actor_critic,
        on_save=on_save,
        anchor_q=tf.keras.models.load_model("critic")
    )

def env_stepper(env, nn):
    obs = env.reset()
    def run_step():
        prev_obs = obs
        action = nn([obs])[0]
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
        return prev_obs, action, obs
    return run_step

def tranceive(env, nn_queue, obs_queue, circular_buffer_size=2000):
    run_step = env_stepper(env, nn)
    next_obs = None
    while True:
        while nn_queue.empty():
            obs = env.
            if next_obs:
                    copied_next_obs=copy.deepcopy(next_obs)
                    current_traj.append((obs, copied_next_obs.prev_action))
                    obs_queue.put((obs, copied_next_obs.prev_action, copied_next_obs))
                    if obs_queue.qsize() > circular_buffer_size:
                        obs_queue.get()
            next_obs = obs
        NN_to_send = nn_queue.get_nowait() # nn_queue should always be non-empty
        time.sleep(.1)
        clear_queue(nn_queue)

if __name__ == '__main__':
    nn_queue = Queue()
    obs_queue = Queue()
    env = Pendulum.PendulumEnv(g=10.0, setpoint=0.0)
    Process(target=live_training, args=(env, nn_queue, obs_queue)).start()
    tranceive(env, nn_queue, obs_queue)
