import tensorflow as tf
import numpy as np
import Pendulum
import matplotlib.pyplot as plt
import pickle
import spinup
import time


def test(actor, env, seed=123, render=True):
    env.seed(seed)
    o = env.reset()
    high = env.action_space.high
    low = env.action_space.low
    os = []
    for _ in range(200):
        o, r, d, i, = env.step(actor(o)*(high - low)/2.0 + (high + low)/2.0)
        os.append(o)
        if render:
            env.render()
    return np.array(os)

# saved = tf.saved_model.load("left_leaning_pendulum/actor")
# saved = tf.saved_model.load("mid_leaning_actor")
# saved = tf.keras.models.load_model("sac_PendulumEnv_seed=0_steps_per_epoch=1000_epochs=200_gamma=0.9_lr=0.001_batch_size=100_start_steps=1000_update_after=900_update_every=50")
saved = tf.saved_model.load("pendulum/actor")
# saved = tf.keras.models.load_model("td3_saved")
# saved = tf.keras.models.load_model("td3_pendulum_left/actor")
# saved = tf.keras.models.load_model("td3_pendulum_anchored/actor")
# saved = tf.keras.models.load_model("sac_left_pendulum/actor/deterministic_actor")
# saved = tf.keras.models.load_model("sac_anchored_pendulum/actor/deterministic_actor")
# saved = tf.saved_model.load("pretty_please")
actor = lambda x: saved(np.array([x]))[0]
env = Pendulum.PendulumEnv(g=10., color=(0.0, 0.8,0.2))

# os = test(actor, env)

runs = np.array(list(map(lambda i: test(actor, env, seed=17+i, render=True)[:,1], range(5))))
# for i in range(5):
# 	os = 
# 	plt.plot(os)
plt.plot(runs.T, color="b")
plt.show()

with open("ddpg_left.p", "wb") as file:
    pickle.dump(runs, file)
