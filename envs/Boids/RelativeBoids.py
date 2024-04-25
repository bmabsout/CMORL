import Boids

import tensorflow as tf
import numpy as np
import gymnasium as gym

class RelativeBoids(gym.Env):
    def __init__(self):
        self.env = Boids.BoidsEnv()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def relative_obs(self):
        relative_obs = k_closest_relative_obs(self.env.state, self.env.numBoids)
        return np.concatenate([np.reshape(relative_obs["rel_pos"], -1), np.reshape(relative_obs["k_closest_vel"],-1)])

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

def multi_dim_reward(flat_state, flat_u, env: Boids.BoidsEnv):
    state = Boids.convert_state_to_dict(flat_state, env.numBoids)
    action = Boids.convert_action_to_dict(flat_u, env.numBoids)
    go_fast = tf.norm(state["vel"], axis=0)/env.max_speed
    # dists = flatten_upper_triangle(toroidal_pairwise_dist(state["pos"], state["pos"]))
    # minimize_distance = 1.0 - dists
    # avoid_collisions = tf.where(dists < 0.025, dists/0.025, 1.0)
    small_actions = 1.0 - tf.abs(action["angle_change"])/Boids.convert_action_to_dict(env.action_space.high, env.numBoids)["angle_change"]
    relative_obs = k_closest_relative_obs(flat_state, env.numBoids)
    rel_pos = relative_obs["rel_pos"]
    # k_closest_vel = relative_obs["k_closest_vel"]
    max_toroidal_distance = (0.5**2.0 + 0.5**2.0)**0.5 # toroidal distance means at worst we are (0.5, 0.5) away
    rel_distances = tf.reshape(tf.norm(rel_pos, axis=2), -1)/max_toroidal_distance
    minimize_distance = 1.0 - rel_distances
    avoid_collisions = tf.where(rel_distances < 0.1, (rel_distances/0.1)**2.0, 1.0)
    # print("go_fast", go_fast)
    # print("minimize_distance", minimize_distance)
    # print("avoid_collisions", avoid_collisions)
    # print("small_actions", small_actions)

    # print(rel_distances)
    return np.concatenate([go_fast, minimize_distance, avoid_collisions, small_actions])


@tf.function
def k_closest_relative_obs(flat_state: tf.Tensor, numBoids:int, k:int=3):
    state = Boids.convert_state_to_dict(flat_state, numBoids)
    # get k-closest boids
    tpos = tf.transpose(state["pos"])
    tvel = tf.transpose(state["vel"])
    dists = Boids.toroidal_pairwise_dist(state["pos"], state["pos"])
    # get the k closest boids
    k_closest = tf.argsort(dists, axis=1)[:,1:k+1]
    # get the k closest boid's positions and velocities
    k_closest_pos = tf.gather(tpos, k_closest)
    # print(k_closest_pos)
    k_closest_vel = tf.gather(tvel, k_closest)
    # # get the k closest boid's relative positions and velocities
    rel_pos = Boids.toroidal_difference(k_closest_pos, tf.expand_dims(tpos, 1))
    return {
        "rel_pos": rel_pos,
        "k_closest_vel": k_closest_vel,
    }