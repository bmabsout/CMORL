# adapted from https://github.com/tanzhenyu/spinup-tf2/blob/master/spinup/algos/ddpg/ddpg.py


# This script needs these libraries to be installed:
#   tensorflow, numpy
import time
from typing import Callable
import numpy as np
import tensorflow as tf # type: ignore
import gymnasium as gym
import keras # type: ignore
import signal
import gymnasium.utils.seeding as seeding
import wandb
from cmorl.rl_algs.ddpg import core
from cmorl.rl_algs.ddpg.hyperparams import HyperParams, default_hypers
from cmorl.utils import reward_utils
from cmorl.utils.logx import TensorflowLogger
from cmorl.utils.loss_composition import (
    geo,
    move_towards_range,
    p_mean,
    scale_gradient,
)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size, rwds_dim=1):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, rwds_dim], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, np_random = np.random):
        idxs = np_random.integers(0, self.size, size=batch_size)
        return dict(
            obs1=self.obs1_buf[idxs],
            obs2=self.obs2_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

"""

Deep Deterministic Policy Gradient (DDPG)

"""
def ddpg(
    env_fn: Callable[[], gym.Env[gym.spaces.Box, gym.spaces.Box]],
    env_name: str | None = None,
    experiment_name: str = str(time.time()),
    hp: HyperParams = default_hypers(),
    actor_critic=core.mlp_actor_critic,
    logger_kwargs=dict(),
    save_freq=1,
    on_save=lambda *_: (),
    experiment_description: str | None = None,
    cmorl: None | reward_utils.CMORL = None,
):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols
            for state, ``x_ph``, and action, ``a_ph``, and returns the main
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q``        (batch,)          | Gives the current estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q_pi``     (batch,)          | Gives the composition of ``q`` and
                                           | ``pi`` for states in ``x_ph``:
                                           | q(x, pi(x)).
            ===========  ================  ======================================

        hp (HyperParams): The hyperparameters for the experiment.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        on_save (callable): A function that is called after the model is saved.

        experiment_description (str): A description of the experiment.

        cmorl (CMORL): A class that defines the reward function and the q-composer.
    """
    # start a new wandb run to track this script

    logger = TensorflowLogger(**logger_kwargs)
    # logger.save_config({"hyperparams": hp.__dict__, "extra_hyperparams": extra_hyperparameters})

    tf.random.set_seed(hp.seed)
    np_random, _ = seeding.np_random(hp.seed)

    env = env_fn()
    o, info = env.reset(seed=hp.seed)
    q_composer = reward_utils.default_q_composer if cmorl is None else cmorl.q_composer

    weights_and_biases = wandb.init(
        # set the wandb project where this run will be logged
        project=env_name,
        # track hyperparameters and run metadata
        entity="cmorl",
        config=hp.__dict__,
        # name the run
        name=experiment_name,
        # write a description of the run
        notes=experiment_description,
    )

    def exited_gracefully(*args, **kwargs):
        weights_and_biases.finish()
        exit(0)

    signal.signal(signal.SIGINT, exited_gracefully)
    signal.signal(signal.SIGTERM, exited_gracefully)
    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    assert isinstance(
        env.observation_space, gym.spaces.Box
    ), "only continuous action space is supported"
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    rew_dims = cmorl.calculate_space(env).shape[0] if cmorl else 1
    max_discounted_sum = np.zeros((rew_dims)) + (1.0 / (1.0 - hp.gamma))

    # Main outputs from computation graph
    with tf.name_scope("main"):
        pi_network, q_network = actor_critic(
            env.observation_space, env.action_space, rew_dims, seed=hp.seed, **hp.ac_kwargs
        )

        pi_and_before_clip = keras.Model(
            pi_network.input,
            {"pi": pi_network.output, "before_clip": pi_network.layers[-2].output},
        )
        pi_and_before_clip.compile()
        # q_and_before_clip = keras.Model(
        #     q_network.input, {"q": q_network.output, "before_clip": q_network.layers[-2].output})
        q_and_before_clip = keras.Model(
            q_network.input,
            {"q": q_network.output, "before_clip": q_network.layers[-2].output},
        )
        q_and_before_clip.compile()
    # Target networks
    with tf.name_scope("target"):
        # Note that the action placeholder going to actor_critic here is
        # irrelevant, because we only need q_targ(s, pi_targ(s)).
        pi_targ_network, q_targ_network = actor_critic(
            env.observation_space, env.action_space, rew_dims, **hp.ac_kwargs
        )

    # make sure network and target network is using the same weights
    pi_targ_network.set_weights(pi_network.get_weights())
    q_targ_network.set_weights(q_network.get_weights())

    # Experience buffer
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=hp.replay_size, rwds_dim=rew_dims
    )
    # Separate train ops for pi, q
    pi_optimizer = keras.optimizers.Adam(learning_rate=hp.pi_lr)
    q_optimizer = keras.optimizers.Adam(learning_rate=hp.q_lr)

    # Polyak averaging for target variables
    @tf.function
    def target_update():
        for v_main, v_targ in zip(
            pi_network.trainable_variables, pi_targ_network.trainable_variables
        ):
            v_targ.assign(hp.polyak * v_targ + (1 - hp.polyak) * v_main)
        for v_main, v_targ in zip(
            q_network.trainable_variables, q_targ_network.trainable_variables
        ):
            v_targ.assign(hp.polyak * v_targ + (1 - hp.polyak) * v_main)

    @tf.function
    def q_update(obs1, obs2, acts, rews, dones):
        with tf.GradientTape() as tape:
            # q_squeezed = tf.squeeze(q, axis=1)
            outputs = q_and_before_clip(tf.concat([obs1, acts], axis=-1))
            q = outputs["q"]

            # tf.print(before_clip)
            # tf.print(outputs["before_clip"])
            pi_targ = pi_targ_network(obs2)
            q_pi_targ = q_targ_network(tf.concat([obs2, pi_targ], axis=-1))

            batch_size = tf.shape(dones)[0]

            dones = tf.broadcast_to(tf.expand_dims(dones, -1), (batch_size, rew_dims))

            backup = tf.stop_gradient(
                rews / max_discounted_sum + (1 - dones) * hp.gamma * q_pi_targ
            )
            keep_in_range = p_mean(
                move_towards_range(outputs["before_clip"], 0.0, 1.0), p=-1.0
            )
            q_bellman_c = 1.0 - p_mean(tf.abs(q - backup), p=2.0)
            with_reg = p_mean(tf.stack([q_bellman_c, keep_in_range]), p=0.0)
            # tf.print(q_bellman_c)
            # tf.print(with_reg)
            # tf.print(before_clip)
            q_loss = 1.0 - with_reg

            # q_loss = tf.reduce_mean((q - backup) ** 2) + before_clip
            # q_loss = p_mean(q_loss, p=2)

        grads = tape.gradient(q_loss, q_network.trainable_variables)
        grads_and_vars = zip(grads, q_network.trainable_variables)
        q_optimizer.apply_gradients(grads_and_vars)
        return q_bellman_c, keep_in_range

    @tf.function
    def pi_update(obs1, obs2, debug=False):
        with tf.GradientTape() as tape:
            outputs = pi_and_before_clip(obs1)
            pi = outputs["pi"]
            before_clip = outputs["before_clip"]
            before_clip_c = p_mean(move_towards_range(before_clip, -1.0, 1.0), p=-1.0)
            q_values = q_network(tf.concat([obs1, pi], axis=-1))
            qs_c, q_c = q_composer(
                q_values, p_batch=hp.p_Q_batch, p_objectives=hp.p_Q_objectives
            )
            all_c = p_mean([q_c, before_clip_c], p=0.0)
            pi_loss = 1.0 - all_c
        grads = tape.gradient(pi_loss, pi_network.trainable_variables)
        # if debug:
        #     tf.print(sum(map(lambda x: tf.reduce_mean(x**2.0), grads)))

        grads_and_vars = zip(grads, pi_network.trainable_variables)
        pi_optimizer.apply_gradients(grads_and_vars)
        return all_c, qs_c, q_c, before_clip_c

    def get_action(o, noise_scale, np_random: np.random.Generator = np.random):
        minus_1_to_1 = pi_network(tf.reshape(o, [1, -1])).numpy()[0]
        noise = noise_scale * np_random.normal(size=act_dim).astype(np.float32)
        a = (minus_1_to_1 + noise) * (
            env.action_space.high - env.action_space.low
        ) / 2.0 + (env.action_space.high + env.action_space.low) / 2.0
        return np.clip(a, env.action_space.low, env.action_space.high)

    def test_agent(n=1):
        print("testing agents")
        sum_step_return = 0
        for j in range(n):
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            while not (d or (ep_len == hp.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _, _ = env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
                env.render()
            # logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            sum_step_return += ep_ret / ep_len
        return sum_step_return / n

    start_time = time.time()
    o, info = env.reset()
    done, ep_ret, ep_len = False, 0.0, 0
    total_steps = hp.steps_per_epoch * hp.epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards,
        use the learned policy (with some noise, via act_noise).
        """
        if t > hp.start_steps:
            action = get_action(o, hp.act_noise * (total_steps - t) / total_steps, np_random)
        else:
            env.action_space._np_random = np_random
            action = env.action_space.sample()

        if np.isnan(action).any():
            # a = env.action_space.sample()
            print(f"nan detected in action {action}")
            # Log the occurrence in Weights and Biases
            weights_and_biases.log({"message": "NaN detected in action"}, step=t)
            # exit(1)
            weights_and_biases.finish()
            return

        # Step the env
        o2, reward, done, truncated, info = env.step(action)
        
        if cmorl:
            reward = cmorl(reward_utils.Transition(o, action, o2, done, info), env) # getting the reward before the step, since the reward is based on the previous state

        ep_ret += reward
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        truncated = truncated or ep_len == hp.max_ep_len
        done = done and not truncated

        # Store experience to replay buffer
        replay_buffer.store(o, action, reward, o2, done)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if t > hp.start_steps and t % hp.train_every == 0:
            """
            Perform all DDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """
            for train_step in range(hp.train_steps):
                batch = replay_buffer.sample_batch(hp.batch_size, np_random=np_random)
                obs1 = tf.constant(batch["obs1"])
                obs2 = tf.constant(batch["obs2"])
                acts = tf.constant(batch["acts"])
                rews = tf.constant(batch["rews"])
                dones = tf.constant(batch["done"])
                # Q-learning update
                qloss_c, q_before_clip_c = q_update(obs1, obs2, acts, rews, dones)
                logger.store(LossQ=1.0 - qloss_c)
                weights_and_biases.log({"Q-Loss": 1.0 - qloss_c}, step=t)
                logger.store(Q_before_clip_c=q_before_clip_c)
                weights_and_biases.log({"Q-before_clip_c": q_before_clip_c}, step=t)
                # Policy update
                (
                    all_c,
                    qs_c,
                    q_c,
                    before_clip_c,
                ) = pi_update(obs1, obs2, (train_step + 1) % 20 == 0)

                qs_c = qs_c.numpy()
                logger.store(
                    Q_comp=q_c,
                    before_clip=before_clip_c,
                )
                weights_and_biases.log(
                    {"Q-composed": q_c, "before_clip": before_clip_c},
                    step=t
                )
                qs_dict_ = {}
                for i, q in enumerate(qs_c):
                    qs_dict_[f"Q{i}"] = q
                logger.store(**qs_dict_)
                weights_and_biases.log(qs_dict_, step=t)

                # target update
                target_update()

        if done or truncated:
            # print(ep_ret)
            ret_dict_ = {}
            for i in range(rew_dims):
                ret_dict_[f"EpRet_{i}"] = ep_ret[i]
            logger.store(**ret_dict_)
            weights_and_biases.log(ret_dict_, step=t)
            logger.store(EpLen=ep_len)
            weights_and_biases.log({"EpLen": ep_len}, step=t)
            o, i = env.reset()
            reward, done, ep_ret, ep_len = 0, False, 0, 0

        # End of epoch wrap-up
        if t > hp.start_steps and t % hp.steps_per_epoch == 0:
            print(hp.steps_per_epoch)
            epoch = t // hp.steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == hp.epochs - 1):
                on_save(pi_network, q_network, epoch // save_freq, replay_buffer)

            # Test the performance of the deterministic version of the agent.
            # test_agent()

            # Log info about epoch
            logger.log_tabular("Epoch", epoch)
            weights_and_biases.log({"Epoch": epoch}, step=t)
            logger.log_tabular("EpLen", average_only=True)
            for i in range(rew_dims):
                logger.log_tabular(f"EpRet_{i}", average_only=True)
            logger.log_tabular("Time", time.time() - start_time)
            logger.log_tabular("TotalEnvInteracts", t)
            # logger.log_tabular("before_clip", average_only=True)
            # logger.log_tabular("Qs", average_only=True)
            for i in range(qs_c.shape[0]):
                logger.log_tabular(f"Q{i}", average_only=True)
            logger.log_tabular("Q_comp", average_only=True)
            # logger.log_tabular("All", average_only=True)
            logger.log_tabular("LossQ", average_only=True)
            logger.dump_tabular(epoch)

    # [optional] finish the wandb run, necessary in notebooks
    weights_and_biases.finish()

    return pi_network