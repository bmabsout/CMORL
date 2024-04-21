# adapted from https://github.com/tanzhenyu/spinup-tf2/blob/master/spinup/algos/ddpg/ddpg.py


# This script needs these libraries to be installed:
#   tensorflow, numpy
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict
import numpy as np
import tensorflow as tf
import gymnasium as gym
import keras
import wandb
from cmorl.rl_algs.ddpg import core
from cmorl.utils.logx import TensorflowLogger
from cmorl.utils.loss_composition import (
    geo,
    p_mean,
    scale_gradient,
    move_toward_zero,
    sigmoid_regularizer,
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

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs1=self.obs1_buf[idxs],
            obs2=self.obs2_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )


class HyperParams:
    def __init__(
        self,
        ac_kwargs={"actor_hidden_sizes": (32, 32), "critic_hidden_sizes": (512, 512)},
        seed=int(time.time() * 1e5) % int(1e6),
        steps_per_epoch=5000,
        epochs=100,
        replay_size=int(1e6),
        gamma=0.9,
        polyak=0.995,
        pi_lr=1e-4,
        q_lr=1e-4,
        batch_size=100,
        start_steps=10000,
        act_noise=0.1,
        max_ep_len=1000,
        train_every=50,
        train_steps=30,
    ):
        self.ac_kwargs = ac_kwargs
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.act_noise = act_noise
        self.max_ep_len = max_ep_len
        self.train_every = train_every
        self.train_steps = train_steps


"""

Deep Deterministic Policy Gradient (DDPG)

"""


def ddpg(
    env_fn: Callable[[], gym.Env],
    hp: HyperParams = HyperParams(),
    actor_critic=core.mlp_actor_critic,
    logger_kwargs=dict(),
    save_freq=1,
    on_save=lambda *_: (),
    run_name: str = None,
    run_description: str = None,
    extra_hyperparameters: dict[str, object] = {},
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

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    # start a new wandb run to track this script

    logger = TensorflowLogger(**logger_kwargs)
    # logger.save_config({"hyperparams": hp.__dict__, "extra_hyperparams": extra_hyperparameters})

    tf.random.set_seed(hp.seed)
    np.random.seed(hp.seed)

    env = env_fn()

    weights_and_biases = wandb.init(
        # set the wandb project where this run will be logged
        project=type(env).__name__,
        # track hyperparameters and run metadata
        config=hp.__dict__,
        # name the run
        name=(
            run_name if run_name is not None else f"{type(env).__name__}-{time.time()}"
        ),
        # write a description of the run
        notes=run_description,
    )

    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    assert isinstance(
        env.observation_space, gym.spaces.Box
    ), "only continuous action space is supported"
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    rew_dims = env.cmorl.dim
    max_q_val = np.zeros((rew_dims)) + (1.0 / (1.0 - hp.gamma))

    # Main outputs from computation graph
    with tf.name_scope("main"):
        pi_network, q_network = actor_critic(
            env.observation_space, env.action_space, rew_dims, **hp.ac_kwargs
        )

        pi_and_before_tanh = keras.Model(
            pi_network.input,
            {"pi": pi_network.output, "before_tanh": pi_network.layers[-2].output},
        )
        pi_and_before_tanh.compile()
        # q_and_before_sigmoid = keras.Model(
        #     q_network.input, {"q": q_network.output, "before_sigmoid": q_network.layers[-2].output})
        q_and_before_sigmoid = keras.Model(
            q_network.input,
            {"q": q_network.output, "before_sigmoid": q_network.layers[-2].output},
        )
        q_and_before_sigmoid.compile()
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
            outputs = q_and_before_sigmoid(tf.concat([obs1, acts], axis=-1))
            q = outputs["q"]
            before_sigmoid = outputs["before_sigmoid"]

            before_sigmoid = tf.reduce_mean(sigmoid_regularizer(before_sigmoid))
            pi_targ = pi_targ_network(obs2)
            q_pi_targ = q_targ_network(tf.concat([obs2, pi_targ], axis=-1))

            batch_size = tf.shape(dones)[0]

            # # calculate q_loss for every single reward by looping through them
            # dones_i = dones
            # dones_i = tf.broadcast_to(tf.expand_dims(dones_i, -1), (batch_size, 1))
            # # split rews and q_pi_targ along the last axis
            # rews_i = rews
            # rews_i = tf.split(rews_i, rew_dims, axis=-1)
            # q_pi_targ_i = q_pi_targ
            # q_pi_targ_i = tf.split(q_pi_targ_i, rew_dims, axis=-1)
            # # create a dict to store the loss for each reward
            # q_loss_dic = {}
            # for i in range(rew_dims):
            #     backup_i = tf.stop_gradient(
            #         rews_i[i] / max_q_val[i] + (1 - dones_i) * hp.gamma * q_pi_targ_i[i]
            #     )
            #     q_loss_i = p_mean(1 - tf.abs(q[:, i] - backup_i), p=0.0)
            #     q_loss_dic[f"Q-loss_{i}"] = q_loss_i

            dones = tf.broadcast_to(tf.expand_dims(dones, -1), (batch_size, rew_dims))

            backup = tf.stop_gradient(
                rews / max_q_val + (1 - dones) * hp.gamma * q_pi_targ
            )
            qc_losses = p_mean(
                1.0 - tf.abs(q - backup),
                p=-4.0,
                axis=0,
            )  # -before_tanh_c*1e-5

            # q_loss_dic = {}
            # tf.print(qc_losses)
            # for i in range(tf.shape(qc_losses)[0]):
            #     q_loss_dic[f"Q-loss_{i}"] = qc_losses[i]

            qc_loss = 1.0 - p_mean(qc_losses, -4.0)

            # q_loss = tf.reduce_mean((q - backup) ** 2) + before_sigmoid
            # q_loss = p_mean(q_loss, p=2)

        grads = tape.gradient(qc_loss, q_network.trainable_variables)
        grads_and_vars = zip(grads, q_network.trainable_variables)
        q_optimizer.apply_gradients(grads_and_vars)
        return qc_loss, qc_losses

    @tf.function
    def pi_update(obs1, obs2, debug=False):
        with tf.GradientTape() as tape:
            outputs = pi_and_before_tanh(obs1)
            pi = outputs["pi"]
            before_tanh = outputs["before_tanh"]
            before_tanh_c = 1.0 - tf.reduce_mean(
                tf.reshape(1.0 - move_toward_zero(before_tanh), [1, -1]) ** 2.0
            )
            #     [tf.reduce_mean(q_network(tf.concat([obs1, pi], axis=-1)))])
            q_values = q_network(tf.concat([obs1, pi], axis=-1))
            qs_c, q_c = env.cmorl.q_composer(q_values)
            all_c = p_mean(
                tf.stack(
                    [
                        scale_gradient(tf.squeeze(q_c), 3e2),
                        scale_gradient(tf.squeeze(before_tanh_c), 0.1),
                    ]
                ),
                p=0.0,
            )
            # print(qs_c.shape, q_c.shape, all_c.shape)
            pi_loss = 1.0 - all_c
        grads = tape.gradient(pi_loss, pi_network.trainable_variables)
        # if debug:
        #     tf.print(sum(map(lambda x: tf.reduce_mean(x**2.0), grads)))
        grads_and_vars = zip(grads, pi_network.trainable_variables)
        pi_optimizer.apply_gradients(grads_and_vars)
        return all_c, qs_c, q_c, before_tanh_c

    def get_action(o, noise_scale):
        minus_1_to_1 = pi_network(tf.constant(o.reshape(1, -1))).numpy()[0]
        noise = noise_scale * np.random.randn(act_dim)
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
    o, _ = env.reset()
    d, ep_ret, ep_len = False, 0.0, 0
    total_steps = hp.steps_per_epoch * hp.epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards,
        use the learned policy (with some noise, via act_noise).
        """
        if t > hp.start_steps:

            a = get_action(o, hp.act_noise * (total_steps - t) / total_steps)
        else:
            a = env.action_space.sample()

        if np.isnan(a).any():
            # a = env.action_space.sample()
            print("nan detected")
            exit(1)

        # Step the env
        o2, r, d, _, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == hp.max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if t > hp.start_steps and t % hp.train_every == 0:
            """
            Perform all DDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """
            for train_step in range(hp.train_steps):
                batch = replay_buffer.sample_batch(hp.batch_size)
                # print(1, batch["obs1"])
                obs1 = tf.constant(batch["obs1"])
                # print(2, obs1)
                obs2 = tf.constant(batch["obs2"])
                acts = tf.constant(batch["acts"])
                rews = tf.constant(batch["rews"])
                dones = tf.constant(batch["done"])
                # Q-learning update
                loss_q, q_losses = q_update(obs1, obs2, acts, rews, dones)
                logger.store(LossQ=loss_q)
                weights_and_biases.log({"Q-Loss": loss_q})
                # print(loss_q)
                # Policy update
                (
                    all_c,
                    qs_c,
                    q_c,
                    before_tanh_c,
                ) = pi_update(obs1, obs2, (train_step + 1) % 20 == 0)

                qs_c = qs_c.numpy()
                logger.store(
                    # All=all_c,
                    Q_comp=q_c,
                    Before_tanh=before_tanh_c,
                )
                weights_and_biases.log(
                    {"Q-composed": q_c, "Before_tanh": before_tanh_c}
                )
                qs_dict_ = {}
                for i, q in enumerate(qs_c):
                    qs_dict_[f"Q{i}"] = q
                logger.store(**qs_dict_)
                weights_and_biases.log(qs_dict_)

                # target update
                target_update()

        if d or (ep_len == hp.max_ep_len):
            # print(ep_ret)
            ret_dict_ = {}
            for i in range(rew_dims):
                ret_dict_[f"EpRet_{i}"] = ep_ret[i]
            logger.store(**ret_dict_)
            weights_and_biases.log(ret_dict_)
            logger.store(EpLen=ep_len)
            weights_and_biases.log({"EpLen": ep_len})
            o, i = env.reset()
            r, d, ep_ret, ep_len = 0, False, 0, 0

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
            # logger.log_tabular("Epoch", epoch)
            logger.log_tabular("Episode", epoch)
            weights_and_biases.log({"Episode": epoch})
            # logger.log_tabular("EpRet", average_only=True)
            logger.log_tabular("EpLen", average_only=True)
            for i in range(rew_dims):
                logger.log_tabular(f"EpRet_{i}", average_only=True)
            logger.log_tabular("Time", time.time() - start_time)
            logger.log_tabular("TotalEnvInteracts", t)
            # logger.log_tabular("Before_tanh", average_only=True)
            # logger.log_tabular("Qs", average_only=True)
            for i in range(rew_dims):
                logger.log_tabular(f"Q{i}", average_only=True)
            logger.log_tabular("Q_comp", average_only=True)
            # logger.log_tabular("All", average_only=True)
            logger.log_tabular("LossQ", average_only=True)
            logger.dump_tabular(epoch)

            weights_and_biases.log(
                {f"LossQ[{i}]": q_loss for i, q_loss in enumerate(q_losses)}
            )

    # [optional] finish the wandb run, necessary in notebooks
    weights_and_biases.finish()

    return pi_network


def parse_args_and_train(args=None):
    import cmorl.utils.train_utils as train_utils
    import cmorl.utils.args_utils as args_utils

    serializer = args_utils.Arg_Serializer.join(
        args_utils.Arg_Serializer(
            {
                "g": args_utils.Serialized_Argument(
                    name="--gym_env", type=str, required=True
                )
            },
            ignored={"gym_env"},
        ),
        args_utils.default_serializer(),
    )
    cmd_args = args_utils.parse_arguments(serializer)
    hp = HyperParams(
        q_lr=cmd_args.learning_rate, pi_lr=cmd_args.learning_rate, seed=cmd_args.seed
    )
    generated_params = train_utils.create_train_folder_and_params(
        hp, cmd_args, serializer
    )
    env_fn = lambda: gym.make(cmd_args.gym_env)
    ddpg(env_fn, **generated_params)


if __name__ == "__main__":
    parse_args_and_train()
