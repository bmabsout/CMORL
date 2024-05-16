from dataclasses import dataclass
from functools import reduce
from pathlib import Path
import time
from argparse import Namespace

from cmorl.utils.args_utils import Arg_Serializer, Serialized_Argument, namespace_serializer

class HyperParams(Namespace):
    ac_kwargs      : dict[str, object]
    prev_folder    : None | Path
    seed           : int
    steps_per_epoch: int
    epochs         : int
    replay_size    : int
    gamma          : float
    polyak         : float
    pi_lr          : float
    q_lr           : float
    batch_size     : int
    start_steps    : int
    act_noise      : float
    max_ep_len     : int
    train_every    : int
    train_steps    : int
    p_Q_batch      : float
    p_Q_objectives : float
    env_args       : dict[str, object] = {}


descriptions: dict[str,  str] = {
    "ac_kwargs": "Any kwargs appropriate for the actor_critic function you provided",
    "prev_folder": "The folder to load the previous model from",
    "seed": "Seed for random number generators",
    "steps_per_epoch": "Number of steps of interaction (state-action pairs) for the agent and the environment in each epoch",
    "epochs": "Number of epochs to run and train agent",
    "replay_size": "Maximum length of replay buffer",
    "gamma": "Discount factor. (Always between 0 and 1)",
    "polyak": "Interpolation factor in polyak averaging for target networks. Target networks are updated towards main networks according to: theta_targ <- rho theta_targ + (1-rho) theta where rho is polyak. (Always between 0 and 1, usually close to 1)",
    "pi_lr": "Learning rate for policy",
    "q_lr": "Learning rate for Q-networks",
    "batch_size": "Minibatch size for SGD",
    "start_steps": "Number of steps for uniform-random action selection, before running real policy. Helps exploration",
    "act_noise": "Stddev for Gaussian exploration noise added to policy at training time. (At test time, no noise is added)",
    "max_ep_len": "Maximum length of an episode",
    "train_every": "Number of steps to wait before training",
    "train_steps": "Number of training steps to take",
    "p_Q_batch": "The p-value for composing the Q-values across the batch",
    "p_Q_objectives": "The p-value for composing the Q-values across the objectives"
}

abbreviations = {
    "prev_folder": "p",
    "seed": "s",
    "epochs": "e",
    "gamma": "g",
    "p_Q_batch": "p_b",
    "p_Q_objectives": "p_o"
}

def default_hypers():
    return HyperParams(
        ac_kwargs       = {},
        prev_folder     = None,
        seed            = int(time.time() * 1e5) % int(1e6),
        steps_per_epoch = 5000,
        epochs          = 100,
        replay_size     = int(1e6),
        gamma           = 0.9,
        polyak          = 0.995,
        pi_lr           = 1e-3,
        q_lr            = 1e-3,
        batch_size      = 100,
        start_steps     = 1000,
        act_noise       = 0.1,
        max_ep_len      = 400,
        train_every     = 50,
        train_steps     = 30,
        p_Q_batch       = 0.0,
        p_Q_objectives  = -4.0,
        env_args        = {},
    )

def combine(*hps: HyperParams):
    """Combine multiple hyperparams objects into one. The later objects override the earlier ones."""
    return reduce(lambda x, y: HyperParams(**{**vars(x), **vars(y)}), hps, HyperParams())


def rl_alg_serializer(experiment_name=None):
    return Arg_Serializer(
        Serialized_Argument(
            name="replay_save",
            abbrev="r",
            action="store_true",
            help="whether to save the replay buffer",
        ),
        Serialized_Argument(
            name="experiment_name",
            abbrev="n",
            type=str,
            required=True if experiment_name is None else False,
            default=experiment_name,
            help="name of the experiment"
        ),
        ignored={"replay_save", "experiment_name"}, # figure out a way to handle experiment_name, as it shouldn't be included int he x: part
    )


def default_serializer(hypers=HyperParams(), experiment_name=None):
    combined_hypers = combine(default_hypers(), hypers)
    return Arg_Serializer.join(
        namespace_serializer(combined_hypers, ignored={"seed"}, descriptions=descriptions, abbrevs=abbreviations),
        rl_alg_serializer(experiment_name=experiment_name),
    )
