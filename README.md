# CMORL

Composition for Multi-Objective Reinforcement Learning

# Installation 
## Via nix
* Install `nix` from https://determinate.systems/posts/determinate-nix-installer
* Run `nix develop --impure`` in the repo's root, this should take some time but then you will be dropped in a bash shell with everything required to run training

## Via Conda
* `conda create -f environment.yml`
* `conda activate cmorl_env`

# Run training
`python envs/Pendulum/train_pendulum.py`, this uses ddpg to train the algorithm and produces automatically checkpoints and logs in the trained folder
`python envs/Pendulum/test_pendulum.py -lr training`


