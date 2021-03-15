import numpy as np
import os
from arguments import get_args
from grid_modules.dqn_agent import DQNAgent
from grid_modules.fb_agent import FBAgent
from grid_modules.gridworld.txt_utilities import get_char_matrix, build_gridworld_from_char_matrix
from grid_modules.gridworld import FOUR_ROOM_TXT, BIG_FOUR_ROOM_TXT
import random
import torch


def build_grid(gamma=0.99, seed=123, p_success=1.0):
    char_matrix = get_char_matrix(FOUR_ROOM_TXT)
    return build_gridworld_from_char_matrix(char_matrix, p_success=p_success, seed=seed, gamma=gamma)


def get_env_params(env):
    params = {'obs': env.state_space,
              'goal': env.state_space,
              'action': env.action_space,
              }
    params['max_timesteps'] = 50
    return params


def launch(args):

    env = build_grid(gamma=args.gamma, seed=args.seed)
    # set random seeds for reproduce
    # env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # get the environment parameters
    env_params = get_env_params(env)
    # create the agent to interact with the environment
    if args.agent == 'DQN':
        dqn_trainer = DQNAgent(args, env, env_params)
        dqn_trainer.learn()
    elif args.agent == 'FB':
        fb_trainer = FBAgent(args, env, env_params)
        fb_trainer.learn()
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    # get the params
    args = get_args()
    launch(args)
