import numpy as np
import os
from arguments import get_args
from continuous_world_modules.dqn_agent import DQNAgent
from continuous_world_modules.env import ContinuousWorld
from continuous_world_modules.geometry import Point
from continuous_world_modules.fb_agent import FBAgent

import random
import torch


def get_env_params(env):
    params = {'obs': 441,
              'goal': 441,
              'action': 5,
              }
    params['max_timesteps'] = 30
    return params


def launch(args):

    # set random seeds for reproduce
    # env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    env = ContinuousWorld(1, wall_pairs=[
        (Point(0.25, 0.0), Point(0.25, 0.4)),
        (Point(0.75, 1.0), Point(0.75, 0.6))],
                          movement_noise=0.01,
                          threshold_distance=0.05,
                          seed=args.seed)

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
