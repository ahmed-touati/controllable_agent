import numpy as np
from arguments import get_args
from atari_modules.dqn_agent import dqn_agent
from atari_modules.fb_agent import FBAgent
from atari_modules.her_dqn_agent import HerDQNAgent
from atari_modules.wrappers import make_goalPacman
import random
import torch

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""


def get_env_params(env):
    params = {'obs': env.observation_space['observation'].shape,
              'goal': 2,
              'action': env.action_space.n,
              }
    params['max_timesteps'] = 50
    return params


def launch(args):

    env = make_goalPacman()
    # set random seeds for reproduce
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # get the environment parameters
    env_params = get_env_params(env)
    # create the agent to interact with the environment
    if args.agent == 'DQN':
        dqn_trainer = dqn_agent(args, env, env_params)
        dqn_trainer.learn()
    elif args.agent == 'FB':
        fb_trainer = FBAgent(args, env, env_params)
        fb_trainer.learn()
    elif args.agent == 'HerDQN':
        her_agent = HerDQNAgent(args, env, env_params)
        her_agent.learn()
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    # get the params
    args = get_args()
    launch(args)
