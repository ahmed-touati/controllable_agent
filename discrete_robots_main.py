import numpy as np
from arguments import get_args
from discrete_action_robots_modules.dqn_agent import DQNAgent
from discrete_action_robots_modules.fb_agent import FBAgent
from discrete_action_robots_modules.robots import FetchReach
import random
import torch


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.num_actions,
            }
    params['max_timesteps'] = env._max_episode_steps
    return params


def launch(args):
    env = FetchReach()
    # import pdb
    # pdb.set_trace()
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
