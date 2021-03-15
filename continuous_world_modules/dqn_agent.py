import torch
import os
from datetime import datetime
import numpy as np
import random
import pickle
import csv
from grid_modules.replay_buffer import ReplayBuffer, her_replay_buffer
from grid_modules.her import her_sampler
from discrete_action_robots_modules.models import critic
from continuous_world_modules.featurizer import RadialBasisFunction2D

"""
DQN agent

"""


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.max(np.abs(goal_a - goal_b), axis=-1)


class DQNAgent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        self.featuriser = RadialBasisFunction2D(1, 21, 0.05, cuda=args.cuda)
        # create the network
        self.critic_network = critic(env_params)
        # build up the target network
        self.critic_target_network = critic(env_params)
        # load the weights into the target networks
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.critic_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr)
        # her sampler
        compute_reward = lambda g_1, g_2: (goal_distance(g_1, g_2) <= env.threshold_distance).astype(np.float32)
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, compute_reward)
        # create the replay buffer
        self.buffer = her_replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # create the replay buffer
        # self.buffer = ReplayBuffer(self.args.buffer_size)

        if args.save_dir is not None:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)

            print(' ' * 26 + 'Options')
            for k, v in vars(self.args).items():
                print(' ' * 26 + k + ': ' + str(v))

            with open(self.args.save_dir + "/arguments.pkl", 'wb') as f:
                pickle.dump(self.args, f)

            with open('{}/score_monitor.csv'.format(self.args.save_dir), "wt") as monitor_file:
                monitor = csv.writer(monitor_file)
                monitor.writerow(['epoch', 'eval', 'dist'])

    def learn(self):
        """
        train the network

        """
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                mb_obs, mb_g, mb_actions = [], [], []
                for _ in range(self.args.num_rollouts_per_cycle):
                    # reset the rollouts
                    ep_obs, ep_g, ep_actions = [], [], []
                    # reset the environment
                    obs = self.env.reset()
                    g = self.env.goal
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            obs_tensor = self._preproc_o(obs)
                            g_tensor = self._preproc_g(g)
                            action = self.act_e_greedy(obs_tensor, g_tensor, update_eps=0.2)
                        # feed the actions into the environment
                        obs_new, reward, done, info = self.env.step(action)
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action)
                        obs = obs_new
                    ep_obs.append(obs.copy())
                    mb_obs.append(ep_obs)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_g, mb_actions])

                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network()
                # soft update
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # start to do the evaluation
            success_rate, dist = self._eval_agent()

            print('[{}] epoch is: {}, eval: {:.3f}, dist: {:.3f}'.format(datetime.now(), epoch, success_rate, dist))
            with open('{}/score_monitor.csv'.format(self.args.save_dir), "a") as monitor_file:
                monitor = csv.writer(monitor_file)
                monitor.writerow([epoch, success_rate, dist])
            torch.save(self.critic_network.state_dict(),
                       os.path.join(self.args.save_dir, 'model.pt'))

    # pre_process the inputs
    def _preproc_o(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            obs_tensor = obs_tensor.cuda()
        obs_tensor = self.featuriser.transform(obs_tensor)
        return obs_tensor

    def _preproc_g(self, g):
        g_tensor = torch.tensor(g, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            g_tensor = g_tensor.cuda()
        g_tensor = self.featuriser.transform(g_tensor)
        return g_tensor

    # Acts based on single state (no batch)
    def act(self, obs, g, target_network=False):
        if target_network:
            q = self.critic_target_network(obs, g)
        else:
            q = self.critic_network(obs, g)
        return q.max(1)[1]

    # Acts with an epsilon-greedy policy
    def act_e_greedy(self, obs, g, update_eps=0.2):
        return random.randrange(self.env_params['action']) if random.random() < update_eps else self.act(obs, g).item()

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def _hard_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)

        # transfer them into the tensor
        obs_tensor = torch.tensor(transitions['obs'], dtype=torch.float32)
        g_tensor = torch.tensor(transitions['g'], dtype=torch.float32)
        obs_next_tensor = torch.tensor(transitions['obs_next'], dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['action'], dtype=torch.long)
        r_tensor = torch.tensor(transitions['reward'], dtype=torch.float32)
        if self.args.cuda:
            obs_tensor = obs_tensor.cuda()
            g_tensor = g_tensor.cuda()
            obs_next_tensor = obs_next_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()

        obs_tensor = self.featuriser.transform(obs_tensor)
        g_tensor = self.featuriser.transform(g_tensor)
        obs_next_tensor = self.featuriser.transform(obs_next_tensor)
        # calculate the target Q value function
        with torch.no_grad():
            q_next_value = self.critic_target_network(obs_next_tensor, g_tensor).max(1)[0].reshape(-1, 1)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, 0, clip_return)
            # the q loss
        real_q_value = self.critic_network(obs_tensor, g_tensor).gather(1, actions_tensor.reshape(-1, 1))
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        total_dist = []
        for _ in range(self.args.n_test_rollouts):
            obs = self.env.reset()
            g = self.env.goal
            # self.env.set_initial_position(Point(0.2, 0.1))
            # self.env.set_goal(Point(0.9, 0.9))
            # obs = self.env.current_position
            # g = self.env.goal
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    obs_norm_tensor = self._preproc_o(obs)
                    g_norm_tensor = self._preproc_g(g)
                    action = self.act_e_greedy(obs_norm_tensor, g_norm_tensor, update_eps=0.02)
                obs, reward, _, info = self.env.step(action)
                if reward > 0:
                    break

            total_success_rate.append(reward)
            dist = goal_distance(obs, g)
            total_dist.append(dist)

        total_success_rate = np.array(total_success_rate)
        total_success_rate = np.mean(total_success_rate)

        total_dist = np.array(total_dist)
        total_dist = np.mean(total_dist)

        return total_success_rate, total_dist