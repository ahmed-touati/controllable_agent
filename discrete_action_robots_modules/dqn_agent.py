import torch
import os
from datetime import datetime
import numpy as np
import random
from discrete_action_robots_modules.replay_buffer import replay_buffer
from discrete_action_robots_modules.models import critic
from discrete_action_robots_modules.normalizer import normalizer
from her_modules.her import her_sampler
from discrete_action_robots_modules.robots import goal_distance
import csv
import pickle


class DQNAgent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
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
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        # create the dict for store the model
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
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_cycle):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            obs_norm_tensor = self._preproc_o(obs)
                            g_norm_tensor = self._preproc_g(g)
                            action = self.act_e_greedy(obs_norm_tensor, g_norm_tensor, update_eps=0.2)
                        # feed the actions into the environment
                        observation_new, _, _, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action)
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network()
                # soft update
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # start to do the evaluation
            # import pdb
            # pdb.set_trace()
            success_rate, dist = self._eval_agent()
            print('[{}] epoch is: {}, train success rate is: {:.3f},'
                  ' dist: {:.3f}'.format(datetime.now(), epoch, success_rate, dist))
            with open('{}/score_monitor.csv'.format(self.args.save_dir), "a") as monitor_file:
                monitor = csv.writer(monitor_file)
                monitor.writerow([epoch, success_rate, dist])
            # torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
            #             self.critic_network.state_dict()], \
            #            self.model_path + '/model.pt')

    # pre_process the inputs
    def _preproc_o(self, obs):
        obs_norm = self.o_norm.normalize(obs)
        obs_norm_tensor = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            obs_norm_tensor = obs_norm_tensor.cuda()
        return obs_norm_tensor

    def _preproc_g(self, g):
        g_norm = self.g_norm.normalize(g)
        g_norm_tensor = torch.tensor(g_norm, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            g_norm_tensor = g_norm_tensor.cuda()
        return g_norm_tensor

    # Acts based on single state (no batch)
    def act(self, obs, g):
        return self.critic_network(obs, g).data.max(1)[1][0]

    # Acts with an epsilon-greedy policy
    def act_e_greedy(self, obs, g, update_eps=0.2):
        return random.randrange(self.env_params['action']) if random.random() < update_eps else self.act(obs, g)

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], _ = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        # transfer them into the tensor
        obs_norm_tensor = torch.tensor(obs_norm, dtype=torch.float32)
        g_norm_tensor = torch.tensor(g_norm, dtype=torch.float32)
        obs_next_norm_tensor = torch.tensor(obs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.long)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        if self.args.cuda:
            obs_norm_tensor = obs_norm_tensor.cuda()
            g_norm_tensor = g_norm_tensor.cuda()
            obs_next_norm_tensor = obs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            q_next_value = self.critic_target_network(obs_next_norm_tensor, g_norm_tensor).max(1)[0].reshape(-1, 1)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(obs_norm_tensor, g_norm_tensor).gather(1, actions_tensor.reshape(-1, 1))
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
            per_success_rate = []
            per_dist = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            # for _ in range(self.env_params['max_timesteps']):
            for _ in range(25):
                with torch.no_grad():
                    obs_norm_tensor = self._preproc_o(obs)
                    g_norm_tensor = self._preproc_g(g)
                    action = self.act(obs_norm_tensor, g_norm_tensor)
                observation_new, _, _, info = self.env.step(action)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                dist = goal_distance(observation_new['achieved_goal'], observation_new['desired_goal'])
                # per_dist.append(dist)
                # per_success_rate.append(info['is_success'])
                per_dist = dist
                per_success_rate = info['is_success']
                if info['is_success'] > 0:
                    break
            total_success_rate.append(per_success_rate)
            total_dist.append(per_dist)
        total_success_rate = np.array(total_success_rate)
        avg_success_rate = np.mean(total_success_rate)
        total_dist = np.array(total_dist)
        avg_dist = np.mean(total_dist)
        return avg_success_rate, avg_dist
