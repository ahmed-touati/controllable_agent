import torch
import os
from datetime import datetime
import numpy as np
import random
import pickle
import csv
from discrete_action_robots_modules.replay_buffer import replay_buffer
from discrete_action_robots_modules.models import ForwardMap, BackwardMap
from her_modules.her import her_sampler
from discrete_action_robots_modules.robots import goal_distance
from grid_modules.mdp_utils import extract_policy
from torch.distributions.cauchy import Cauchy

"""
FB agent with HER (MPI-version)

"""


class FBAgent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        self.cauchy = Cauchy(torch.tensor([0.0]), torch.tensor([0.5]))
        # create the network
        self.forward_network = ForwardMap(env_params, args.embed_dim)
        self.backward_network = BackwardMap(env_params, args.embed_dim)
        # build up the target network
        self.forward_target_network = ForwardMap(env_params, args.embed_dim)
        self.backward_target_network = BackwardMap(env_params, args.embed_dim)
        # load the weights into the target networks
        self.forward_target_network.load_state_dict(self.forward_network.state_dict())
        self.backward_target_network.load_state_dict(self.backward_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.forward_network.cuda()
            self.backward_network.cuda()
            self.forward_target_network.cuda()
            self.backward_target_network.cuda()
        # create the optimizer
        f_params = [param for param in self.forward_network.parameters()]
        b_params = [param for param in self.backward_network.parameters()]
        self.fb_optim = torch.optim.Adam(f_params + b_params, lr=self.args.lr)
        # self.backward_optim = torch.optim.Adam(self.backward_network.parameters(), lr=self.args.lr_backward)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)

        if args.save_dir is not None:
            # create the dict for store the model
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)

            print(' ' * 26 + 'Options')
            for k, v in vars(self.args).items():
                print(' ' * 26 + k + ': ' + str(v))

            with open(self.args.save_dir + "/arguments.pkl", 'wb') as f:
                pickle.dump(self.args, f)

            with open('{}/score_monitor.csv'.format(self.args.save_dir), "wt") as monitor_file:
                monitor = csv.writer(monitor_file)
                monitor.writerow(['epoch', 'eval', 'avg dist', 'eval (GPI)', 'avg dist (GPI)'])

    def learn(self):
        """
        train the network

        """
        # start to collect samples
        # print('MPI SIZE: ', MPI.COMM_WORLD.Get_size())
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
                    if self.args.w_sampling == 'goal_oriented':
                        g_tensor = self._preproc_g(g)
                        with torch.no_grad():
                            w = self.backward_network(g_tensor)
                    elif self.args.w_sampling == 'uniform_ball':
                        w = self.sample_uniform_ball(1)
                    elif self.args.w_sampling == 'cauchy_ball':
                        w = self.sample_cauchy_ball(1)

                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            obs_tensor = self._preproc_o(obs)
                            action = self.act_e_greedy(obs_tensor, w, update_eps=0.2)
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
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network()
                # soft update
                self._soft_update_target_network(self.forward_target_network, self.forward_network)
                self._soft_update_target_network(self.backward_target_network, self.backward_network)
            # start to do the evaluation
            success_rate, avg_dist = self._eval_agent()
            success_rate_gpi, avg_dist_gpi = self._eval_gpi_agent(num_gpi=self.args.num_gpi)
            print('[{}] epoch is: {}, eval: {:.3f}, avg_dist : {:.3f}, '
                  'eval (GPI): {:.3f}, avg_dist (GPI): {:.3f}'.format(datetime.now(), epoch, success_rate, avg_dist,
                                                                      success_rate_gpi, avg_dist_gpi))
            with open('{}/score_monitor.csv'.format(self.args.save_dir), "a") as monitor_file:
                monitor = csv.writer(monitor_file)
                monitor.writerow([epoch, success_rate, avg_dist, success_rate_gpi, avg_dist_gpi])
            torch.save([self.forward_network.state_dict(), self.backward_network.state_dict()],
                       os.path.join(self.args.save_dir, 'model.pt'))

    def sample_uniform_ball(self, n, eps=1e-10):
        gaussian_rdv = torch.FloatTensor(n, self.args.embed_dim).normal_(mean=0, std=1)
        gaussian_rdv /= torch.norm(gaussian_rdv, dim=-1, keepdim=True) + eps
        uniform_rdv = torch.FloatTensor(n, 1).uniform_()
        w = np.sqrt(self.args.embed_dim) * gaussian_rdv * uniform_rdv
        if self.args.cuda:
            w = w.cuda()
        return w

    def sample_cauchy_ball(self, n, eps=1e-10):
        gaussian_rdv = torch.FloatTensor(n, self.args.embed_dim).normal_(mean=0, std=1)
        gaussian_rdv /= torch.norm(gaussian_rdv, dim=-1, keepdim=True) + eps
        cauchy_rdv = self.cauchy.sample((n, ))
        w = np.sqrt(self.args.embed_dim) * gaussian_rdv * cauchy_rdv
        if self.args.cuda:
            w = w.cuda()
        return w

    # pre_process the inputs
    def _preproc_o(self, obs):
        obs = self._clip(obs)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            obs_tensor = obs_tensor.cuda()
        return obs_tensor

    def _preproc_g(self, g):
        g = self._clip(g)
        g_tensor = torch.tensor(g, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            g_tensor = g_tensor.cuda()
        return g_tensor

    def act_gpi(self, obs, w_train, w_eval):
        # import pdb
        # pdb.set_trace()
        num_gpi = w_train.shape[0]
        obs_repeat = obs.repeat(num_gpi, 1)
        w_eval_repeat = w_eval.repeat(num_gpi, 1)
        f = self.forward_network(obs_repeat, w_train)
        z = torch.einsum('sda, sd -> sa', f, w_eval_repeat).max(0)[0]
        return z.max(0)[1]

    # Acts based on single state (no batch)
    def act(self, obs, w, target_network=False):
        if target_network:
            f = self.forward_target_network(obs, w)
        else:
            f = self.forward_network(obs, w)
        z = torch.einsum('sda, sd -> sa', f, w)
        return z.max(1)[1]

    def get_policy(self, obs, w, policy_type='boltzmann', temp=1, eps=0.01, target_network=False):
        if target_network:
            f = self.forward_target_network(obs, w)
        else:
            f = self.forward_network(obs, w)
        z = torch.einsum('sda, sd -> sa', f, w)
        return extract_policy(z, policy_type=policy_type, temp=temp, eps=eps)

    # Acts with an epsilon-greedy policy
    def act_e_greedy(self, obs, g, update_eps=0.2):
        return random.randrange(self.env_params['action']) if random.random() < update_eps else self.act(obs, g).item()

    def _clip(self, o):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        return o

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        other_transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g, ag = transitions['obs'], transitions['obs_next'], transitions['g'], transitions['ag']
        transitions['obs'], transitions['g'] = self._clip(o), self._clip(g)
        transitions['obs_next'] = self._clip(o_next)
        transitions['ag'] = self._clip(ag)
        other_transitions['ag'] = self._clip(other_transitions['ag'])

        # transfer them into the tensor
        obs_tensor = torch.tensor(transitions['obs'], dtype=torch.float32)
        g_tensor = torch.tensor(transitions['g'], dtype=torch.float32)
        obs_next_tensor = torch.tensor(transitions['obs_next'], dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.long)
        ag_tensor = torch.tensor(transitions['ag'], dtype=torch.float32)
        ag_other_tensor = torch.tensor(other_transitions['ag'], dtype=torch.float32)
        if self.args.cuda:
            obs_tensor = obs_tensor.cuda()
            g_tensor = g_tensor.cuda()
            obs_next_tensor = obs_next_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            ag_tensor = ag_tensor.cuda()
            ag_other_tensor = ag_other_tensor.cuda()

        if self.args.w_sampling == 'goal_oriented':
            with torch.no_grad():
                w = self.backward_network(g_tensor)
                w = w.detach()
        elif self.args.w_sampling == 'uniform_ball':
            w = self.sample_uniform_ball(self.args.batch_size)
        elif self.args.w_sampling == 'cauchy_ball':
            w = self.sample_cauchy_ball(self.args.batch_size)

        # calculate the target Q value function
        with torch.no_grad():
            if self.args.soft_update:
                pi = self.get_policy(obs_next_tensor, w, policy_type='boltzmann', temp=self.args.temp,
                                     target_network=True)
                f_next = torch.einsum('sda, sa -> sd', self.forward_target_network(obs_next_tensor, w), pi)
            else:
                actions_next_tensor = self.act(obs_next_tensor, w, target_network=True)
                next_idxs = actions_next_tensor[:, None].repeat(1, self.args.embed_dim)[:, :, None]
                f_next = self.forward_target_network(obs_next_tensor, w).gather(-1, next_idxs).squeeze()  # batch x dim


            b_next = self.backward_target_network(ag_other_tensor)  # batch x dim
            z_next = torch.einsum('sd, td -> st', f_next, b_next)  # batch x batch
            z_next = z_next.detach()
            # # clip the q value
            # clip_return = 1 / (1 - self.args.gamma)
            # target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the forward loss
        idxs = actions_tensor[:, None].repeat(1, self.args.embed_dim)[:, :, None]
        f = self.forward_network(obs_tensor, w).gather(-1, idxs).squeeze()
        b = self.backward_network(ag_tensor)
        b_other = self.backward_network(ag_other_tensor)
        z_diag = torch.einsum('sd, sd -> s', f, b)  # batch
        z = torch.einsum('sd, td -> st', f, b_other)  # batch x batch
        fb_loss = 0.5 * (z - self.args.gamma * z_next).pow(2).mean() - z_diag.mean()
        # compute orthonormality's regularisation loss
        b_b_other = torch.einsum('sd, xd -> sx', b, b_other)  # batch x batch
        b_b_other_detach = torch.einsum('sd, xd -> sx', b, b_other.detach())  # batch x batch
        b_b_detach = torch.einsum('sd, sd -> s', b, b.detach())  # batch
        reg_loss = (b_b_detach * b_b_other.detach()).mean() - b_b_other_detach.mean()
        fb_loss += self.args.reg_coef * reg_loss

        # update the forward_network
        self.fb_optim.zero_grad()
        fb_loss.backward()
        self.fb_optim.step()

        # the backward loss
        # f = self.forward_network(obs_norm_tensor, actions_tensor, w)
        # b = self.backward_network(ag_norm_tensor)
        # b_other = self.backward_network(g_other_norm_tensor)
        # z_diag = torch.einsum('sd, sd -> s', f, b)  # batch
        # z = torch.einsum('sd, td -> st', f, b_other)  # batch x batch
        # b_loss = 0.5 * (z - self.args.gamma * z_next).pow(2).mean() - z_diag.mean()
        # compute orthonormality's regularisation loss
        # b_b_other = torch.einsum('sd, xd -> sx', b, b_other)  # batch x batch
        # b_b_other_detach = torch.einsum('sd, xd -> sx', b, b_other.detach())  # batch x batch
        # b_b_detach = torch.einsum('sd, sd -> s', b, b.detach())  # batch
        # reg_loss = (b_b_detach * b_b_other.detach()).mean() - b_b_other_detach.mean()
        # b_loss += self.args.reg_coef * reg_loss
        #
        # # update the backward_network
        # self.backward_optim.zero_grad()
        # b_loss.backward()
        # sync_grads(self.backward_network)
        # self.backward_optim.step()

        # print('f_loss: {}, b_loss: {}'.format(f_loss.item(), b_loss.item()))

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
                    g_tensor = self._preproc_g(g)
                    w = self.backward_network(g_tensor)
                    obs_tensor = self._preproc_o(obs)
                    action = self.act(obs_tensor, w).item()
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

    def _eval_gpi_agent(self, num_gpi=20):
        total_success_rate = []
        total_dist = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            per_dist = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            if self.args.w_sampling == 'goal_oriented':
                transitions = self.buffer.sample(num_gpi)
                g_train = transitions['g']
                g_train_tensor = torch.tensor(g_train, dtype=torch.float32)
                if self.args.cuda:
                    g_train_tensor = g_train_tensor.cuda()
                w_train = self.backward_network(g_train_tensor)
            elif self.args.w_sampling == 'uniform_ball':
                w_train = self.sample_uniform_ball(num_gpi)
            elif self.args.w_sampling == 'cauchy_ball':
                w_train = self.sample_cauchy_ball(num_gpi)

            # for _ in range(self.env_params['max_timesteps']):
            for _ in range(25):
                with torch.no_grad():
                    g_tensor = self._preproc_g(g)
                    w = self.backward_network(g_tensor)
                    obs_tensor = self._preproc_o(obs)
                    action = self.act_gpi(obs_tensor, w_train, w).item()
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
