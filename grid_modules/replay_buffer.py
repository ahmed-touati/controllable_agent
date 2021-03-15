import numpy as np
import random

"""
Slightly different the replay buffer here is basically from the openai baselines code
in order to include done for atari games and discrete actions

"""


class ReplayBuffer(object):
    """taken from https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py"""

    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs, g, action, reward, obs_next, done):
        data = (obs, g, action, reward, obs_next, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses, gs, actions, rewards, obses_next, dones = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs, g, action, reward, obs_next, done = data
            obses.append(np.array(obs, copy=False))
            # gs.append(g.copy())
            gs.append(g)
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_next.append(np.array(obs_next, copy=False))
            dones.append(done)
        transitions = {'obs': np.array(obses),
                       'g': np.array(gs),
                       'action': np.array(actions),
                       'reward': np.array(rewards),
                       'obs_next': np.array(obses_next),
                       'done': np.array(dones)}
        return transitions

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class her_replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, 2]),
                        'g': np.empty([self.size, self.T, 2]),
                        'action': np.empty([self.size, self.T])
                        }

    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        idxs = self._get_storage_idx(inc=batch_size)
        # store the informations
        self.buffers['obs'][idxs] = mb_obs
        self.buffers['g'][idxs] = mb_g
        self.buffers['action'][idxs] = mb_actions
        self.n_transitions_stored += self.T * batch_size

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx