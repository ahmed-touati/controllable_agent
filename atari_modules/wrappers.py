import numpy as np
import os
os.environ.setdefault('PATH', '')
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)
from atariari.benchmark.wrapper import AtariARIWrapper, ram2label
from baselines.common.atari_wrappers import MaxAndSkipEnv, WarpFrame, LazyFrames


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noops=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        modification from baselines
        """
        gym.Wrapper.__init__(self, env)
        self.noops = noops
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        obs = None
        for _ in range(self.noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        --------
        modification from baselines.common.atari_wrappers (step function)
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(3):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            self.frames.append(obs)
            if done:
                break
        return self._get_ob(), total_reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


def make_atari(env_id):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noops=240)
    env = MaxAndSkipEnv(env, skip=4)
    return env


# def goal_distance(goal_a, goal_b):
#     assert goal_a.shape == goal_b.shape
#     return np.linalg.norm(goal_a - goal_b, axis=-1)

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.max(np.abs(goal_a - goal_b), axis=-1)


class CroppedFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(170, 160, 3), dtype=np.uint8)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return observation[: 170, :, :]


class LifeLossEnv(gym.Wrapper):
    def __init__(self, env):
        """Make a life loss an end of the episode.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info


class GoalMsPacman(gym.Wrapper):
    def __init__(self, env, distance_threshold=6, reward_type='sparse'):
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.env = env
        # Maintain list of reachable goals
        # is_valid_idx = np.array([
        #     [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        #     [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #     [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
        #     [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
        #     [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        #     [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        #     [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        #     [1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1],
        #     [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        #     [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
        #     [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1],
        #     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.bool)
        # valid_idx = np.transpose(np.nonzero(is_valid_idx))
        #
        # # columns and rows correspond to x and y respectively
        # self.all_goals = []  # Translate goal space to pixel space range
        # prev_row = 0
        # for row_i, row in enumerate(range(14, 171, 12)):
        #     prev_col = 0
        #     for col_i, col in enumerate(range(14, 160, 8)):
        #         if np.sum(np.all(valid_idx == np.array([row_i, col_i]), axis=1)) > 0:
        #             self.all_goals.append(np.array([int((prev_col + col)/2), int((prev_row + row)/2)]))
        #         prev_col = col
        #     prev_row = row
        # self.total_num_goals = len(self.all_goals)

        all_goals = [[11, 9], [18, 9], [27, 9], [34, 9], [50, 9], [58, 9], [66, 9], [76, 9], [86, 9], [95, 9], [104, 9], [112, 9], [127, 9], [135, 9], [143, 9], [150, 9],
          [36, 21], [50, 21], [112, 21], [128, 21], [152, 21],
          [11, 34], [18, 34], [35, 34], [43, 34], [51, 34], [59, 34], [68, 34], [76, 34], [87, 34], [95, 34], [103, 34], [111, 34], [119, 34], [127, 34], [135, 34], [143, 34], [151, 34],
          [20, 46], [34, 46], [60, 46], [103, 46], [129, 46], [144, 46],
          [11, 57], [19, 57], [35, 57], [43, 57], [51, 57], [59, 57], [67, 57], [75, 57], [87, 57], [95, 57], [103, 57], [111, 57], [118, 57], [127, 57], [143, 57], [151, 57],
          [20, 69], [59, 69], [104, 69], [144, 69],
          [20, 82], [27, 82], [35, 82], [43, 82], [43, 82], [51, 82], [59, 82], [103, 82], [111, 82], [128, 82], [135, 82], [142, 82],
          [20, 93], [58, 93], [103, 93], [143, 93],
          [12, 105], [19, 105], [35, 105], [43, 105], [51, 105], [110, 105], [118, 105], [126, 105], [143, 105], [151, 105],
          [19, 117], [35, 117], [50, 117], [50, 117], [67, 117], [94, 117], [110, 117], [127, 117], [143, 117],
          [12, 129], [19, 129], [27, 129], [35, 129], [35, 129], [51, 129], [67, 129], [75, 129], [86, 129], [95, 129], [111, 129], [126, 129], [135, 129], [143, 129], [151, 129],
          [12, 141], [35, 141], [50, 141], [67, 141], [95, 141], [102, 141], [112, 141], [127, 141], [151, 141],
          [12, 153], [35, 153], [67, 153], [95, 153], [127, 153], [151, 153],
          [12, 165], [19, 165], [27, 165], [35, 165], [43, 165], [50, 165], [59, 165], [68, 165], [76, 165], [87, 165], [87, 165], [96, 165], [103, 165],
          [112, 165], [119, 165], [127, 165], [136, 165], [142, 165], [151, 165]]
        self.all_goals = np.array(all_goals)
        self.all_goals += np.array([[0, -2]])
        self.total_num_goals = len(self.all_goals)

        self.action_space = spaces.Discrete(5)
        obs = self.reset()
        self.observation_space = dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=self.env.observation_space,
        )

        # sanity check
        # obs = env.reset()
        # for g in all_goals:
        #     obs[g[1]-2: g[1]+2, g[0]-2:g[0]+2, :] = 255

    def _get_pos(self):
        ram = self.env.unwrapped.ale.getRAM()
        label_dict = ram2label(self.env.spec.id, ram)
        return np.array([label_dict['player_x'], label_dict['player_y']]) + np.array([-8, 6])

    def reset(self):
        lazy_obs = self.env.reset()
        achieved_goal = self._get_pos()
        self.goal = self._sample_goal()
        return {
            'observation': lazy_obs,
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy()
        }

    def step(self, action):
        lazy_obs, reward, done, info = self.env.step(action)
        achieved_goal = self._get_pos()
        obs = {
            'observation': lazy_obs,
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy()
        }
        info['is_success'] = self._is_success(obs['achieved_goal'], self.goal)
        reward = self.compute_reward(obs['achieved_goal'], self.goal, None)
        return obs, reward, done, info

    def _sample_goal(self):
        id = np.random.randint(self.total_num_goals)
        return self.all_goals[id]

    def set_goal(self, g):
        self.goal = g

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d <= self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return (d <= self.distance_threshold).astype(np.float32)
        else:
            return -d


def make_goalPacman():
    env = make_atari('MsPacmanNoFrameskip-v4')
    env = LifeLossEnv(env)
    env = CroppedFrame(env)
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    env = GoalMsPacman(env)
    return env


if __name__=='__main__':
    import matplotlib as mpl
    # mpl.use('TkAgg')
    import matplotlib.pyplot as plt
    env = make_atari('MsPacmanNoFrameskip-v4')
    env = LifeLossEnv(env)
    env = CroppedFrame(env)
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    env = GoalMsPacman(env)
    obs = env.reset()

    env.set_goal(np.array([40, 160]))
    for i in range(100):
        plt.imsave('plots/step_{}.jpg'.format(i), env.unwrapped._get_obs())
        if i < 30:
            obs, reward, done, info = env.step(3)
        else:
            obs, reward, done, info = env.step(4)
        if done:
            print('Death')
        if info['is_success'] > 0:
            print('Success !!')
            print(i)
            break

    # raw_obs = env.unwrapped._get_obs()
    # plt.imshow(raw_obs)
    # plt.show()

