import itertools
import numpy as np
import gym
from gym import spaces

""" Modification from https://github.com/paulorauber/hpg/blob/master/hpg/environments/robotics.py"""


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def generate_itoa_dict(
        bucket_values=[-0.33, 0, 0.33], valid_movement_direction=[1, 1, 1, 1]):
    """
    Set cartesian product to generate action combination
        spaces for the fetch environments
    valid_movement_direction: To set
    """
    action_space_extended = [bucket_values if m == 1 else [0]
                             for m in valid_movement_direction]
    return list(itertools.product(*action_space_extended))


class FetchReach:
    def __init__(self,
                 action_mode="cart", action_buckets=[-1, 0, 1],
                 action_stepsize=1.0,
                 reward_type="sparse"):
        """
        Parameters:
            action_mode {"cart","cartmixed","cartprod","impulse","impulsemixed"}
            action_stepsize: Step size of the action to perform.
                            Int for cart and impulse
                            List for cartmixed and impulsemixed
            action_buckets: List of buckets used when mode is cartprod
            reward_mode = {"sparse","dense"}

        Reward Mode:
            `sparse` rewards are like the standard HPG rewards.
            `dense` rewards (from the paper/gym) give -(distance to goal) at every timestep.

        Modes:
            `cart` is for manhattan style movement where an action moves the arm in one direction
                for every action.

            `impulse` treats the action dimensions as velocity and adds/decreases
                the velocity by action_stepsize depending on the direction picked.
                Adds current direction
                velocity to state


            `impulsemixed` and `cartmixed` does the above with multiple magnitudes of action_stepsize.
                Needs the action_stepsize as a list.

            `cartprod` takes combinations of actions as input
        """

        try:
            self.env = gym.make("FetchReach-v1")
        except Exception as e:
            print(
                "You do not have the latest version of gym (gym-0.10.5). Falling back to v0 with movable table")
            self.env = gym.make("FetchReach-v0")

        self.action_directions = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        self.valid_action_directions = np.float32(
            np.any(self.action_directions, axis=0))

        self.distance_threshold = self.env.distance_threshold
        # self.distance_threshold = 0.06
        self.action_mode = action_mode
        self.num_actions = self.generate_action_map(action_buckets, action_stepsize)

        obs_dim = 10 + 4 * (action_mode == "impulse" or action_mode == "impulsemixed")
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=(3, ), dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=(3, ), dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=(obs_dim, ), dtype='float32'),
        ))
        self._max_episode_steps = self.env._max_episode_steps
        self.reward_type = reward_type

    def generate_action_map(self, action_buckets, action_stepsize=1.):

        action_directions = self.action_directions
        if self.action_mode == "cart" or self.action_mode == "impulse":
            assert isinstance(action_stepsize, float)
            self.action_space = np.vstack(
                (action_directions * action_stepsize, -action_directions * action_stepsize))

        elif self.action_mode == "cartmixed" or self.action_mode == "impulsemixed":
            assert isinstance(action_stepsize, list)
            action_space_list = []
            for ai in action_stepsize:
                action_space_list += [action_directions * ai,
                                      -action_directions * ai]
            self.action_space = np.vstack(action_space_list)

        elif self.action_mode == "cartprod":
            self.action_space = generate_itoa_dict(
                action_buckets, self.valid_action_directions)

        return len(self.action_space)

    def seed(self, seed):
        self.env.seed(seed)

    def action_map(self, action):
        # If the modes are direct, just map the action as an index
        # else, accumulate them

        if self.action_mode in ["cartprod", "cart", "cartmixed"]:
            return self.action_space[action]
        else:
            self.action_vel += self.action_space[action]
            self.action_vel = np.clip(self.action_vel, -1, 1)
            return self.action_vel

    def reset(self):
        self.action_vel = np.zeros(4)  # Initialize/reset
        obs = self.env.reset()
        if self.action_mode == "impulse" or self.action_mode == "impulsemixed":
            obs["observation"] = np.hstack((obs["observation"], self.action_vel))
        return obs

    def step(self, a):

        action_vec = self.action_map(a)
        obs, reward, done, info = self.env.step(action_vec)
        if self.action_mode == "impulse" or self.action_mode == "impulsemixed":
            obs["observation"] = np.hstack(
                (obs["observation"], np.clip(self.action_vel, -1, 1)))

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.env.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.env.goal, info)
        return obs, reward, done, info

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d <= self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def __del__(self):
        self.env.close()


class FetchPush(FetchReach):
    def __init__(self,
                 action_mode="impulsemixed", action_buckets=[-1, 0, 1],
                 action_stepsize=[0.1, 1.0],
                 reward_type="sparse"):

        try:
            self.env = gym.make("FetchPush-v1")
        except Exception as e:
            print(
                "You do not have the latest version of gym (gym-0.10.5). Falling back to v0 with movable table")
            self.env = gym.make("FetchPush-v0")

        self.action_directions = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.valid_action_directions = np.float32(
            np.any(self.action_directions, axis=0))

        self.goal = self.env.goal
        self.distance_threshold = self.env.distance_threshold
        self.action_mode = action_mode
        self.num_actions = self.generate_action_map(action_buckets, action_stepsize)

        obs_dim = 25 + 4 * (action_mode == "impulse" or action_mode == "impulsemixed")
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=(3, ), dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=(3, ), dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=(obs_dim, ), dtype='float32'),
        ))
        self._max_episode_steps = self.env._max_episode_steps
        self.reward_type = reward_type
        self.is_train = False


class FetchSlide(FetchReach):
    def __init__(self,
                 action_mode="cart", action_buckets=[-1, 0, 1],
                 action_stepsize=1.0,
                 reward_type="sparse"):

        try:
            self.env = gym.make("FetchSlide-v1")
        except Exception as e:
            print(
                "You do not have the latest version of gym (gym-0.10.5). Falling back to v0 with movable table")
            self.env = gym.make("FetchSlide-v0")

        self.action_directions = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.valid_action_directions = np.float32(
            np.any(self.action_directions, axis=0))

        self.goal = self.env.goal
        self.distance_threshold = self.env.distance_threshold
        self.action_mode = action_mode
        self.num_actions = self.generate_action_map(action_buckets, action_stepsize)
        obs_dim = 25 + 4 * (action_mode == "impulse" or action_mode == "impulsemixed")
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=(3, ), dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=(3, ), dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=(obs_dim, ), dtype='float32'),
        ))
        self._max_episode_steps = self.env._max_episode_steps
        self.reward_type = reward_type


if __name__=='__main__':
    env = FetchReach()
    obs = env.reset()
