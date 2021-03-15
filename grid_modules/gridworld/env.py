"""
A simple grid world environment
"""

import numpy as np
from grid_modules.common import MDP
from .helper_utilities import flatten_state, unflatten_state, get_reachable_id,\
    from_xy_to_id, from_id_to_xy


class GridWorldMDP(MDP):
    def __init__(self, P, R, gamma, p0, terminal_states, size, wall_locs, goal_loc=None,
                 seed=1337, skip_check=False, convert_terminal_states_to_ints=False):
        """
        (!) if terminal_states is not empty then there will be an absorbing state. So
            the actual number of states will be size x size + 1
            if there is a terminal state, it should be the last one.
        :param P: Transition matrix |S| x |A| x |S|
        :param R: Transition matrix |S| x |A|
        :param gamma: discount factor
        :param p0: initial starting distribution
        :param terminal_states: Must be a list of (x,y) tuples.  use skip_terminal_state_conversion if giving ints
        :param size: the size of the grid world (i.e there are size x size (+ 1)= |S| states)
        :param seed:
        :param skip_check:
        """
        if not convert_terminal_states_to_ints:
            terminal_states = list(map(lambda tupl: int(size * tupl[0] + tupl[1]), terminal_states))
        self.size = size
        self.wall_locs = wall_locs
        if goal_loc is not None:
            self.goal_loc = goal_loc
            self.goal_id = from_xy_to_id(goal_loc, self.size)
            self.goal = flatten_state(self.goal_loc, self.size, self.state_space)
        self.human_state = (None, None)
        self.has_absorbing_state = len(terminal_states) > 0
        super().__init__(P, R, gamma, p0, terminal_states, seed=seed, skip_check=skip_check)
        self.reachable_states = get_reachable_id(self.state_space, self.size, self.wall_locs)
        # set initial state distribution to uniform
        p0 = np.zeros(self.state_space)
        p0[self.reachable_states] = 1
        p0 /= p0.sum()
        self.p0 = p0

    def set_goal(self, goal_id):
        self.goal_id = goal_id
        self.goal_loc = from_id_to_xy(goal_id, self.size)
        self.goal = flatten_state(self.goal_loc, self.size, self.state_space)
        R = np.zeros((self.state_space, self.action_space))  # S x A
        R[goal_id, :] = 1
        self.R = R

    def reset(self):
        super().reset()
        self.goal_id = self.sample_goal()
        self.goal_loc = from_id_to_xy(self.goal_id, self.size)
        self.goal = flatten_state(self.goal_loc, self.size, self.state_space)
        R = np.zeros((self.state_space, self.action_space))  # S x A
        R[self.goal_id, :] = 1
        self.R = R
        self.human_state = self.unflatten_state(self.current_state)
        return self.current_state

    def sample_goal(self):
        goal = np.random.choice(self.reachable_states, size=1, replace=False)
        return goal

    def flatten_state(self, state):
        """Flatten state (x,y) into a one hot vector"""
        return flatten_state(state, self.size, self.state_space)

    def unflatten_state(self, onehot):
        """Unflatten a one hot vector into a (x,y) pair"""
        return unflatten_state(onehot, self.size, self.has_absorbing_state)

    def step(self, action):
        state, reward, done, info = super().step(action)
        self.human_state = self.unflatten_state(self.current_state)
        return state, reward, done, info

    def set_current_state_to(self, tuple_state):
        return super().set_current_state_to(self.flatten_state(tuple_state).argmax())