"""
Utilities to help build more complex grid worlds.
"""
import numpy as np
from grid_modules.gridworld.helper_utilities import (build_simple_grid,
                                                     get_possible_actions,
                                                     check_can_take_action,
                                                     get_state_after_executing_action,
                                                     flatten_state,
                                                     unflatten_state)
from grid_modules.gridworld.env import GridWorldMDP


class TransitionMatrixBuilder(object):
    """
    Builder object to build a transition matrix for a grid world
    """
    def __init__(self, grid_size, action_space=5, has_terminal_state=False):
        self.has_terminal_state = has_terminal_state
        self.grid_size = grid_size
        self.action_space = action_space
        self.state_space = grid_size * grid_size + int(has_terminal_state)
        self._P = np.zeros((self.state_space, self.action_space, self.state_space))
        self.grid_added = False
        self.P_modified = False

    def add_grid(self, terminal_states=[], p_success=1):
        """
        Adds a grid so that you cant walk off the edges of the grid
        :param terminal_states:
        :param p_success:
        :return:
        """
        if self.has_terminal_state and len(terminal_states) == 0:
            raise ValueError('has_terminal_states is true, but no terminal states supplied.')

        if self.grid_added:
            raise ValueError('Grid has already been added')

        if self.P_modified:
            raise ValueError('transition matrix has already been modified. '
                             'Adding a grid now can lead to weird behaviour')

        self._P = build_simple_grid(size=self.grid_size, p_success=p_success, terminal_states=terminal_states)
        self.grid_added = True
        self.P_modified = True

    def add_wall_at(self, tuple_location):
        """
        Add a blockade at this position
        :param tuple_location: (x,y) location of the wall
        :return:
        """
        target_state = flatten_state(tuple_location, self.grid_size, self.state_space)
        target_state = target_state.argmax()
        # find all the ways to go to "target_state"
        # from_states contains states that can lead you to target_state by executing from_action
        from_states, from_actions = np.where(self._P[:, :, target_state] != 0)

        # get the transition probability distributions that go from s--> t via some action
        transition_probs_from = self._P[from_states, from_actions, :]
        # TODO: optimize this loop
        for i, from_state in enumerate(from_states): # enumerate over states
            tmp = transition_probs_from[i, target_state] # get the prob of transitioning
            transition_probs_from[i, target_state] = 0 # set it to zero
            transition_probs_from[i, from_state] += tmp # add the transition prob to staying in the same place

        self._P[from_states, from_actions, :] = transition_probs_from

        # Get the probability of going to any state for all actions from target_state.
        transition_probs_from_wall = self._P[target_state, :, :]
        for i, probs_from_action in enumerate(transition_probs_from_wall):
            # Reset the probabilities.
            transition_probs_from_wall[i, :] = 0.0
            # Set the probability of going to the target state to be 1.0
            transition_probs_from_wall[i, target_state] = 1.0
        # Now set the probs of going to any state from target state as above (i.e only targets).
        self._P[target_state, :, :] = transition_probs_from_wall

        # renormalize and update transition matrix.
        normalization = self._P.sum(2)
        # normalization[normalization == 0] = 1
        normalization = 1/normalization
        self._P = (self._P * np.repeat(normalization, self._P.shape[0]).reshape(*self._P.shape))

        assert np.allclose(self._P.sum(2), 1), 'Normalization did not occur correctly: {}'.format(self._P.sum(2))
        assert np.allclose(self._P[target_state, :, target_state], 1.0), 'All actions from wall should lead to wall!'
        self._P_modified = True

    @property
    def P(self, nocopy=False):
        """
        Returns a new array with the transition matrix built so far.
        :param nocopy:
        :return:
        """
        if nocopy:
            return self._P
        else:
            return self._P.copy()

    def add_wall_between(self, start, end):
        """
        Adds a wall between the starting and ending location
        :param start: tuple (x,y) representing the starting position of the wall
        :param end: tuple (x,y) representing the ending position of the wall
        :return:
        """
        if not(start[0] == end[0] or start[1] == end[1]):
            raise ValueError('Walls can only be drawn in straight lines. '
                             'Therefore, at least one of the x or y between '
                             'the states should match.')

        if start[0] == end[0]:
            direction = 1
        else:
            direction = 0

        constant_idx = start[int(not direction)]
        start_idx = start[direction]
        end_idx = end[direction]

        if end_idx < start_idx:
            # flip start and end directions
            # to ensure we can still draw walls
            start_idx, end_idx = end_idx, start_idx

        for i in range(start_idx, end_idx+1):
            my_location = [None, None]
            my_location[direction] = i
            my_location[int(not direction)] = constant_idx
            print(my_location)
            self.add_wall_at(tuple(my_location))


def create_reward_matrix(state_space, size, reward_spec, action_space=5):
    """
    Abstraction to create reward matrices.
    :param state_space: Size of the state space
    :param size: Size of the gird world (width)
    :param reward_spec: The reward specification
    :param action_space: The size of the action space
    :return:
    """
    R = np.zeros((state_space, action_space))
    for (reward_location, reward_value) in reward_spec.items():
        reward_location = flatten_state(reward_location, size, state_space).argmax()
        R[reward_location, :] = reward_value

    return R


def sample_goal(mdp):
    # Sample goal
    n = 0
    max_tries = 1e5
    found = False
    start_loc = unflatten_state(mdp.p0, mdp.size)
    while n < max_tries and not found:
        goal_loc = (np.random.randint(mdp.size), np.random.randint(mdp.size))
        if goal_loc not in mdp.wall_locs and goal_loc != start_loc:
            # Reachable state found!
            found = True
        n += 1
    if not found:
        raise ValueError('Failed to sample a goal state.')
    reward_spec = {(goal_loc[0], goal_loc[1]): 1}
    R = create_reward_matrix(mdp.state_space, mdp.size, reward_spec, action_space=mdp.action_space)
    mdp.R = R
    mdp.goal_loc = goal_loc


def sample_reachable_states(mdp, nb_goals=5, dist=np.random.randn):
    n = 0
    max_tries = 1e5
    reward_spec = dict()
    for _ in range(nb_goals):
        found = False
        while n < max_tries and not found:
            state_loc = (np.random.randint(mdp.size), np.random.randint(mdp.size))
            if state_loc not in mdp.wall_locs:
                # Reachable state found!
                reward_spec[state_loc] = dist()
                found = True
            n += 1
        if not found:
            raise ValueError('Failed to sample a reachable state.')

    R = create_reward_matrix(mdp.state_space, mdp.size, reward_spec, action_space=mdp.action_space)
    return R



