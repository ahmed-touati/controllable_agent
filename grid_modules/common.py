import numpy as np
from grid_modules import utils
from grid_modules.exceptions import InvalidActionError, EpisodeDoneError


class Env(object):
    """
    Abstract Environment wrapper.
    """
    def __init__(self, seed):
        """
        :param seed: A seed for the random number generator.
        """
        self.set_seed(seed)

    def set_seed(self, seed):
        self.rng = np.random.RandomState(seed)


class MDP(Env):
    def __init__(self, P, R, gamma, p0, terminal_states, seed=1337, skip_check=False):
        """
        A simple MDP simulator.
        :param P: The transition matrix of size |S|x|A|x|S|
        :param R: The reward criterion |S|x|A|
        :param gamma: the discount factor.
        :param p0: the distribution over starting states |S| (must sum to 1.)
        :param terminal_states: A list of integers which indicate terminal states, used to end episodes.
                                Note that in the transition matrix these
                                should be absorbing states to ensure calculations are correct.
        :param seed: the random seed for simulations.
        """
        super().__init__(seed)
        if not skip_check: assert np.allclose(P.sum(axis=2), 1), 'Transition matrix does not seem to be a stochastic matrix ' \
                                           '(i.e. the sum over states for each action doesn not equal 1'
        self.P = P
        self.R = R
        self.state_space = P.shape[0]
        self.action_space = R.shape[1]
        if not skip_check: assert self.state_space == P.shape[2], '3rd Dimension of Transition Matrix is not of size |S|'
        if not skip_check: assert self.action_space == P.shape[1], '2nd Dimension of Transition Matrix is not of size |A|'
        if not skip_check: assert self.state_space == R.shape[0], '1st Dimesnion of Reward Matrix is not of size |S|'
        self.gamma = gamma
        if not skip_check: assert self.state_space == p0.shape[0], 'Distribution over initial states is not over |S|'
        self.p0 = p0
        self.terminal_states = terminal_states
        self.current_state = None
        # self.reset()

    def reset(self):
        integer_representation = np.random.choice(np.arange(self.state_space), p=self.p0)
        self.current_state = utils.convert_int_rep_to_onehot(integer_representation, self.state_space)
        self.done = False
        return self.current_state

    def set_current_state_to(self, state):
        self.current_state = utils.convert_int_rep_to_onehot(state, self.state_space)
        self.done = False
        return self.current_state

    def step(self, action):
        """
        :param action: An integer representing the action taken.
        :return:
        """
        if self.done:
            raise EpisodeDoneError('The episode has terminated. Use .reset() to restart the episode.')
        if action >= self.action_space or not isinstance(action, int):
            raise InvalidActionError('Invalid action {}. It must be an integer between 0 and {}'.format(action, self.action_space-1))

        # we end from this episode onwards.
        # this check is done after entering terminal state
        # because we can only give the reward after leaving
        # a terminal state.
        if self.current_state.argmax() in self.terminal_states:
            self.done = True

        # get the vector representing the next state probabilities:
        current_state_idx = utils.convert_onehot_to_int(self.current_state)
        next_state_probs = self.P[current_state_idx, action]

        # sample the next state
        sampled_next_state = self.rng.choice(np.arange(self.state_space), p=next_state_probs)
        # observe the reward
        reward = self.R[current_state_idx, action]

        self.current_state = utils.convert_int_rep_to_onehot(sampled_next_state, self.state_space)

        return self.current_state, reward, self.done, {'gamma':self.gamma}