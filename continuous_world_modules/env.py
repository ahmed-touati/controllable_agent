from typing import Optional, List, Tuple, Any, Dict, Union, Callable
import numpy as np
import dataclasses
import operator
import random

from continuous_world_modules.geometry import Point, intersect, on_segment

# d = {'LEFT': 0, 'RIGHT': 1, 'UP': 2, 'DOWN': 3, 'STAY': 4}
# ACTIONS = SimpleNamespace(**d)
CARDINAL_ACTIONS = [Point(-0.1, 0), Point(0.1, 0.), Point(0., 0.1), Point(0., -0.1), Point(0., 0.)]
ACTIONS_STR = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'STAY']

carrot = (235 / 255, 137 / 255, 33 / 255, 0.8)
marker_style = dict(linestyle=':', color=carrot, markersize=15)


class ContinuousWorld(object):
    r"""The ContinuousWorld Environment.

    An agent can be anywhere in the grid. The agent provides Forces to move. When
    the agent provides a force, it is applied and the final position is jittered.

    Walls can be specified in this environment. Detection works by checking if the
    agents action forces it to go in a direction which collides with a wall.
    """

    def __init__(
      self,
      size: float,
      wall_pairs: Optional[List[Tuple[Point, Point]]] = None,
      movement_noise: float = 0.01,
      seed: int = 1,
      reset_noise: Optional[float] = None,
      verbose_reset: bool = False,
      threshold_distance: float = 0.5
      ):
        """Initializes the Continuous World Environment.

        Args:
          size: The size of the world.
          wall_pairs: A list of tuple of points representing the start and end
            positions of the wall.
          movement_noise: The noise around each position after movement.
          seed: The seed for the random number generator.
          max_episode_length: The maximum length of the episode before resetting.
          max_action_force: If using random_step() this will be the maximum random
            force applied in the x and y direction.
          verbose_reset: Prints out every time the global starting position is
            reset.
        """
        self.actions = CARDINAL_ACTIONS
        self.actions_str = ACTIONS_STR
        self._size = size
        self._wall_pairs = wall_pairs or []
        self._verbose_reset = verbose_reset
        self.threshold_distance = threshold_distance

        self._noise = movement_noise
        self._reset_noise = reset_noise or movement_noise
        self._rng = np.random.RandomState(seed)
        random.seed(seed)

        # self.update_start_position()
        # self.set_agent_position(self._start_position)

    def _past_edge(self, x: float) -> Tuple[bool, float]:
        """Checks if coordinate is beyond the edges."""
        if x >= self._size:
          return True, self._size
        elif x <= 0.0:
          return True, 0.0
        else:
          return False, x

    def _wrap_coordinate(self, point: Point) -> Point:
        """Wraps coordinates that are beyond edges."""
        wrapped_coordinates = map(self._past_edge, dataclasses.astuple(point))
        return Point(*map(operator.itemgetter(1), wrapped_coordinates))

    # def set_agent_position(self, new_position: Point):
    #     self._current_position = self._wrap_coordinate(new_position)

    def set_goal(self, new_position: Point):
        self._goal = self._wrap_coordinate(new_position)

    def set_initial_position(self, new_position: Point):
        self._current_position = self._wrap_coordinate(new_position)

    # def update_start_position(self):
    #     self._start_position = Point(*np.random.uniform(0, self._size, 2))

    def reset(self):
        """Reset the current position of the agent and move the global mu."""
        self._current_position = self.sample_point()
        self._goal = self.sample_point()
        return self.current_position

    def sample_point(self):
        on_wall = True
        while on_wall:
            p = Point(*np.random.uniform(0, self._size, 2))
            on_wall = self._check_on_wall(p)
        return p

    @property
    def goal(self):
        return np.array(dataclasses.astuple(self._goal))
    # def agent_position(self):
    #     return dataclasses.astuple(self._current_position)

    @property
    def current_position(self):
        return np.array(dataclasses.astuple(self._current_position))

    # @property
    # def start_position(self):
    #     return dataclasses.astuple(self._start_position)

    @property
    def size(self):
        return self._size

    @property
    def walls(self):
        return self._wall_pairs

    def _check_goes_through_wall(self, start: Point, end: Point):
        if not self._wall_pairs:
            return False

        for pair in self._wall_pairs:
            if intersect((start, end), pair):
                return True
        return False

    def _check_on_wall(self, p: Point):
        if not self._wall_pairs:
            return False
        for pair in self._wall_pairs:
            if on_segment(pair[0], pair[1], p):
                return True

    def step(self, id_action) -> Tuple[Tuple[float, float], Optional[float], bool, Dict[str, Any]]:
        """Does a step in the environment using the action.

        Args:
          action: action's index to be executed.

        Returns:
          Agent position: A tuple of two floats.
          The reward.
          An indicator if the episode terminated.
          A dictionary containing any information about the step.
        """
        perturbed_action = self.actions[id_action].normal_sample_around(self._noise)
        proposed_position = self._wrap_coordinate(self._current_position + perturbed_action)
        goes_through_wall = self._check_goes_through_wall(self._current_position, proposed_position)

        if not goes_through_wall:
            self._current_position = proposed_position
        done = False
        reward = 1.0 if self._current_position.is_close_to(self._goal, diff=self.threshold_distance) else 0.0
        return self.current_position, reward, done, {'goes_through_wall': goes_through_wall, 'proposed_position': proposed_position}


def visualize_environment(
    world,
    ax,
    scaling=1.0,
    agent_color='r',
    agent_size=0.2,
    start_color='g',
    draw_initial_position=True,
    draw_goal=True,
    write_text=False):
    """Visualize the continuous grid world.

    The agent will be drawn as a circle. The start and target
    locations will be drawn by a cross. Walls will be drawn in
    black.

    Args:
    world: The continuous gridworld to visualize.
    ax: The matplotlib axes to draw the gridworld.
    scaling: Scale the plot by this factor.
    agent_color: Color of the agent.
    agent_size: Size of the agent in the world.
    start_color: Color of the start marker.
    draw_agent: Boolean that controls drawing agent.
    draw_start_mu: Boolean that controls drawing starting position.
    draw_target_mu: Boolean that controls drawing ending position.
    draw_walls: Boolean that controls drawing walls.
    write_text: Boolean to write text for each component being drawn.
    """
    carrot = (235 / 255, 137 / 255, 33 / 255, 0.8)
    marker_style = dict(linestyle=':', color=carrot, markersize=15)

    scaled_size = scaling * world.size

    # Draw the outer walls.
    ax.hlines(0, 0, scaled_size, color='k')
    ax.hlines(scaled_size, 0, scaled_size, color='k')
    ax.vlines(scaled_size, 0, scaled_size, color='k')
    ax.vlines(0, 0, scaled_size, color='k')

    for wall_pair in world.walls:
        ax.plot(
            [p.x * scaling for p in wall_pair],
            [p.y * scaling for p in wall_pair],
            color='k')

    if draw_initial_position:
        # Draw the position of the start dist.
        x, y = [p * scaling for p in world.current_position]
        ax.plot(x, y, marker='o', **marker_style)
        if write_text:
            ax.text(x, y, 'starting position.')

    if draw_goal:
        # Draw the target position.
        x, y = [p * scaling for p in world.goal]
        ax.plot(x, y, marker='*', **marker_style)
        if write_text:
            ax.text(x, y, 'target position.')

    # if draw_agent:
    #     # Draw the position of the agent as a circle.
    #     x, y = [scaling * p for p in world.current_position]
    #     ax.plot(x, y, marker='H', **marker_style)
    #     # agent_circle = plt.Circle((x, y), agent_size, color=agent_color)
    #     # ax.add_artist(agent_circle)
    #     if write_text:
    #         ax.text(x, y, 'current position.')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('off')
    ax.grid(False)
    return ax