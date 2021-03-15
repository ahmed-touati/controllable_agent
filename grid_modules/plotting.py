import numpy as np
from grid_modules.gridworld import actions as ACTIONS
from grid_modules.gridworld.helper_utilities import from_id_to_xy

asbestos = (127 / 255, 140 / 255, 141 / 255, 0.8)
carrot = (235 / 255, 137 / 255, 33 / 255, 0.8)
emerald = (80 / 255, 200 / 255, 120 / 255, 0.8)
red = (255 / 255, 0 / 255, 0 / 255, 0.8)

marker_style = dict(linestyle=':', color=carrot, markersize=15)

DEFAULT_ARROW_COLOR = '#1ABC9C'


def plot_environment(
        mdp, ax, wall_locs=None, goal_ids=None, initial_state=None, door_ids=None, plot_grid=False,
        grid_kwargs=None,
        wall_color=asbestos  # (127 / 255, 140 /255, 141 / 255 , 0.8), # R, G, B, alpha
):
    """Function to plot emdp environment

        Args:
          mdp: The MDP to use.
          ax: The axes to plot this on.
          wall_locs: Locations of the walls for plotting them in a different color.
          plot_grid: Boolean indicating if the overlay grid should be plotted.
          grid_kwargs: Grid keyword argrument specification.
          wall_color: RGB color of the walls.

        Returns:
          ax: The axes of the final plot.
          imshow_ax: The final plot.
    """
    grid_kwargs = grid_kwargs or {}

    # Plot states with white background.
    state_background = np.ones((mdp.size, mdp.size))

    # Walls appear in a different color.
    wall_img = np.ones((mdp.size, mdp.size, 4))
    if wall_locs is not None:
        for state in wall_locs:
            y_coord = state[0]
            x_coord = state[1]
            wall_img[y_coord, x_coord, :] = np.array(wall_color)

    # Render the heatmap and overlay the walls.
    imshow_ax = ax.imshow(state_background, interpolation=None)
    imshow_ax = ax.imshow(wall_img, interpolation=None)

    # add initial state
    if initial_state is None:
        initial_state = mdp.reset()
    y_coord, x_coord = mdp.unflatten_state(initial_state)
    ax.plot(x_coord, y_coord, marker='H', **marker_style)

    # add door state
    if door_ids is not None:
        for door_id in door_ids:
            y_coord, x_coord = from_id_to_xy(door_id, size=mdp.size)
            ax.plot(x_coord, y_coord, marker='s', color=red, markersize=10)

    # add goal state
    if goal_ids is not None:
        for goal_id in goal_ids:
            y_coord, x_coord = from_id_to_xy(goal_id, size=mdp.size)
            ax.plot(x_coord, y_coord, marker='*', **marker_style)

    ax.grid(False)

    # Switch on flag if you want to plot grid
    if plot_grid:
        for i in range(mdp.size + 1):
            ax.plot(
                np.arange(mdp.size + 1) - 0.5,
                np.ones(mdp.size + 1) * i - 0.5,
                **grid_kwargs)
        for i in range(mdp.size + 1):
            ax.plot(
                np.ones(mdp.size + 1) * i - 0.5,
                np.arange(mdp.size + 1) - 0.5,
                **grid_kwargs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return ax, imshow_ax


def get_current_state_integer(state_):
    return np.argmax(state_, axis=0)


def _is_absorbing(state_int, mdp_size):
    """Checks if the state_int is an absorbing state"""
    return state_int == mdp_size * mdp_size


def _checking_P(P):
    """Checks if the P matrix is valid."""
    assert np.all(P <= 1.0) and np.all(P >= 0.0)
    assert not np.allclose(P, 1.0)
    assert not np.allclose(P, 0.0)


def plot_action(ax, y_pos, x_pos, a, headwidth=5, linewidths=1, scale=1.9, headlength=4):
    left_arrow = (-0.6, 0)
    right_arrow = (0.6, 0)
    up_arrow = (0, -0.6)
    down_arrow = (0, 0.6)
    if a == ACTIONS.LEFT:  # Left
        ax.quiver(
            x_pos, y_pos, *left_arrow, color=DEFAULT_ARROW_COLOR, alpha=1.0,
            angles='xy', scale_units='xy', scale=scale,
            headwidth=headwidth, linewidths=linewidths,
            headlength=headlength) #L
    if a == ACTIONS.RIGHT:  #Right
        ax.quiver(
            x_pos, y_pos, *right_arrow, color=DEFAULT_ARROW_COLOR, alpha=1.0,
            angles='xy', scale_units='xy', scale=scale,
            headwidth=headwidth, linewidths=linewidths,
            headlength=headlength) #R
    if a == ACTIONS.UP:  #Up
        ax.quiver(
            x_pos, y_pos, *up_arrow, color=DEFAULT_ARROW_COLOR, alpha=1.0,
            angles='xy', scale_units='xy', scale=scale,
            headwidth=headwidth, linewidths=linewidths,
            headlength=headlength) #U
    if a == ACTIONS.DOWN:  #Down
        ax.quiver(
            x_pos, y_pos, *down_arrow, color=DEFAULT_ARROW_COLOR, alpha=1.0,
            angles='xy', scale_units='xy', scale=scale,
            headwidth=headwidth, linewidths=linewidths,
            headlength=headlength) #D