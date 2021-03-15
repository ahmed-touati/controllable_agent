from continuous_world_modules.geometry import Point
from continuous_world_modules.env import ContinuousWorld, visualize_environment
import numpy as np
import matplotlib.pyplot as plt

AGENT = 'random'
env = ContinuousWorld(1, wall_pairs=[
            (Point(0.25, 0.0), Point(0.25, 0.4)),
            (Point(0.75, 1), Point(0.75, 0.6))],
                      movement_noise=0.01,
                      threshold_distance=0.05)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)

if AGENT == 'random':
    current_state = env.reset()
    # env.set_initial_position(Point(0.2, 0.1))
    # env.set_goal(Point(0.9, 0.9))
    # env.set_initial_position(Point(2, 1))
    # current_state = env.current_position
    # env.set_agent_position(Point(3, 4))
    visualize_environment(env, ax)
    for _ in range(100):
        x_pos, y_pos = current_state[0], current_state[1]
        action = np.random.choice(5)
        state, reward, done, info = env.step(action)
        # print(state)
        perturbed_action = state - current_state
        # action = ACTIONS.STAY
        # x_pos, y_pos = env.agent_position
        ax.quiver(x_pos, y_pos, perturbed_action[0], perturbed_action[1], color='#1ABC9C', alpha=1.0,
                  angles='xy', scale_units='xy', scale=1,
                  headwidth=5, linewidths=1,
                  headlength=4)

        current_state = state

    plt.show()