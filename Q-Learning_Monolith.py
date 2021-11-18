import numpy as np
import gym
import random
from time import sleep
from IPython.display import clear_output

def print_frames(frames):
    '''Map drawing function'''
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'].getvalue())
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)

# Q-Learning hyperparameters
alpha = 0.2     # Learning rate
gamma = 0.7
epsilon = 0.1   # Exploring factor

# Metrics arrays
all_epochs = []
all_penalties = []

# Animation
frames = []
draw = False

env = gym.make("Taxi-v3").env
q_table = np.zeros([env.observation_space.n, env.action_space.n])

for i in range(10001):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()      # Explore actions
        else:
            action = np.argmax(q_table[state])      # Choose best action known

        # Step action
        next_state, reward, done, info = env.step(action)

        # Save old Q-table value and take next maximum value from Q-table
        old_value = q_table[state, action]
        next_max = np.max([q_table[next_state]])

        # Apply Q-Learning expression
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        if draw:
            frames.append({
                'frame': env.render(),
                'state': state,
                'action': action,
                'reward': reward
            })

        state = next_state
        epochs += 1
    # end while

    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Training Episode: {i}")
# end for

print("Training finished. \n")

if draw:
    print_frames(frames)


# Agent evaluation

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    if _ % 10 == 0:
        clear_output(wait=True)
        print(f"Evaluation Episode: {_}")

    total_epochs += epochs
    total_penalties += penalties

print("Evaluation finished.\n")
print(f"Total epochs: {total_epochs}\nTotal penalties: {total_penalties}")
