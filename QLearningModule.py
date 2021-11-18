import numpy as np
import gym
import random
from time import sleep
from IPython.display import clear_output


class QLearn():
    def __init__(self, environment="Taxi-v3", alpha=0.2, gamma=0.6, epsilon=0.0):
        # Play with the parameters. Sometimes it gets stuck in an endless loop.
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.environment = environment
        self.frames_train = []
        self.frames_eval = []

        self.env = gym.make(self.environment).env
        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])

    def save_frames(self, frames, state, action, reward):
        frames.append({
            'frame': self.env.render(),
            'state': state,
            'action': action,
            'reward': reward
        })
        return frames

    # def print_frames(self, frames):
    #     for i, frame in enumerate(frames):
    #         clear_output(wait=True)
    #         print(frame['frame'].getvalue())
    #         print(f"Timestep: {i + 1}")
    #         print(f"State: {frame['state']}")
    #         print(f"Action: {frame['action']}")
    #         print(f"Reward: {frame['reward']}")
    #         sleep(.1)

    def q_learning(self, draw=False):
        done = False
        state = self.env.reset()
        epochs, penalties, reward = 0, 0, 0
        while not done:
            if random.uniform(0, 1) < self.epsilon:
                action = self.env.action_space.sample()  # Explore actions
            else:
                action = np.argmax(self.q_table[state])  # Choose best action known

            # Step action
            next_state, reward, done, info = self.env.step(action)

            # Save old Q-table value and take next maximum value from Q-table
            old_value = self.q_table[state, action]
            next_max = np.max([self.q_table[next_state]])

            # Apply Q-Learning expression
            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
            self.q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            if draw:
                self.frames_train = self.save_frames(self.frames_train, state, action, reward)
                # self.print_frames(self.frames_train)

            state = next_state
            epochs += 1
        # end while
        return self.q_table

    def q_evaluation(self, draw=False):
        state = self.env.reset()
        epochs, penalties, reward = 0, 0, 0
        done = False
        while not done:
            action = np.argmax(self.q_table[state])
            state, reward, done, info = self.env.step(action)
            if reward == -10:
                penalties += 1
            if draw:
                self.frames_eval = self.save_frames(self.frames_eval, state, action, reward)
                # self.print_frames(self.frames_eval)
            epochs += 1
        # end while
        return epochs, penalties


def main():
    qlearner = QLearn()

    # Agent training
    for i in range(10001):
        q_table = qlearner.q_learning()
        if i % 100 == 0:
            clear_output(wait=True)
            print(f"Training Episode: {i}")
    print("Training finished. \n")

    # Agent evaluation
    total_epochs, total_penalties = 0, 0
    episodes = 100
    for _ in range(episodes):
        if _ % 10 == 0:
            clear_output(wait=True)
            print(f"Evaluation Episode: {_}")
        epochs, penalties = qlearner.q_evaluation()
        total_epochs += epochs
        total_penalties += penalties
    print("Evaluation finished.\n")
    print(f"Total epochs: {total_epochs}\nTotal penalties: {total_penalties}")


if __name__ == "__main__":
    main()
