# Mountain Car- v0 solution using action-value neural network function approximator

import torch
import torch.nn as nn
from torchvision import transforms

# One hidden layer function approximator
class action_value_function(nn.Module):

    def __init__(self):

        super(action_value_function, self).__init__()
        self.block = nn.Sequential(
            torch.nn.Linear(2, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 3),
            torch.nn.Softmax(),
        )

    def forward(self, input):
        return self.block(input)

q_hat = action_value_function()
# q_hat = q_hat.float()
# state#.double()
# print(q_hat((state.float())))
print(torch.argmax(q_hat((torch.ones(1,2).float()))))

import gym
import numpy as np

done = False
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 2000
BATCH_SIZE = 4

env = gym.make("MountainCar-v0")
env.reset()

# DISCRETE_OS_SIZE = [20, 20] #* len(env.observation_space.high)
# discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

# Test number of actions and other parameters
print('observation_space:',env.observation_space.low,'to',env.observation_space.high,'| Number of action values:',env.action_space.n)

# Function for discretization of observation state space
# def get_discrete_state(state):
#     discrete_state = (state - env.observation_space.low)/discrete_os_win_size
#     return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table


# q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# Train agent
for episode in range(EPISODES):
    state = torch.from_numpy(env.reset()).float()
    torch.tensor(state)
    done = False

    if episode%SHOW_EVERY == 0:
        print(episode)
        RENDER = True
    else:
        RENDER = False


    while not done:
        action = torch.argmax(q_hat(state)).numpy()
        next_state, reward, done, _ = env.step(action)

        if RENDER:
            env.render()

        if not done:

            max_next_q = np.max(q_hat(next_state))
            current_q = q_hat(state + (action,))

#             updated_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_next_q)
#             q_hat[discrete_state + (action, )] = new_q # Improve q_hat

        state = next_state

env.close()
