import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity=int(1e5), batch_size=64):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions),
                np.array(rewards), np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)
