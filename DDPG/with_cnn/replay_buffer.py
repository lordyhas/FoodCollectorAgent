import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=int(1e5), batch_size=64):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = []
        self.position = 0

    def push(self, state_img, action, reward, next_state_img, done):
        # On stocke l'image state, l'action, la reward, la next image, le done
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state_img, action, reward, next_state_img, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        batch = random.sample(self.buffer, k=self.batch_size)
        state_imgs, actions, rewards, next_imgs, dones = zip(*batch)
        return np.array(state_imgs), np.array(actions), np.array(rewards), np.array(next_imgs), np.array(dones)

    def __len__(self):
        return len(self.buffer)
