import numpy as np
import random
from collections import deque

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        """
        Ajoute du bruit OU à l'action continue (forme (action_dim,)).
        Clip final dans [-1,1].
        """
        ou_state = self.evolve_state()
        # Decay de sigma
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma)*min(1.0, float(t)/self.decay_period)
        noisy_action = action + ou_state
        return np.clip(noisy_action, -1.0, 1.0)

