""" Multi-Armed Bandit implementation for retraining. """

import numpy as np

class MultiArmedBandit:
    def __init__(self, n_arms : int):
        """
        Initializes the Multi-Armed Bandit with Beta priors for each arm.

        Parameters:
        - n_arms (int): Number of arms/models.
        """

        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self):
        """
        Uses Thompson Sampling to select an arm
        """

        samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n_arms)]
        return np.argmax(samples)


    def update(self, chosen_arm : int, reward : int):
        """
        Updates posterior distribution
        """

        if reward == 1:
            self.alpha[chosen_arm] += 1
        else:
            self.beta[chosen_arm] += 1

    def get_estimates(self):
        """
        Returns the current estimates of each arm's success probability.

        Returns:
        - estimates (list): Estimated probabilities for each arm.
        """
        return self.alpha / (self.alpha + self.beta)