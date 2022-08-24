"""

"""
import numpy as np


class QAgent:
    def __init__(self, q_estimator, epsilon, epsilon_decay):
        self.q_estimator = q_estimator
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def play_next_move(self, board):
        """
        Agent with epsilon greedy policy.
        Random move with probability epsilon else best move from Q estimator
        """
        policy = np.random.choice(['random', 'q_agent'], 1, p=[self.epsilon, 1-self.epsilon])[0]
        if policy == 'random':
            move = RandomAgent.play_next_move(board)
        else:
            move, q_estimates = self.q_estimator.predict(board)
        return move, policy


class RandomAgent:
    """
    Plays moves randomly with uniform distribution
    """
    @staticmethod
    def play_next_move(board):
        a, = np.where(board == 0)
        choice = np.random.choice(a)
        return choice
