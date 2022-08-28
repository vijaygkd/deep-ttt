"""

"""
import numpy as np


class QAgent:
    def __init__(self, q_estimator, epsilon=1., epsilon_min=0.1, epsilon_decay_steps=10000):
        self.q_estimator = q_estimator
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = (epsilon - epsilon_min) / epsilon_decay_steps

    def play_next_move(self, board):
        """
        Agent with epsilon greedy policy.
        Random move with probability epsilon else best move from Q estimator
        """
        policy = np.random.choice(['random', 'q_agent'], 1, p=[self.epsilon, 1-self.epsilon])[0]
        if policy == 'random':
            move = RandomAgent.play_next_move(board)
        else:
            dummy = np.array([-1])
            input_board = board.reshape(1, len(board))
            move = self.q_estimator.model.predict([input_board, dummy], verbose=0)
        self.after_move()
        return move, policy

    def after_move(self):
        """
        Run after agent makes a move
        """
        self.epsilon -= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.epsilon_min)


class RandomAgent:
    """
    Plays moves randomly with uniform distribution
    """
    @staticmethod
    def play_next_move(board):
        a, = np.where(board == 0)
        choice = np.random.choice(a)
        return choice
