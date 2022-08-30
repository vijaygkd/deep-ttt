"""

"""
import numpy as np


class QAgent:
    def __init__(self, q_estimator, epsilon=1., epsilon_min=0.1, epsilon_decay_steps=10000):
        self.q_estimator = q_estimator
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = (epsilon - epsilon_min) / epsilon_decay_steps

    def play_epsilon_greedy_policy(self, board):
        """
        Agent with epsilon greedy policy.
        Random move with probability epsilon else best move from Q estimator
        """
        policy = np.random.choice(['random', 'q_agent'], 1, p=[self.epsilon, 1-self.epsilon])[0]
        if policy == 'random':
            move = RandomAgent.play(board)
        else:
            move, q_value = self.play(board)
        self.after_move()
        return move, policy

    def play(self, board):
        dummy = np.array([-1])
        input_board = board.reshape(1, len(board))
        output = self.q_estimator.model.predict([input_board, dummy], verbose=0)
        move = output[0]
        q_value = output[1]
        return move, q_value

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
    def play(board):
        a, = np.where(board == 0)
        choice = np.random.choice(a)
        return choice