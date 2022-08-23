"""

"""
import numpy as np


class RandomAgent:
    """
    Plays moves randomly with uniform distribution
    """
    @staticmethod
    def play_next_move(board):
        a, = np.where(board == 0)
        choice = np.random.choice(a)
        return choice
