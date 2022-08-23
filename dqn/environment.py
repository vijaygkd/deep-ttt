"""
Environment for play ttt
"""
import numpy as np

## REWARDS
WIN = 1
DRAW = 0.5
MOVE = -0.1
LOSE = -1
INVALID = -5


class TicTacToe:
    winning_combo = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],    # horizontal
        [0, 3, 6], [1, 4, 7], [2, 5, 8],    # vertical
        [0, 4, 8], [2, 4, 6]                # diagonal
    ]

    def __init__(self):
        self.board = np.zeros(9)
        self.current_player = 1
        self.game_over = 0

    def get_current_state(self):
        return self.board

    def set_next_player(self):
        # alternate between +1 / -1
        self.current_player *= -1

    def get_reward(self):
        """return reward based on board state after move
        """
        # check draw
        if 0 not in self.board:
            self.game_over = 1
            return DRAW
        # check win
        for c in self.winning_combo:
            if abs(self.board[c].sum()) == 3:
                self.game_over = 1
                return WIN

        # TODO LOSE reward is never sent to agent ??
        return MOVE

    def execute_action(self, position):
        """
        Execute action on board
        """
        assert(0 <= position <= 8)
        # check if move is INVALID
        if self.board[position] != 0:
            reward = INVALID
            return reward, self.game_over, self.board,   # no change to board state

        # make move and check if terminated
        self.board[position] = self.current_player
        self.set_next_player()
        reward = self.get_reward()
        return reward, self.game_over, self.board,

    def print_board(self):
        m = {
            1: 'X',
            -1: 'O',
            0: '-'
        }
        for i in range(3):
            s = ''
            for j in range(3):
                c = m[self.board[i * 3 + j]]
                s += c + ' '
            print(f'{s}\n')
