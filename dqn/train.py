"""
Train DQN network for learning TTT by using re-inforcement learning
"""
from dqn.environment import TicTacToe
from dqn.agent import RandomAgent


def generate_random_states(n):
    """
    Generate random state data for validation by using random action policy
    """
    D = []
    while len(D) < n:
        t = TicTacToe()
        while t.game_over == 0:
            current_state = t.get_current_state()
            choice = RandomAgent.play_next_move(current_state)
            reward, game_over = t.execute_action(choice)
            next_state = t.get_current_state()
            d = (current_state, choice, next_state, game_over)
            D.append(d)

    return D
