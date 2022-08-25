"""
Train DQN network for learning TTT by using re-inforcement learning
"""
import random
import numpy as np

from dqn.environment import TicTacToe
from dqn.agent import RandomAgent, QAgent
from dqn.model import DQN


class Memory:
    def __init__(self, size):
        self.size = size
        self.current_states = np.empty((size, 9))
        self.next_states = np.empty((size, 9))
        self.rewards = np.empty(size)
        self.actions = np.empty(size)
        self.is_terminal_state = np.empty(size)

    def add_record(self, record):
        def add_fifo(tape, value):
            # insert records FIFO
            tape[0] = value
            return np.roll(tape, 1, axis=0)

        self.current_states = add_fifo(self.current_states, record[0])
        self.actions = add_fifo(self.actions, record[1])
        self.rewards = add_fifo(self.rewards, record[2])
        self.next_states = add_fifo(self.next_states, record[3])
        self.is_terminal_state = add_fifo(self.is_terminal_state, record[4])

    def sample_records(self, sample_size):
        rand_index = np.random.randint(0, self.size, size=sample_size)
        return [
            self.current_states[rand_index],
            self.actions[rand_index],
            self.rewards[rand_index],
            self.next_states[rand_index],
            self.is_terminal_state[rand_index],
        ]


class QLearning:
    memory = Memory(size=10000)
    net = DQN()
    agent = QAgent(net)

    def train(self, episodes=100):
        steps_counter = 0
        for i in range(episodes):
            game = TicTacToe()      # new game
            while game.game_over == 0:
                # execute action
                current_state = game.get_current_state()
                action, _ = self.agent.play_next_move(current_state)
                reward, is_game_over = game.execute_action(action)
                next_state = game.get_current_state()
                # store transition
                state_data = [current_state, action, reward, next_state, is_game_over]
                self.memory.add_record(state_data)
            # perform gradient decent
            self.do_gradient_decent(batch_size=32)


        self.net.save('dqn_net')

    def do_gradient_decent(self, batch_size):
        mini_batch = self.memory.sample_records(batch_size)
        current_states = mini_batch[0]
        actions = mini_batch[1]
        rewards = mini_batch[1]
        next_states = mini_batch[1]
        is_terminal_state = mini_batch[1]

        self.net.model.train_on_batch(
            x=current_states,
            y=next_states       # todo fix y value
        )

    @staticmethod
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
                # TODO : next state for Q training ??
                # option 1: next state is the board state after executing agent's move.
                # Drawback is Q doesn't know what the opponent will play.
                # Next action can't be based on his own actions.
                # option 2. the next state for the Q should be state after the opponent has played.
                # more intuitive, if we think of the opponent as part of the environment
                next_state = t.get_current_state()
                d = (current_state, choice, next_state, game_over)
                D.append(d)

        return D
