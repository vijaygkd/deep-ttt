"""
Train DQN network for learning TTT by using re-inforcement learning
"""
import numpy as np
from tqdm import tqdm
import mlflow

from dqn.environment import TicTacToe
from dqn.agent import RandomAgent, QAgent
from dqn.model import DQN


class Memory:
    def __init__(self, size):
        self.size = size
        self.current_states = np.empty((size, 9))
        self.next_states = np.empty((size, 9))
        self.rewards = np.zeros(size)
        self.actions = np.zeros(size)
        self.is_terminal_state = np.zeros(size)
        self.records_added = 0

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
        self.records_added += 1

    def sample_records(self, sample_size):
        end = min(self.size, self.records_added)
        rand_index = np.random.randint(0, end, size=sample_size)
        return [
            self.current_states[rand_index],
            self.actions[rand_index],
            self.rewards[rand_index],
            self.next_states[rand_index],
            self.is_terminal_state[rand_index],
        ]


class QLearning:
    def __init__(self, gamma=0.1):
        print("hey4")
        # mlflow init
        mlflow.tensorflow.autolog()
        mlflow.set_experiment(experiment_id='1')
        # init training entities
        self.memory = Memory(size=10000)
        self.net = DQN()
        self.agent = QAgent(self.net)
        self.gamma = gamma      # Q update rate
        # validation data
        # generate game state data using random agent policy
        self.val_memory = self.generate_random_states(n=1000)

    def train(self, episodes=100):
        for i in tqdm(range(episodes)):
            game = TicTacToe()      # new game
            rewards = []
            while game.game_over == 0:
                # get next action from agent
                current_state = game.get_current_state()
                action, _ = self.agent.play_next_move(current_state)
                # execute action in environment
                reward, is_game_over = game.execute_action(action)
                next_state = game.get_current_state()
                # store transition
                state_data = [current_state, action, reward, next_state, is_game_over]
                self.memory.add_record(state_data)
                rewards.append(reward)

            # perform gradient decent
            if i > 100:
                self.do_gradient_update(batch_size=32)
                if i % 10 == 0:
                    # metrics 1: avg reward per episode
                    mlflow.log_metric("avg_reward_per_episode", np.mean(rewards))
                    rewards = []
                    # metrics 2: avg_max_q value of fixed random states
                    self.calculate_validation_score()

        # end of training
        self.net.save('dqn_net')

    def get_training_targets(self, rewards, next_states, is_terminal_state):
        """
        target = reward                                 --> when next state is terminal
        target = reward + gamma * max Q (next start)    --> when next state is non-terminal
        """
        output = self.net.model.predict([next_states, np.zeros_like(is_terminal_state)], verbose=0)
        next_max_actions = output[0]
        next_states_max_q = output[1]           # select output array containing max Q values
        game_continue = 1 - is_terminal_state
        targets = rewards + (self.gamma * next_states_max_q * game_continue)
        return targets

    def do_gradient_update(self, batch_size):
        mini_batch = self.memory.sample_records(batch_size)
        current_states = mini_batch[0]
        actions = mini_batch[1]
        rewards = mini_batch[2]
        next_states = mini_batch[3]
        is_terminal_state = mini_batch[4]
        targets = self.get_training_targets(rewards, next_states, is_terminal_state)
        # gradient update
        self.net.model.train_on_batch(
            # during training model predicts max Q values for given actions
            x=[current_states, actions],
            y=[actions, targets],
            # epochs=1,
            # verbose=0
        )

    def calculate_validation_score(self):
        val_records = self.val_memory.sample_records(sample_size=self.val_memory.size)
        current_states = val_records[0]
        actions = val_records[1]
        outputs = self.net.model.predict([current_states, actions])
        max_q_values = outputs[1]
        avg_max_q_value = np.mean(max_q_values)
        mlflow.log_metric("val_avg_max_q", avg_max_q_value)


    @staticmethod
    def generate_random_states(n):
        """
        Generate random state data for validation by using random action policy
        """
        val_memory = Memory(size=n)
        while val_memory.records_added < n:
            t = TicTacToe()
            while t.game_over == 0:
                current_state = t.get_current_state()
                action = RandomAgent.play_next_move(current_state)
                reward, game_over = t.execute_action(action)
                # TODO : next state for Q training ??
                # option 1: next state is the board state after executing agent's move.
                # Drawback is Q doesn't know what the opponent will play.
                # Next action can't be based on his own actions.
                # option 2. the next state for the Q should be state after the opponent has played.
                # more intuitive, if we think of the opponent as part of the environment
                next_state = t.get_current_state()
                is_game_over = t.game_over

                state_data = [current_state, action, reward, next_state, is_game_over]
                val_memory.add_record(state_data)

        return val_memory
