"""
Train DQN network for learning TTT by using re-inforcement learning
"""
import numpy as np
from tqdm import tqdm
import mlflow

from dqn.environment import *
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
        print("hey8")
        # mlflow init
        mlflow.tensorflow.autolog()
        mlflow.set_experiment(experiment_id='1')
        # init training entities
        self.memory = Memory(size=10000)
        self.net = DQN()
        self.agent = QAgent(self.net)
        self.gamma = 0.95      # Q wt of next state's reward
        # validation data
        # generate game state data using random agent policy
        self.val_memory = self.generate_random_states(n=1000)

    def train(self, episodes=100):
        total_rewards_per_game = []
        observe_player = 1
        for e in tqdm(range(episodes)):
            game = TicTacToe()      # new game
            total_game_reward = 0
            game_transactions = []
            while game.game_over == 0:
                # get next action from agent
                current_state = game.get_current_state()
                action, _ = self.agent.play_epsilon_greedy_policy(current_state)
                # execute action in environment
                reward, is_game_over = game.execute_action(action)
                next_state = game.get_current_state()
                # store transition
                state_data = [current_state, action, reward, next_state, is_game_over]
                game_transactions.append(state_data)
                # record metrics
                if game.current_player == observe_player:
                    total_game_reward += reward

            # after game is complete
            # assign next states and save transactions to memory
            for k, transaction in enumerate(game_transactions):
                # the next state for the Q should be state after the opponent has played.
                # more intuitive, if we think of the opponent as part of the environment
                if transaction[4] == 0:             # if game is not over
                    next_state = game_transactions[k+1][3].copy()  # next state from the opponent's move
                else:
                    next_state = np.full(9, -1)     # invalid data as it's never used for training
                transaction[3] = next_state
                self.memory.add_record(transaction)

            total_rewards_per_game.append(total_game_reward)
            observe_player *= -1    # track alternate first and second player
            # perform gradient decent
            if e > 100:
                self.do_gradient_update(batch_size=32)
                if e % 100 == 0:
                    # metrics 1: avg reward per episode
                    mlflow.log_metric("avg_reward_per_episode", np.mean(total_rewards_per_game))
                    total_rewards_per_game = []
                    # metrics 2: avg_max_q value of fixed random states
                    self.calculate_validation_score()
                    # metrics 3: games won against random
                    stats = play_games_against_random(self.agent, 100)
                    mlflow.log_metrics(stats)
                    # log other metrics
                    mlflow.log_metric("step", e)
                    mlflow.log_metric("epsilon", self.agent.epsilon)

        # end of training
        # self.net.save('dqn_net')

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

        # outputs = self.net.model.predict(x=[current_states, actions])
        #
        # print(current_states)
        # print(actions)
        # print(rewards)
        # print(targets)
        # print(outputs)

        # gradient update
        self.net.model.train_on_batch(
            # during training model predicts max Q values for given actions
            x=[current_states, actions],
            y=[actions, targets],
            # return_dict=True,
        )

        # outputs = self.net.model.predict(x=[current_states, actions])
        # print(outputs)

    def calculate_validation_score(self):
        val_records = self.val_memory.sample_records(sample_size=self.val_memory.size)
        current_states = val_records[0]
        actions = val_records[1]
        outputs = self.net.model.predict([current_states, actions], verbose=0)
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
            game_transactions = []
            while t.game_over == 0:
                current_state = t.get_current_state()
                action, _ = RandomAgent.play(current_state)
                reward, game_over = t.execute_action(action)
                next_state = t.get_current_state()
                is_game_over = t.game_over

                state_data = [current_state, action, reward, next_state, is_game_over]
                game_transactions.append(state_data)

            # after game is over
            # assign next states and save transactions to memory
            for k, transaction in enumerate(game_transactions):
                if transaction[4] == 0:  # if game is not over
                    next_state = game_transactions[k+1][3].copy()  # next state from opponent's move
                else:
                    next_state = np.full(9, -1)  # invalid data as it's never used for training
                transaction[3] = next_state
                val_memory.add_record(transaction)

        return val_memory


def play_games_against_random(agent, no_of_games):
    random_agent = RandomAgent()
    players = [random_agent, agent]
    stats = {
        'win': 0,
        'lost': 0,
        'draw': 0,
    }
    for i in range(no_of_games):
        g = TicTacToe()
        c = i
        while g.game_over == 0:
            current_player = c % 2
            c += 1
            player = players[current_player]
            move, _ = player.play(g.get_current_state())
            reward, game_over = g.execute_action(move)

        if reward == DRAW:
            stats['draw'] += 1/no_of_games
        elif reward == WIN and current_player == 1:
            stats['win'] += 1/no_of_games
        else:
            stats['lost'] += 1/no_of_games

    for k, v in stats.items():
        stats[k] = np.round(v, 2)
    return stats



