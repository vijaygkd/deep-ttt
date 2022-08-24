from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import optimizers
from tensorflow.keras import metrics, losses


class DQN:
    def __init__(self, learning_rate=3e-4):
        self.learning_rate = learning_rate

    def get_initial_model(self):
        """
        Initialize DQN model with random weights
        """
        # input layer - board state
        input_layer = Input(shape=(9,), name='input')
        # Hidden layers
        hidden = Dense(64, activation='relu', name='hidden_1')(input_layer)
        hidden = Dense(32, activation='relu', name='hidden_2')(hidden)
        # Output layer - one for each position
        output_layer = Dense(9, activation='linear', name='output')(hidden)
        # Model
        model = Model(inputs=input_layer, outputs=output_layer)

        # Optimizer
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)

        # TODO - Define DQN loss and metrics
        model.compile(
            loss=losses.MeanSquaredError(),
            optimizer=optimizer,
            metrics=[
                metrics.MeanSquaredError(),
                metrics.RootMeanSquaredError(),
                metrics.MeanAbsoluteError(),
            ])

        return model
