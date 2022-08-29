import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Input
from tensorflow.keras import optimizers
from tensorflow.keras import metrics, losses


class QOutputLayer(Layer):
    def __init__(self, **kwargs):
        super(QOutputLayer, self).__init__(**kwargs)

    def call(self, inputs, indices, training=None):
        if training:
            # return [action, Q value of selection positions / actions]
            indices = tf.cast(indices, tf.int32)
            # return tf.gather(inputs, indices, axis=1, batch_dims=1)
            return [indices, tf.gather(inputs, indices, axis=1, batch_dims=1)]
        else:
            # during inference, return argmax Q. ie. action with higher Q value
            # return tf.argmax(inputs, axis=1)
            # return [max action, Q value of max action]
            return [tf.argmax(inputs, axis=1), tf.reduce_max(inputs, axis=1)]


class DQN:
    def __init__(self, learning_rate=3e-4):
        self.learning_rate = learning_rate
        self.model = self.get_initial_model()

    def get_initial_model(self):
        """
        Initialize DQN model with random weights
        """
        # input layer - board state
        input_states_layer = Input(shape=(9,), name='input_states')
        input_actions_layer = Input(shape=(1,), name='input_actions')
        # Hidden layers
        hidden = Dense(64, activation='relu', name='hidden_1')(input_states_layer)
        hidden = Dense(32, activation='relu', name='hidden_2')(hidden)
        # Output layer - Q value for each action
        output_layer = Dense(9, activation='linear', name='output')(hidden)
        # Reduce output to single value
        final_layer = QOutputLayer(name='final_layer')(output_layer, input_actions_layer)
        # Model
        model = Model(inputs=[input_states_layer, input_actions_layer], outputs=final_layer)

        # Optimizer
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)

        # TODO - Define DQN loss and metrics
        model.compile(
            loss=losses.MeanSquaredError(),
            optimizer=optimizer,
            metrics=[
                metrics.MeanSquaredError(),
            ])
        return model

    def save(self, path):
        self.model.save(path)
