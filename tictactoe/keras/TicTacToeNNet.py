import sys
sys.path.append('..')
import numpy as np

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

from utils import *
from Network import *
"""
NeuralNet for the game of TicTacToe.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloNNet by SourKream and Surag Nair.
"""

  
class TicTacToeNNet(Network):
    def __init__(self, config):
        # game params
        self.board_x = self.board_y = int(math.sqrt(config.action_space_size))
        self.num_channels = 512
        self.dropout = 0.3
        
        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y

        # Representation function
        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        print('\nINPUT INFERENCE ', x_image.shape)
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels, 3, padding='same')(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels, 3, padding='same')(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels, 3, padding='same')(h_conv2)))        # batch_size  x (board_x) x (board_y) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels, 3, padding='same')(h_conv3)))        # batch_size  x (board_x) x (board_y) x num_channels

        h_conv4_flat = Flatten()(h_conv4)
        h_conv4_dense = Dense(self.board_x*self.board_y)(h_conv4_flat)
        
        self.representation = Model(inputs=self.input_boards, outputs=h_conv4_dense)
        print('\n\n\nOUTPUT REP ', h_conv4_dense.shape)

        
        # Dynamics function
        x_action_state_image = Reshape((self.board_x, self.board_y, 2))(self.input_boards)                # batch_size  x board_x x board_y x 1
        h_conv5 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels, 3, padding='same')(x_action_state_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv6 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels, 3, padding='same')(h_conv5)))         # batch_size  x board_x x board_y x num_channels
        h_conv7 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels, 3, padding='same')(h_conv6)))        # batch_size  x (board_x) x (board_y) x num_channels
        h_conv8 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels, 3, padding='same')(h_conv7)))        # batch_size  x (board_x-2) x (board_y-2) x num_chan

        h_conv8_flat = Flatten()(h_conv8)
        h_conv8_dense = Dense(self.board_x*self.board_y)(h_conv8_flat)

        self.dynamics = Model(self.input_boards, h_conv8_dense)
        print('\n\n\nOUTPUT DYNAMICS ', h_conv8_dense.shape)

        
        # Policy function
        h_conv9 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels, 3, padding='same')(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv10 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels, 3, padding='valid')(h_conv9)))         # batch_size  x board_x x board_y x num_channel
        h_conv10_flat = Flatten()(h_conv10)       
        s_fc1 = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv10_flat))))  # batch_size x 1024
        s_fc2 = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))          # batch_size x 1024
        self.pi = Dense(config.action_space_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

        self.prediction = Model(inputs=self.input_boards, outputs=[self.pi, self.v])

    def initial_inference(self, image) -> NetworkOutput:
        image = image[np.newaxis, :, :]
        
        # representation + prediction function
        # Neural Net
        print('\nIMAGE ', image)
        output = self.representation(image)
        return NetworkOutput(0, 0, {}, output)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction function
        #print('\n\nHIDDEN STATE ', hidden_state,'\n', hidden_state.shape)
        #print('\n\nACTION ', hash(action))

        board = int(math.sqrt(hidden_state.shape[1]))
        hidden_state = tf.reshape(hidden_state, (1, board, board))
        
        action_state = np.zeros(board*board)
        action_state[hash(action)] = 1
        action_state = action_state.reshape((1, board, board))
        
        data = tf.concat((hidden_state, action_state), -1) #.reshape(2, shape[0], shape[1])
        hidden_state = self.dynamics(data)
        #print('\n\n\nHIDDEN STATE SHAPE ', hidden_state.shape)
        policy, value = self.prediction(hidden_state)
        policy = np.asarray(policy[0])
        policy_logits = {Action(a): policy[a] for a in range(0, self.board_x * self.board_y)}
        return NetworkOutput(value,  0, policy_logits, hidden_state)

    def get_weights(self):
        # Returns the weights of this network.
        return []

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return 0


class TicTacToeSharedStorage(SharedStorage):
    def __init__(self, config):
        super(TicTacToeSharedStorage, self).__init__()
        self.config = config
        
    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return TicTacToeNNet(self.config)
