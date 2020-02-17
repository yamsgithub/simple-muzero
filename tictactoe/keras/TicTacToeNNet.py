import sys
sys.path.append('..')
from utils import *

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from Network import Network
"""
NeuralNet for the game of TicTacToe.

Author: Yamuna Krishnamurthy, github.com/yamsgithub

"""
class TicTacToeNNet(Network):
    def __init__(self, game, config):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.num_channels = 512

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y

        # Representation function
        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(config.num_channels, 3, padding='same')(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(config.num_channels, 3, padding='same')(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(config.num_channels, 3, padding='same')(h_conv2)))        # batch_size  x (board_x) x (board_y) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(config.num_channels, 3, padding='same')(h_conv3)))        # batch_size  x (board_x) x (board_y) x num_channels

        self.representation = Model(inputs=self.input_boards, outputs=h_conv4)

        # Dynamics function
        self.input_actions = Input(shape=(self.board_x, self.board_y))
        x_action_image = Reshape((self.board_x, self.board_y, 1))(self.self.dyn_actions)                # batch_size  x board_x x board_y x 1
        h_conv5 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(config.num_channels, 3, padding='same')([x_image, x_action_image])))         # batch_size  x board_x x board_y x num_channels
        h_conv6 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(config.num_channels, 3, padding='same')(h_conv5)))         # batch_size  x board_x x board_y x num_channels
        h_conv7 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(config.num_channels, 3, padding='same')(h_conv6)))        # batch_size  x (board_x) x (board_y) x num_channels
        h_conv8 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(config.num_channels, 3, padding='same')(h_conv7)))        # batch_size  x (board_x-2) x (board_y-2) x num_chan

        self.dynamics = Model([self.input_boards, self.input_actions], h_vonv8)
        
        # Policy function
        h_conv9 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(config.num_channels, 3, padding='same')(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv10 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(config.num_channels, 3, padding='valid')(h_conv9)))         # batch_size  x board_x x board_y x num_channel
        h_conv10_flat = Flatten()(h_conv10)       
        s_fc1 = Dropout(config.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv10_flat))))  # batch_size x 1024
        s_fc2 = Dropout(config.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))          # batch_size x 1024
        self.pi = Dense(config.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

        self.prediction = Model(inputs=self.input_boards, outputs=[self.pi, self.v])

    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function
        return NetworkOutput(0, 0, {}, [])

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction function
        return NetworkOutput(0, 0, {}, [])

    def get_weights(self):
        # Returns the weights of this network.
        return []

  def training_steps(self) -> int:
    # How many steps / batches the network has been trained for.
    return 0

