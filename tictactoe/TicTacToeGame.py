import sys
sys.path.append('..')
from Game import Game, Action
from utils import MuZeroConfig, Environment, Player, make_board_game_config
from .TicTacToeLogic import Board
import numpy as np
from math import sqrt

"""
Game class implementation for the game of TicTacToe.
Based on the OthelloGame then getGameEnded() was adapted to new rules.

Author: Yamuna Krishnamurthy, github.com/yamsgithub

Based on the TicTacToe Game by Surag Nair.
"""



class TicTacToeEnv(Environment):
    board = Board()
    
    def step(self, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        move = (int(action/self.n), action%self.n)
        return self.board.execute_move(move, player)
        
class TicTacToePlayer(Player):
    player = 1
    def __init__(self, name):
        self.player = name

    def turn(self):
        return self.player

    
class TicTacToeGame(Game):
    environment = TicTacToeEnv()
    observations = []

    def __init__(self, action_space_size, discount):
        super(TicTacToeGame, self).__init__(action_space_size, discount)
        self.n = int(sqrt(action_space_size))
        self.player = TicTacToePlayer(1)

    def legal_actions(self):
        moves = []  # stores the legal moves.
        
        for i in range(self.n*self.n):
            moves.append(Action(i))
        return list(moves)
    
    def make_image(self, state_index: int):
        # Game specific feature planes.
        if not self.observations:
            b = Board(self.n)
            self.observations.append(b)
            return b.pieces
        elif state_index < len(self.observations):
            return self.observations[state_index].pieces
    
    def terminal(self):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        if not self.observations:
            return 0
        
        b = self.observations[-1]
        player = self.player.player
        
        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves():
            return 0
        # draw has a very little value 
        return 1e-4

    def getCanonicalForm(self, board):
        # return state if player==1, else return -state if player==-1
        return self.player.player*np.asarray(board).astype(float)

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l
    
    def apply(self, action: Action):
        self.environment.step(action)
        self.history.append(action)

    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tostring()

    def to_play(self):
        return TicTacToePlayer(-1 * self.player.player)

    @staticmethod
    def display(board):
        n = board.shape[0]

        print("   ", end="")
        for y in range(n):
            print (y,"", end="")
        print("")
        print("  ", end="")
        for _ in range(n):
            print ("-", end="-")
        print("--")
        for y in range(n):
            print(y, "|",end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                if piece == -1: print("X ",end="")
                elif piece == 1: print("O ",end="")
                else:
                    if x==n:
                        print("-",end="")
                    else:
                        print("- ",end="")
            print("|")

        print("  ", end="")
        for _ in range(n):
            print ("-", end="-")
        print("--")

class TicTacToeConfig(MuZeroConfig):
    def __init__(self,
                 action_space_size: int,
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 batch_size: int,
                 td_steps: int,
                 num_actors: int,
                 lr_init: float,
                 lr_decay_steps: float,
                 visit_softmax_temperature_fn):
        super(TicTacToeConfig, self).__init__(action_space_size = action_space_size,
                                              max_moves = max_moves,
                                              discount = discount,
                                              dirichlet_alpha = dirichlet_alpha,
                                              num_simulations = num_simulations,
                                              batch_size = batch_size,
                                              td_steps = td_steps, 
                                              num_actors = num_actors,
                                              lr_init = lr_init,
                                              lr_decay_steps = lr_decay_steps,
                                              visit_softmax_temperature_fn = visit_softmax_temperature_fn)
    def new_game(self):
        return TicTacToeGame(self.action_space_size, self.discount)


def make_tictactoe_config(board_size) -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        if num_moves < 30:
            return 1.0
        else:
            return 0.0  # Play according to the max.
        
    return TicTacToeConfig(action_space_size=(board_size*board_size),
                           max_moves=200,
                           discount=0.1,
                           dirichlet_alpha=0.3,
                           num_simulations=25,
                           batch_size = 64,
                           td_steps = 100,
                           num_actors=5,
                           lr_init=0.001,
                           lr_decay_steps=5,
                           visit_softmax_temperature_fn = visit_softmax_temperature)

