from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .TicTacToeLogic import Board
import numpy as np

"""
Game class implementation for the game of TicTacToe.
Based on the OthelloGame then getGameEnded() was adapted to new rules.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloGame by Surag Nair.
"""
from MuZeroConfig import MuZeroConfig
import Game.*

def make_tictactoe_config(board_size, game) -> MuZeroConfig:
    return make_board_game_config(game, 
                                  action_space_size=(board_size*board_size) + 1, max_moves=200, dirichlet_alpha=0.3, lr_init=0.1, num_simulations=25, num_actors=5)

class TicTacToeEnv(Environment):
    self.board = Board()
    
    def step(self, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        move = (int(action/self.n), action%self.n)
        return self.board.execute_move(move, player)
        

class TicTacToeGame(Game):
    self.environment = TicTacToeEnv()
    def __init__(self, n=3):
        self.n = n

    def legal_actions(self):
        moves = []  # stores the legal moves.
        
        for i in range(self.n*self.n):
            moves.append(Action(i))
        return list(moves)

    def terminal(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves():
            return 0
        # draw has a very little value 
        return 1e-4

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

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
