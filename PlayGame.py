import numpy as np

from tictactoe.TicTacToeGame import TicTacToeGame as Game
from tictactoe.TicTacToeGame import TicTacToeConfig as MuZeroConfig
from tictactoe.TicTacToeGame import make_tictactoe_config
from utils import launch_job, expand_node, Node
from tictactoe.keras.TicTacToeNNet import TicTacToeNNet as GameNetwork
from tictactoe.keras.TicTacToeNNet import TicTacToeSharedStorage as GameSharedStorage
from Game import ReplayBuffer
from MCTS import *

# MuZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def muzero(config: MuZeroConfig):
  storage = GameSharedStorage(config)
  replay_buffer = ReplayBuffer(config)

  for _ in range(config.num_actors):
    launch_job(run_selfplay, config, storage, replay_buffer)

  train_network(config, storage, replay_buffer)

  return storage.latest_network()

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: MuZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
  while True:
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: MuZeroConfig, network: GameNetwork) -> Game:
    game = config.new_game()
    while not game.terminal() and len(game.history) < config.max_moves:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1)
        expand_node(root, game.to_play(), game.legal_actions(),
                    network.initial_inference(game.getCanonicalForm(current_observation)))
        add_exploration_noise(config, root)
        
        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history(), network)
        action = select_action(config, len(game.history), root, network)
        game.apply(action)
        game.store_search_statistics(root)
    return game


def select_action(config: MuZeroConfig, num_moves: int, node: Node,
                  network: GameNetwork):
  visit_counts = [
      (child.visit_count, action) for action, child in node.children.items()
  ]
  t = config.visit_softmax_temperature_fn(
      num_moves=num_moves, training_steps=network.training_steps())
  _, action = softmax_sample(visit_counts, t)
  return action



# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: MuZeroConfig, node: Node):
  actions = list(node.children.keys())
  noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

if __name__=="__main__":
    config = make_tictactoe_config(3)
    muzero(config)
    
