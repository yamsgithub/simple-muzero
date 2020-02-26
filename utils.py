import collections
import typing
from typing import Dict, List, Optional
import math

MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

class MinMaxStats(object):
  """A class that holds the min-max values of the tree."""

  def __init__(self, known_bounds: Optional[KnownBounds]):
    self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
    self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

  def update(self, value: float):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      # We normalize only when we have set the maximum and minimum values.
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value

class MuZeroConfig(object):

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
               visit_softmax_temperature_fn,
               known_bounds: Optional[KnownBounds] = None):
    ### Self-Play
    self.action_space_size = action_space_size
    self.num_actors = num_actors

    self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
    self.max_moves = max_moves
    self.num_simulations = num_simulations
    self.discount = discount

    # Root prior exploration noise.
    self.root_dirichlet_alpha = dirichlet_alpha
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    # If we already have some information about which values occur in the
    # environment, we can use them to initialize the rescaling.
    # This is not strictly necessary, but establishes identical behaviour to
    # AlphaZero in board games.
    self.known_bounds = known_bounds

    ### Training
    self.training_steps = int(1000e3)
    self.checkpoint_interval = int(1e3)
    self.window_size = int(1e6)
    self.batch_size = batch_size
    self.num_unroll_steps = 5
    self.td_steps = td_steps

    self.weight_decay = 1e-4
    self.momentum = 0.9

    # Exponential learning rate schedule
    self.lr_init = lr_init
    self.lr_decay_rate = 0.1
    self.lr_decay_steps = lr_decay_steps

def make_board_game_config(action_space_size: int, max_moves: int,
                           dirichlet_alpha: float,
                           lr_init: float, num_simulations: int, num_actors: int) -> MuZeroConfig:

  def visit_softmax_temperature(num_moves, training_steps):
    if num_moves < 30:
      return 1.0
    else:
      return 0.0  # Play according to the max.

  return MuZeroConfig(
      action_space_size=action_space_size,
      max_moves=max_moves,
      discount=1.0,
      dirichlet_alpha=dirichlet_alpha,
      num_simulations=num_simulations,
      batch_size=2048,
      td_steps=max_moves,  # Always use Monte Carlo return.
      num_actors=num_actors,
      lr_init=lr_init,
      lr_decay_steps=400e3,
      visit_softmax_temperature_fn=visit_softmax_temperature,
      known_bounds=KnownBounds(-1, 1))

class Action(object):

  def __init__(self, index: int):
    self.index = index

  def __hash__(self):
    return self.index

  def __eq__(self, other):
      return self.index == other.index

  def __gt__(self, other):
    return self.index > other.index


class Player(object):
    def turn(self):
        pass

class Node(object):

  def __init__(self, prior: float):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children = {}
    self.hidden_state = None
    self.reward = 0

  def expanded(self) -> bool:
    return len(self.children) > 0

  def value(self) -> float:
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count


class ActionHistory(object):
  """Simple history container used inside the search.

  Only used to keep track of the actions executed.
  """

  def __init__(self, history: List[Action], action_space_size: int):
    self.history = list(history)
    self.action_space_size = action_space_size

  def clone(self):
    return ActionHistory(self.history, self.action_space_size)

  def add_action(self, action: Action):
    self.history.append(action)

  def last_action(self) -> Action:
    return self.history[-1]

  def action_space(self) -> List[Action]:
    return [Action(i) for i in range(self.action_space_size)]

  def to_play(self) -> Player:
    return Player()

class Environment(object):
  """The environment MuZero is interacting with."""
  
  def step(self, action):
    pass

class NetworkOutput(typing.NamedTuple):
  value: float
  reward: float
  policy_logits: Dict[Action, float]
  hidden_state: List[float]

# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(node: Node, to_play: Player, actions: List[Action],
                network_output: NetworkOutput):
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    import numpy as np
    #print('\n\n\nNODE HIDDEN STATE ', np.asarray(node.hidden_state).shape)
    node.reward = network_output.reward
    if network_output.policy_logits:
        policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    else:
        policy = {a: 1/len(actions) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)

# Stubs to make the typechecker happy.
def softmax_sample(distribution, temperature: float):
  return 0, 0

def launch_job(f, *args):
  f(*args)


def make_uniform_network():
  return Network()


