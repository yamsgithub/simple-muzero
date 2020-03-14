from typing import List
from utils import Node, Action, Player, ActionHistory, MuZeroConfig, Environment
import collections
import numpy as np

class Game(object):
  """A single episode of interaction with the environment."""

  def __init__(self, action_space_size: int, discount: float):
    self.environment = Environment()  # Game specific environment.
    self.history = []
    self.rewards = []
    self.child_visits = []
    self.root_values = []
    self.action_space_size = action_space_size
    self.discount = discount

  def terminal(self) -> bool:
    # Game specific termination rules.
    pass

  def legal_actions(self) -> List[Action]:
    # Game specific calculation of legal actions.
    return []

  def apply(self, action: Action):
    reward = self.environment.step(action)
    self.rewards.append(reward)
    self.history.append(action)

  def store_search_statistics(self, root: Node):
    sum_visits = sum(child.visit_count for child in root.children.values())
    action_space = (Action(index) for index in range(self.action_space_size))
    self.child_visits.append([
        root.children[a].visit_count / sum_visits if a in root.children else 0
        for a in action_space
    ])
    self.root_values.append(root.value())

  def make_image(self, state_index: int):
    # Game specific feature planes.
    return []

  def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int,
                  to_play: Player):
    # The value target is the discounted root value of the search tree N steps
    # into the future, plus the discounted sum of all rewards until then.
    targets = []
    for current_index in range(state_index, state_index + num_unroll_steps + 1):
      bootstrap_index = current_index + td_steps
      if bootstrap_index < len(self.root_values):
        value = self.root_values[bootstrap_index] * self.discount**td_steps
      else:
        value = 0

      if self.rewards:
        for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
          value += reward * self.discount**i  # pytype: disable=unsupported-operands

      if current_index < len(self.root_values):
        if self.rewards:
          targets.append((value, self.rewards[current_index],
                          self.child_visits[current_index]))
        else:
          targets.append((value, 0,
                          self.child_visits[current_index]))
      else:
        # States past the end of games are treated as absorbing states.
        targets.append((0, 0, []))
    return targets

  def to_play(self) -> Player:
    return Player()

  def action_history(self) -> ActionHistory:
    return ActionHistory(self.history, self.action_space_size)

class ReplayBuffer(object):

  def __init__(self, config: MuZeroConfig):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self, num_unroll_steps: int, td_steps: int):
    games = [self.sample_game() for _ in range(self.batch_size)]
    game_pos = [(g, self.sample_position(g)) for g in games]
    #print('SAMPLED GAME POS ', game_pos)
    # for (g, i) in game_pos:
    #   print('SAMPLE POS ', i)
    #   print('OBSERVATIONS ', [b.pieces for b in g.observations])
    #   print('SAMPLE BATCH EXAMPLE ', g.getCanonicalForm(g.make_image(i)))
    #   break
    return [(g.getCanonicalForm(g.make_image(i)), g.history[i:i + num_unroll_steps],
             g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
            for (g, i) in game_pos]

  def sample_game(self) -> Game:
    # Sample game from buffer either uniformly or according to some priority.
    index = np.random.choice(len(self.buffer))
    return self.buffer[index]

  def sample_position(self, game) -> int:
    # Sample position from game either uniformly or according to some priority.
    return 0

