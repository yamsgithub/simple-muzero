from utils import *
from Network import *


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: MuZeroConfig, root: Node, action_history: ActionHistory,
             network: Network):
  #print('\n\nRUN MCTS\n\n')
  min_max_stats = MinMaxStats(config.known_bounds)

  for sim in range(config.num_simulations):
    history = action_history.clone()
    node = root
    search_path = [node]

    while node.expanded():
      action, node = select_child(config, node, min_max_stats)
      # x = np.zeros(history.action_space_size)
      # x[hash(action)] = 1
      # print(x.reshape(3,3))
      history.add_action(action)
      search_path.append(node)

    # Inside the search tree we use the dynamics function to obtain the next
    # hidden state given an action and the previous hidden state.
    parent = search_path[-2]
    network_output = network.recurrent_inference(parent.hidden_state,
                                                 history.last_action())
    expand_node(node, history.to_play(), history.action_space(), network_output)

    backpropagate(search_path, network_output.value, history.to_play(),
                  config.discount, min_max_stats)

# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play: Player,
                  discount: float, min_max_stats: MinMaxStats):
  for node in search_path:
    node.value_sum += value if node.to_play == to_play else -value
    node.visit_count += 1
    min_max_stats.update(node.value())

    value = node.reward + discount * value

  # Select the child with the highest UCB score.
def select_child(config: MuZeroConfig, node: Node,
                 min_max_stats: MinMaxStats):
  _, action, child = max(
      (ucb_score(config, node, child, min_max_stats), action,
       child) for action, child in node.children.items())
  return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: MuZeroConfig, parent: Node, child: Node,
              min_max_stats: MinMaxStats) -> float:
  pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                  config.pb_c_base) + config.pb_c_init
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  value_score = min_max_stats.normalize(child.value())
  return prior_score + value_score
