from utils import Action, MuZeroConfig, NetworkOutput
from Game import ReplayBuffer

import tensorflow as tf


class Network(object):

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

class SharedStorage(object):

  def __init__(self):
    self._networks = {}

  def latest_network(self) -> Network:
    if self._networks:
      return self._networks[max(self._networks.keys())]
    else:
      # policy -> uniform, value -> 0, reward -> 0
      return make_uniform_network()

  def save_network(self, step: int, network: Network):
    self._networks[step] = network



####### Part 2: Training #########

def train_network(config: MuZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
  network = Network()
  learning_rate = config.lr_init * config.lr_decay_rate**(
      tf.train.get_global_step() / config.lr_decay_steps)
  optimizer = tf.keras.optimizers.SGD(learning_rate, config.momentum)

  for i in range(config.training_steps):
    if i % config.checkpoint_interval == 0:
      storage.save_network(i, network)
    batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
    update_weights(optimizer, network, batch, config.weight_decay)
  storage.save_network(config.training_steps, network)


def update_weights(optimizer: tf.keras.optimizers.Optimizer, network: Network, batch,
                   weight_decay: float):
  loss = 0
  for image, actions, targets in batch:
    # Initial step, from the real observation.
    value, reward, policy_logits, hidden_state = network.initial_inference(
        image)
    predictions = [(1.0, value, reward, policy_logits)]

    # Recurrent steps, from action and previous hidden state.
    for action in actions:
      value, reward, policy_logits, hidden_state = network.recurrent_inference(
          hidden_state, action)
      predictions.append((1.0 / len(actions), value, reward, policy_logits))

      hidden_state = tf.scale_gradient(hidden_state, 0.5)

    for prediction, target in zip(predictions, targets):
      gradient_scale, value, reward, policy_logits = prediction
      target_value, target_reward, target_policy = target

      l = (
          scalar_loss(value, target_value) +
          scalar_loss(reward, target_reward) +
          tf.nn.softmax_cross_entropy_with_logits(
              logits=policy_logits, labels=target_policy))

      loss += tf.scale_gradient(l, gradient_scale)

  for weights in network.get_weights():
    loss += weight_decay * tf.nn.l2_loss(weights)

  optimizer.minimize(loss)


def scalar_loss(prediction, target) -> float:
  # MSE in board games, cross entropy between categorical values in Atari.
  return -1

######### End Training ###########


def make_uniform_network():
  return Network()
