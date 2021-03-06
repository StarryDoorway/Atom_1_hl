import gym
import logging
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow import keras
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import time
import random
from collections import deque
from CalculateCost import CalculateCost

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=500)
parser.add_argument('-n', '--num_updates', type=int, default=500)
parser.add_argument('-lr', '--learning_rate', type=float, default=7e-3)
parser.add_argument('-r', '--render_test', action='store_true', default=False)
parser.add_argument('-p', '--plot_results', action='store_true', default=False)
REPLAY_SIZE = 500  # experience replay buffer size
BATCH_SIZE = 8  # size of minibatch


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        # Sample a random categorical action from the given logits.
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        # Note: no tf.get_variable(), just simple Keras API!
        self.actor_hidden1 = kl.Dense(16, activation='relu')
        self.actor_hidden2 = kl.Dense(16, activation='relu')
        self.actor_hidden3 = kl.Dense(16, activation='relu')

        self.critic_hidden1 = kl.Dense(64, activation='relu')
        self.value = kl.Dense(1, name='value')
        # Logits are unnormalized log probabilities.
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs, **kwargs):
        # Inputs is a numpy array, convert to a tensor.
        x = tf.convert_to_tensor(inputs)
        # Separate hidden layers from the same input tensor.
        hidden_logs = self.actor_hidden1(x)
        hidden_logs = self.actor_hidden2(hidden_logs)
        hidden_logs = self.actor_hidden3(hidden_logs)


        hidden_vals = self.critic_hidden1(x)
        return self.logits(hidden_logs), self.value(hidden_vals)

    def action_value(self, obs):
        # Executes `call()` under the hood.
        logits, value = self.predict_on_batch(obs)
        action = self.dist.predict_on_batch(logits)

        # Another way to sample actions:
        #   action = tf.random.categorical(logits, 1)
        # Will become clearer later why we don't use it.
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

    # def get_weights(self):
    #   return self.actor_hidden1.get_weights()


class A2CAgent:
    def __init__(self, model, num_actions, lr=7e-3, gamma=0.95, value_c=0.5, entropy_c=1e-4):
        # `gamma` is the discount factor; coefficients are used for the loss terms.
        self.replay_buffer = deque()
        self.gamma = gamma
        self.value_c = value_c
        self.entropy_c = entropy_c
        self.num_actions = num_actions
        self.num_atoms = int(num_actions/6)
        self.model = model
        self.model.compile(
            optimizer=ko.RMSprop(lr=lr),
            # Define separate losses for policy logits and value estimate.
            loss=[self._logits_loss, self._value_loss])

        # self.model = keras.models.load_model('net_model',
        #                                      {
        #                                       '_logits_loss': self._logits_loss,
        #                                       '_value_loss':self._value_loss
        #                                       })
        # print(self.model.get_weights())

    def train(self, env, batch_sz=400, updates=600):
        # Storage helpers for a single batch of data.
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + env.observation_space.shape)
        ep_rewards = [0.0]
        next_obs = env.reset()
        train_costs = np.empty((updates,), dtype=float)

        count = 0  # the number of structrue_found
        starttime = time.time()  # record the interval between two structrue_found

        # ????????????
        losses = self.do_train_and_getloss(observations, actions, rewards, next_obs, dones, values)
        self.model.load_weights('./checkpoints/my_checkpoint_1_0330.h5')
        print("have loaded the model weights")
        print(self.model.get_weights())

        for update in range(updates):
            if (update + 1) % 40 > 10 and len(self.replay_buffer) > BATCH_SIZE * 2:
                self.train_Q_network()
                logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                # ??????cost?????????????????????
                coordinates_for_cost = next_obs.reshape((self.num_atoms, 3))
                cost = CalculateCost('R-3M', [13.5365, 13.5365, 14.5528, 90, 90, 120], coordinates_for_cost)
                train_costs[update] += cost

                ep_rewards[-1] += rewards[step]
                # ??????????????????,???????????????????????????????????????
                # if rewards[step] > 400:
                #     time_interval = time.time() - starttime
                #     count += 1
                #     fo = open("./found_structure/running_record.txt", "a+")
                #     goal = 'count:' + str(count) + '  cost:' + str(cost) + '\t' + 'time interval: ' + str(time_interval) + '\t' + str(
                #         time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + '\r\n'
                #     fo.write(goal)
                #     fo.close()
                #     print(observations[step], '   count:', count, '   cost:', cost , 'time interval: ', time_interval,
                #           time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                #     starttime = time.time()

                if dones[step]:
                    ep_rewards.append(0.0)
                    next_obs = env.reset()
                    logging.info("Episode: %03d, Reward: %03d" % (len(ep_rewards) - 1, ep_rewards[-2]))

            print('Have done: {} episodes'.format(update + 1))

            # ????????????????????????,???????????????????????????????????????
            if np.any(rewards > 490) or np.any(rewards < -490):
                self.perceive(observations, actions, rewards, next_obs, dones, values)
            print('the length of replay buffer:  ', len(self.replay_buffer))

            if (update + 1) % 50 == 0:
                # ????????????????????????
                self.model.save_weights('./checkpoints/my_checkpoint_1_0330.h5')
                print('have saved the weight  ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '\n')
                # ??????Replay_Buffer,?????????

            losses = self.do_train_and_getloss(observations, actions, rewards, next_obs, dones, values)
            logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))

        return ep_rewards, train_costs

    def test(self, env, render=False):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        return ep_reward

    def _returns_advantages(self, rewards, dones, values, next_value):
        # `next_value` is the bootstrap value estimate of the future state (critic).
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        # Advantages are equal to returns - baseline (value estimates in our case).
        advantages = returns - values
        return returns, advantages

    def _value_loss(self, returns, value):
        # Value loss is typically MSE between value estimates and returns.
        return self.value_c * kls.mean_squared_error(returns, value)

    def _logits_loss(self, actions_and_advantages, logits):
        # A trick to input actions and advantages through the same API.
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
        # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
        # `from_logits` argument ensures transformation into normalized probabilities.
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # Entropy loss can be calculated as cross-entropy over itself.
        probs = tf.nn.softmax(logits)
        entropy_loss = kls.categorical_crossentropy(probs, probs)
        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        return policy_loss - self.entropy_c * entropy_loss

    # experience replay
    def perceive(self, state, action, reward, next_state, done, value):
        # one_hot_action = np.zeros(self.num_actions)
        # one_hot_action[action] = 1
        self.replay_buffer.append((state, action, reward, next_state, done, value))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        # if len(self.replay_buffer) > BATCH_SIZE:
        #     self.train_Q_network()

    def train_Q_network(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        for i in range(BATCH_SIZE):
            observations = minibatch[i][0]
            actions = minibatch[i][1]
            rewards = minibatch[i][2]
            next_obs = minibatch[i][3]
            dones = minibatch[i][4]
            values = minibatch[i][5]

            # Step 2: calculate y
            losses = self.do_train_and_getloss(observations, actions, rewards, next_obs, dones, values)
        print('have used experience to train the network')

    def do_train_and_getloss(self, observations, actions, rewards, next_obs, dones, values):
        _, next_value = self.model.action_value(next_obs[None, :])
        returns, advs = self._returns_advantages(rewards, dones, values, next_value)
        # A trick to input actions and advantages through same API.
        acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
        # Performs a full training step on the collected batch.
        # Note: no need to mess around with gradients, Keras API handles it.
        losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
        return losses


if __name__ == '__main__':
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    env = gym.make('Atom-v0')
    model = Model(num_actions=env.action_space.n)

    agent = A2CAgent(model, env.action_space.n, args.learning_rate)
    rewards_history, train_costs = agent.train(env, args.batch_size, args.num_updates)

    print("Finished training. Testing...")
    # print("Total Episode Reward: %d out of 200" % agent.test(env, args.render_test))

    # if args.plot_results:
    plt.plot(np.arange(0, len(train_costs), 10), train_costs[::10])
    plt.xlabel('cost')
    plt.ylabel('')
    plt.show()

    plt.style.use('seaborn')
    plt.plot(np.arange(0, len(rewards_history), 10), rewards_history[::10])
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
    plt.style.use('seaborn')
