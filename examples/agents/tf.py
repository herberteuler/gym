import argparse
from   collections import deque
import gym
from   gym import logger, spaces, wrappers
import json
import os
from   os import path
import random
from   six.moves import cPickle as pickle
import sys
import tensorflow as tf

class QLearner(object):

    def __init__(self, target, output_dir,
                 num_steps, capacity, batch_size,
                 history_size, e_greedy, gamma,
                 image_inputs = True, learning_rate = 1e-4):
        self._target = target
        self._init_env()
        self._output_dir = path.join(output_dir, 'tf_model')
        self._num_steps = num_steps
        self._capacity = capacity
        self._batch_size = batch_size
        self._history_size = history_size
        self._e_greedy = e_greedy
        self._gamma = gamma
        self._image_inputs = image_inputs
        self._learning_rate = learning_rate
        self._init_estimator()
        self._samples = deque([], capacity)

    def close(self):
        self._env.env.close()

    def auto_play(self, render = False, steps = None):
        total_reward = 0
        k = 0
        while True:
            _, _, reward, done = self._play(-1, render)
            total_reward += reward
            if done: break
            k += 1
            if steps is not None and k >= steps: break
        return total_reward

    def iterate(self):
        self.reset_env()
        _, frames, _, done = self._play(1)
        if done: return None
        total_reward = 0
        for _ in xrange(self._num_steps):
            action, frames1, reward, done = self._play(self._e_greedy)
            total_reward += reward
            self._add_sample(dict(old_frames = frames, frames = frames1,
                                  reward = reward, action = action,
                                  done = done))
            self._train()
            if done: break
            frames = frames1
        return total_reward

    def _init_env(self):
        env = gym.make(self._target)
        env.seed(0)
        env = wrappers.Monitor(env, self._output_dir, force = True)
        assert isinstance(env.action_space, spaces.Discrete), \
               f"Action space of env {self._target} is not discrete"
        self._env = env
        self._obs = deque([], self._history_size)

    def _init_estimator(self):
        def create_model():
            l = tf.keras.layers
            head = pass if self._image_inputs else []
            return tf.keras.Sequential(head + [
                l.Dense(256, activation = tf.nn.relu),
                l.Dense(action_space.n)
            ])
        def model_fn(features, labels, mode, params, config):
            model = create_model()
            if mode == tf.estimator.ModeKeys.TRAIN:
                rewards = model(features, training = True)
                loss = tf.losses.mean_squared_error(labels = labels,
                                                    predictions = rewards)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                global_step = tf.train.get_or_create_global_step()
                return tf.estimator.EstimatorSpec(
                    mode = tf.estimator.ModeKeys.TRAIN,
                    loss = loss,
                    train_op = optimizer.minimize(loss, global_step)
                )
            elif mode == tf.estimator.ModeKeys.PREDICT:
                rewards = model(features, training = False)
                predictions = {
                    'rewards': rewards,
                    'max_reward': tf.reduce_max(rewards),
                    'action': tf.argmax(rewards),
                }
                output = tf.estimator.export.PredictOutput(predictions)
                return tf.estimator.EstimatorSpec(
                    mode = tf.estimator.ModeKeys.PREDICT,
                    predictions = predictions,
                    export_outputs = { 'results': output },
                )
        self._estimator = tf.estimator.Estimator(model_fn = model_fn)

    def _play(self, e_greedy, render = True):
        total_reward = 0
        env = self._env
        obs = self._obs
        history_size = self._history_size
        action = self._act(e_greedy)
        done = False
        for _ in xrange(history_size):
            ob, reward, done, _ = env.step(action)
            total_reward += reward
            obs.append(ob)
            if done: break
        if render: env.render()
        return action, list(obs), total_reward, done

    def _act(self, e_greedy):
        if random.random() < e_greedy: return self._env.action_space.sample()
        return self._predict(list(self._obs), 'action')

    def _add_sample(self, sample): self._samples.append(sample)

    def _predict(self, frames, key):
        def input_fn(): return tf.data.Dataset.from_tensors(frames)
        return self._estimator.predict(input_fn = input_fn, predict_keys = key)

    def _train():
        n = len(self._samples)
        count = n if n < self._batch_size else self._batch_size
        samples = random.sample(list(self._samples), count)
        estimator = self._estimator
        def calc_reward(old_frames, frames, reward, action, done):
            return reward if done else \
                reward + self._gamma * self._predict(frames, 'max_reward')
        inputs = [(sample['old_frames'], calc_reward(**sample))
                  for sample in samples]
        def gen():
            for v in inputs: yield v
        def input_fn():
            return tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))
        estimator.train(input_fn = input_fn)

if __name__ == '__main__':

    logger.set_level(logger.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action = 'store_true')
    parser.add_argument('target', nargs = "?", default = "CartPole-v0")
    args = parser.parse_args()

    outdir = '/tmp/tf-agent-results'
    learner = QLearner(target = args.target, output_dir = outdir,
                       num_steps = 200, capacity = 1000, batch_size = 100,
                       history_size = 4, e_greedy = 0.1, gamma = 0.2,
                       image_inputs = False)
    episodes = 10
    k = 0
    while True:
        reward = learner.iterate()
        if reward is None: continue
        k += 1
        print('Iteration %2i. Episode reward: %7.3f' % (k, reward))
        if args.display: learner.auto_play()
        if k >= episodes: break
    learner.close()

    def writefile(fname, s):
        with open(path.join(outdir, fname), 'w') as fh: fh.write(s)
    info = dict(params = params, argv = sys.argv, env_id = env.spec.id)
    writefile('info.json', json.dumps(info))
