import gym
from gym import wrappers, logger
from six.moves import cPickle as pickle
import json, sys, os
from os import path
import argparse

def do_rollout(agent, env, num_steps, render = False):
    total_new = 0
    ob = env.reset()
    for t in range(num_steps):
        a = agent.act(ob)
        ob, reward, done, _ = env.step(a)
        total_new += reward
        if render and t % 3 == 0: env.render()
        if done: break
    return total_new, t + 1

class TFAgent(object):
    def __init__(self):
        pass
    def act(self, ob):
        pass

def tf_iters():
    pass

if __name__ == '__main__':

    def writefile(fname, s):
        with open(path.join(outdir, fname), 'w') as fh: fh.write(s)

    logger.set_level(logger.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action = 'store_true')
    parser.add_argument('target', nargs = "?", default = "CartPole-v0")
    args = parser.parse_args()

    env = gym.make(args.target)
    env.seed(0)
    outdir = '/tmp/tf-agent-results'
    env = wrappers.Monitor(env, outdir, force = True)

    params = dict(env = env, episodes = 10, num_steps = 200,
                  capacity = 1000, e_greedy = 0.1)

    for (i, iterdata) in enumerate(tf_iters(**params)):
        print('Iteration %2i. Episode mean reward: %7.3f'
              % (i, iterdata['y_mean']))
        agent = TFAgent()
        if args.display:
            do_rollout(agent, env, params['num_steps'], render = True)

    info = dict(params = params, argv = sys.argv, env_id = env.spec.id)
    writefile('info.json', json.dumps(info))

    env.env.close()
