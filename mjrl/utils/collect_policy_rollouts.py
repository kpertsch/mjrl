import gym
import mjrl.envs
import click 
import os
import gym
import numpy as np
import pickle
from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
import d4rl
import d4rl.kitchen_2
#import trajopt.envs

DESC = '''
Helper script to collect policy rollouts (in SPiRL data format).\n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
@click.option('--policy', type=str, help='absolute path of the policy file', default=None)
@click.option('--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('--episodes', type=int, help='number of episodes to visualize', default=20)
@click.option('--save_dir', type=str, help='directory to save rollouts to', default=None)

def main(env_name, policy, mode, seed, episodes, save_dir):
    e = GymEnv(env_name)
    e.set_seed(seed)
    if policy is not None:
        pi = pickle.load(open(policy, 'rb'))
    else:
        pi = MLP(e.spec, hidden_sizes=(32,32), seed=seed, init_log_std=-1.0)
    # render policy
    e.collect_policy_rollouts(pi, num_episodes=episodes, horizon=e.horizon, mode=mode, save_path=save_dir)

if __name__ == '__main__':
    main()

