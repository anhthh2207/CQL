

import gym
import math
# import pybullet_envs
import numpy as np
from collections import deque
import torch
# import wandb
import argparse
from buffer import ReplayBuffer
import glob
from utils import save, collect_random
import random
from data import Data
from agent import CQLAgent
from preprocess import AtariEnv

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL-DQN", help="Run name, default: CQL-DQN")
    parser.add_argument("--env", type=str, default="Breakout-v0", help="Gym environment name, default: Breakout-v0")
    parser.add_argument("--episodes", type=int, default=210, help="Number of episodes, default: 200")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--min_eps", type=float, default=0.01, help="Minimal Epsilon, default: 4")
    parser.add_argument("--eps_frames", type=int, default=1e3, help="Number of steps for annealing the epsilon value to the min epsilon, default: 1e-5")
    parser.add_argument("--log_video", type=int, default=1, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    
    args = parser.parse_args()
    return args

def eval(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    # env = gym.make(config.env)
    env = AtariEnv(game='Breakout')
    
    env.seed(config.seed)
    env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    eps = 1.
    d_eps = 1 - config.min_eps
    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0
    
    print(env.observation_space.shape)
    # agent = CQLAgent(state_size=84*84,
    #                     action_size=env.action_space.n,
    #                     device=device)
    agent = CQLAgent(state_size=math.prod(env.observation_space.shape),
                        action_size=env.action_space.n,
                        device=device)
    agent.network.load_state_dict(torch.load('trained_models/online_Breakout-v4_1430.pth'))
    agent.network.eval()


    # collect_random(env=env, dataset=buffer, num_samples=10000)
    
    if config.log_video:
        env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)

    for i in range(1, config.episodes+1):
        # perform with the learned model
        state = env.reset()
        rewards = 0
        episode_steps = 0
        while True:
            action = agent.get_action(state, epsilon=0.01)
            steps += 1
            next_state, reward, done, _ = env.step(action[0])
            state = next_state
            rewards += reward
            episode_steps += 1
            eps = max(1 - ((steps*d_eps)/config.eps_frames), config.min_eps)
            env.render()
            if done:
                break
        

        average10.append(rewards)
        total_steps += episode_steps
        print("Episode: {} | Reward: {} | Steps: {}".format(i, rewards, steps,))
        # print("Episode: {} | Reward: {} | Steps: {}".format(i, reward_, steps,))

        if (i %10 == 0) and config.log_video:
            mp4list = glob.glob('video/*.mp4')
            if len(mp4list) > 1:
                mp4 = mp4list[-2]

if __name__ == "__main__":
    config = get_config()
    eval(config)