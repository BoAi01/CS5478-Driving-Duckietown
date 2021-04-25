#!/usr/bin/env python3

"""
This script will train a CNN model using imitation learning from a PurePursuit Expert.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.insert(0, '/home/aibo/duck/gym-duckietown/learning')

import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from datetime import date
from torchvision import transforms
from imitation.basic.resnet_controller import Model
import math

import argparse
import numpy as np
from gym_duckietown.envs import DuckietownEnv

# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', default='map1')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--ckpt', type=str, default='210424.pt')

args = parser.parse_args()

env = DuckietownEnv(
    map_name=args.map_name,
    domain_rand=False,
    draw_bbox=False,
    max_steps=args.max_steps,
    seed=args.seed
)

obs = env.reset()
env.render()

total_reward = 0

OBS_SIZE = (112, 112)
CHANNELS = [128, 192, 256]
FC_DROPOUT_KEEP = 0.8
LSTM_DROPOUT_KEEP = 0.8
model = Model().cuda()
model.load_state_dict(torch.load(args.ckpt))
model.eval()
print(f'load from checkpoint {args.ckpt}')

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(OBS_SIZE),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761])
])

total_reward = 0
action = [0, 0]
for i in range(args.max_steps):
    obs, reward, done, info = env.step(action)
    obs = preprocess(obs)
    action = model(obs.unsqueeze(0).cuda())
    action = action.view(-1).tolist()
    action[1] = action[1] * 2 * math.pi
    env.render()
    print(f'model output {action}')
    total_reward += reward
    print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))

print("Total Reward", total_reward)
