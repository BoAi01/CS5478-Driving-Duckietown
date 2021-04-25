#!/usr/bin/env python3

"""
This script will train a CNN model using imitation learning from a PurePursuit Expert.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

import sys
sys.path.insert(0, '/home/aibo/duck/gym-duckietown/learning')

import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from datetime import date
from torchvision import transforms
from imitation.basic.convlstm_net import ConvLSTMNet
import math
from gym_duckietown.envs import DuckietownEnv
from train_resnet import shuffle_pairs

OBS_SIZE = (112, 112)
CHANNELS = [128, 192, 256]
FC_DROPOUT_KEEP = 0.8
LSTM_DROPOUT_KEEP = 0.8
SEQ_LEN = 100
k1, k2_n = 25, 2      # for TBPTT

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(OBS_SIZE),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761])
])

def preprocesss_images(imgs):
    new_imgs = []
    for img in imgs:
        new_imgs.append(preprocess(img))
    return torch.stack(new_imgs, dim=0)


def chunk_by_len(seq_1, seq_2, length):
    assert len(seq_1) == len(seq_1), f'sequence length mismatch: len(seq_1) ! len(seq_1)'
    num_seq = len(seq_1) / length * 2
    seqs_1, seqs_2 = [], []
    for i in range(int(num_seq)):
        start_idx = random.randint(0,  len(seq_1))
        seqs_1.append(seq_1[start_idx: start_idx + length])
        seqs_2.append(seq_2[start_idx: start_idx + length])
    seqs_1 = list(filter(lambda x: len(x) == length, seqs_1))
    seqs_2 = list(filter(lambda x: len(x) == length, seqs_2))
    return seqs_1, seqs_2


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _train(args):
    # model
    model = ConvLSTMNet('/home/aibo/pretrained/resnet50.pth', OBS_SIZE, CHANNELS,
                        FC_DROPOUT_KEEP, LSTM_DROPOUT_KEEP)
    model = torch.nn.DataParallel(model).cuda()

    # train tools
    print(f'base learning rate {args.base_lr:.5f}')
    optimizer = optim.Adam(model.parameters(), lr=args.base_lr * args.batch_size, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                           min_lr=0, eps=1e-08, verbose=True)
    criterion = torch.nn.MSELoss()

    # logger
    running_loss = AverageMeter()
    val_loss = AverageMeter()

    # read data
    anno_dir = os.path.join(args.data_dir, args.map, 'anno')
    obs_dir = os.path.join(args.data_dir, args.map, 'obs')
    actions_orig, observations_orig = [], []
    for obs_file_name in os.listdir(obs_dir):
        if f"{args.map_name}" not in obs_file_name:
            continue
        anno_file_name = obs_file_name.replace("obs_", "").replace("npy", "txt")
        observations_orig.extend(np.load(os.path.join(obs_dir, obs_file_name)))
        actions_orig.extend(np.loadtxt(os.path.join(anno_dir, anno_file_name), delimiter=','))
        print(f'loaded {obs_file_name}, {anno_file_name}')

    actions_orig = torch.tensor(actions_orig)
    actions_orig[:, 1] = actions_orig[:, 1] / math.pi / 2
    observations_orig = np.array(observations_orig)
    observations_orig = preprocesss_images(observations_orig)
    print(1, actions_orig.shape, observations_orig.shape)
    # observations_orig = observations_orig.transpose(0, 3, 1, 2)
    # print(f'observations_orig shape {len(observations_orig)}, actions_orig shape {len(actions_orig)}')

    print(f'start training, checkpoint name {args.ckpt_name}, base-lr {args.base_lr}')
    for epoch in range(args.epochs):
        print(f'epoch {epoch} starts')

        # re-chunk for each epoch to avoid overfitting
        actions, observations = chunk_by_len(actions_orig, observations_orig, SEQ_LEN)

        actions, observations = shuffle_pairs(actions, observations)
        actions, observations = torch.stack(actions, dim=0), torch.stack(observations, dim=0)
        # print(2, actions.shape, observations.shape)

        effective_len = actions.size(0) // args.batch_size * args.batch_size
        actions, observations = actions[:effective_len], observations[:effective_len]
        # print(3, {actions.shape}, {observations.shape}, effective_len)
        actions, observations = actions.split(args.batch_size, dim=0), observations.split(args.batch_size, dim=0)
        # print(4, {actions.shape}, {observations.shape}, effective_len)

        for b in range(len(actions)):
            act_batch, obs_batch = actions[b].float(), observations[b].float()
            # print(4, obs_batch.shape, act_batch.shape)
            orig_states, detached_states = [], []  # queue to store states

            # one update
            for t in range(0, act_batch.size(1), k1):
                mid = obs_batch[:, t: t + k1].cuda()
                labels = act_batch[:, t: t + k1].cuda()
                # print(mid, labels)
                import pdb
                # pdb.set_trace()

                # compute predictions
                # print(5, mid.shape, labels.shape)
                outs, states = model(mid, None if len(detached_states) == 0 else detached_states[-1])

                # process states
                orig_states.append(states)
                detached_states.append(model.module.detach_states(states))
                orig_states, detached_states = orig_states[-k2_n - 1:], detached_states[-k2_n - 1:]

                # compute loss
                loss = criterion(outs, labels)
                running_loss.update(loss.item())
                loss.backward()  # backprop the loss to the last state, False still in testing
                if k2_n > 1 and t > k2_n * k1:
                    # backprop from the last state to previous k2_n states
                    for count in range(k2_n):
                        model.module.derive_grad(detached_states[-count - 2], orig_states[-count - 2])

                # optim step
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step(running_loss.avg)
        print('epoch %d, avg loss=%.5f' % (epoch, running_loss.avg))

        # Periodically save the trained model
        if epoch % 20 == 0:
            torch.save(model.module.state_dict(), f'{args.ckpt_name}.pt')

        # roll-out
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


        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(OBS_SIZE),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761])
        ])

        total_reward = 0
        action = [0, 0]
        states = None
        for i in range(args.max_steps):
            obs, reward, done, info = env.step(action)
            obs = preprocess(obs)
            action, states = model(obs.unsqueeze(0).unsqueeze(0).cuda(), None)
            action = action.view(-1).tolist()
            action[1] = action[1] * 2 * math.pi
            env.render()
            print(f'model output {action}')
            total_reward += reward
            print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))

        print("Total Reward", total_reward)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1234, type=int, help="Sets Gym, TF, and Numpy seeds")
    parser.add_argument("--episodes", default=50, type=int, help="Number of epsiodes for experts")
    parser.add_argument("--steps", default=100*30, type=int, help="Number of steps per episode")
    parser.add_argument("--batch-size", default=16, type=int, help="Training batch size")
    parser.add_argument("--epochs", default=200, type=int, help="Number of training epochs")
    parser.add_argument("--model-directory", default="models/", type=str, help="Where to save models")
    parser.add_argument("--base-lr", default=1e-5, type=int, help="Base learning rate")
    parser.add_argument("--ckpt-name", default=date.today().strftime("%y%m%d"), type=str, help="Checkpoint name")
    parser.add_argument("--map", default="map3", type=str, help="Map name")
    parser.add_argument("--data-dir", default="/home/aibo/duck_dataset", type=str, help="Dataset dir")

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    _train(args)
