#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import argparse
import os
import datetime
import random
import warnings


import time
warnings.filterwarnings("ignore",category=DeprecationWarning)# 不显示DeprecationWarning
import config
import torch
import pprint
import gymnasium as gym
from gymnasium import spaces

import numpy as np

from tianshou.utils.net.discrete import Actor, Critic
from tianshou.data import Batch, ReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.policy import DiscreteSACPolicy
from tianshou.utils import TensorboardLogger
from tianshou.trainer import OffpolicyTrainer
from tianshou.data import Collector, VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from Track_net import Track_Net
from tianshou.env import SubprocVectorEnv, DummyVectorEnv,ShmemVectorEnv
def get_args() -> argparse.ArgumentParser:
   parser = argparse.ArgumentParser()
   parser.add_argument('--headless', type=bool, default=True)
   parser.add_argument('--test', type=bool, default=True)
   parser.add_argument("--load-model", type=bool, default=True)
   parser.add_argument("--reward-threshold", type=float, default=150000)
   parser.add_argument("--seed", type=int, default=1)
   parser.add_argument("--buffer-size", type=int, default=10000)
   parser.add_argument("--actor-lr", type=float, default=1e-3)
   parser.add_argument("--critic-lr", type=float, default=1e-3)
   parser.add_argument("--alpha-lr", type=float, default=3e-4)
   parser.add_argument("--gamma", type=float, default=0.99)
   parser.add_argument("--tau", type=float, default=0.005)
   parser.add_argument("--alpha", type=float, default=0.05)
   parser.add_argument("--auto-alpha", action="store_true", default=True)
   parser.add_argument("--epoch", type=int, default=20000)
   parser.add_argument("--step-per-epoch", type=int, default=3000) #一个epoch要采集多少step的数据
   parser.add_argument("--step-per-collect", type=int, default=1024) # 每采集多少step数据进行一次更新
   parser.add_argument("--update-per-step", type=float, default=0.01) # 相当于每收集step-per-collect数据做step-per-collect*update-per-step次更新
   parser.add_argument("--batch-size", type=int, default=512)
   parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128,128,128,128])
   parser.add_argument('--prioritized-replay', action="store_true", default=True)
   parser.add_argument("--training-num", type=int, default=20)
   parser.add_argument("--test-num", type=int, default=4)
   parser.add_argument("--logdir", type=str, default="log")
   parser.add_argument("--render", type=float, default=0.0)
   parser.add_argument("--n-step", type=int, default=3)
   parser.add_argument('--buffer_alpha', type=float, default=0.6)
   parser.add_argument('--beta', type=float, default=0.4)
   parser.add_argument(
       "--device",
       type=str,
       default="cuda" if torch.cuda.is_available() else "cpu",
   )
   args = parser.parse_known_args()[0]
   return args


def test_discrete_sac(args=get_args()):
   
  
   print("******The programe is run on :",args.device,"******")

   args.state_shape = (3,250,250)
   action_space=spaces.Discrete(5)
   args.action_shape = 5

   # model
   preprocess_net = Track_Net(
        *args.state_shape, 
        get_feature = True,
        action_shape=args.action_shape,
        feature_dim=128,
        device=args.device,
        )
   actor = Actor(
       preprocess_net, 
       args.action_shape,
       hidden_sizes=args.hidden_sizes, 
       softmax_output=False,
       device=args.device,
       ).to(args.device)
   actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

   critic1 = Critic(
       preprocess_net, 
       last_size=args.action_shape, 
       hidden_sizes=args.hidden_sizes,
       device=args.device,
       ).to(args.device)
   critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)

   critic2 = Critic(
       preprocess_net, 
       last_size=args.action_shape, 
       hidden_sizes=args.hidden_sizes,
       device=args.device,
       ).to(args.device)
   critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

   # better not to use auto alpha in CartPole
   if args.auto_alpha:
       target_entropy = 0.98 * np.log(np.prod(args.action_shape))
       log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
       alpha_optim = torch.optim.Adam([log_alpha], lr=                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      args.alpha_lr)
       args.alpha = (target_entropy, log_alpha, alpha_optim)

   policy = DiscreteSACPolicy(
       actor=actor,
       actor_optim=actor_optim,
       critic1=critic1,
       action_space=action_space,
       critic1_optim=critic1_optim,
       critic2=critic2,
       critic2_optim=critic2_optim,
       tau=args.tau,
       gamma=args.gamma,
       alpha=args.alpha,
       estimation_step=args.n_step,
   )
   if args.load_model:
       policy.load_state_dict(torch.load('./log/20240125/Track_test.pth'))
       print('Policy load!')
    # train_collector.collect(n_step=args.buffer_size)
    # log
   log_path = os.path.join(args.logdir, "20240227")
   writer = SummaryWriter(log_path)
   logger = TensorboardLogger(writer)
   def save_best_fn(policy,str):
        torch.save(policy.state_dict(), os.path.join(log_path, "Track_{}.pth".format(str)))

   def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold
   if args.test == False:# trainer
      env=gym.make('Track-v1')
      train_envs =  ShmemVectorEnv([lambda: env for _ in range(args.training_num)])
      test_envs =  ShmemVectorEnv([lambda: env for _ in range(args.test_num)])
  # collector
      train_collector = Collector(
        policy,
        train_envs,
        PrioritizedVectorReplayBuffer(args.buffer_size, len(train_envs),alpha=args.buffer_alpha,beta=args.beta),
        label='train',
    )
      test_collector = Collector(policy, test_envs,label='test')
         # seed
      np.random.seed(args.seed)
      torch.manual_seed(args.seed)
      train_envs.seed(args.seed)
      test_envs.seed(args.seed)
      result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False,
    ).run()
      assert stop_fn(result.best_reward)
      pprint.pprint(result)
      """
      env_step:采集到的step
      len:每次epoch所运行的episode的平均回合长度
      n/ep:每次epoch所运行的episode数量，也就是说需要这么多的episode才能采集够单次更新所需的step 
      n/st:采集到的step数量
      rew:每次epoch所运行的episode的平均奖励值
      ] 
      """
   else:# Let's watch its performance!
      env=gym.make('Track-v1')
      policy.eval()
      collector = Collector(policy, env)
      result = collector.collect(render=args.render,n_step=100000)
      print(f"Final reward: {result['rew']}, length: {result['len']}")
  
if __name__ == "__main__":
   test_discrete_sac()