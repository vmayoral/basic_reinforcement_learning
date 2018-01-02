#!/bin/sh

EPISODES=100000
ENVIRONMENT="MountainCarContinuous-v0"
# PPO1 benchmark
python3 train_ppo1.py --environment $ENVIRONMENT --max_episode_steps $EPISODES
# VPG benchmark
python3 train_vpg.py --environment $ENVIRONMENT --max_episode_steps $EPISODES
