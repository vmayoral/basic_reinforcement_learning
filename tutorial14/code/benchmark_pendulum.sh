#!/bin/sh

TIMESTEPS=100000
ENVIRONMENT="Pendulum-v0"
# PPO1 benchmark
python3 train_ppo1.py --environment $ENVIRONMENT --num_timesteps $TIMESTEPS
# VPG benchmark
python3 train_vpg.py --environment $ENVIRONMENT --num_timesteps $TIMESTEPS
# PPO2 benchmark
python3 train_ppo2.py --environment $ENVIRONMENT --num_timesteps $TIMESTEPS
# TRPO benchmark
python3 train_trpo.py --environment $ENVIRONMENT --num_timesteps $TIMESTEPS
# DDPG benchmark
python3 train_ddpg.py --environment $ENVIRONMENT --num_timesteps $TIMESTEPS
