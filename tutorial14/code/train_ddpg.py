from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import baselines.ddpg.training as training
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *
from baselines import logger, bench
import gym
import tensorflow as tf
import argparse
import baselines.common.tf_util as U
from mpi4py import MPI
import time

#parser
parser = argparse.ArgumentParser()
parser.add_argument('--environment', dest='environment', type=str, default='MountainCarContinuous-v0')
parser.add_argument('--num_timesteps', dest='num_timesteps', type=int, default=100000)
boolean_flag(parser, 'render_eval', default=False)
boolean_flag(parser, 'layer_norm', default=True)
boolean_flag(parser, 'render', default=False)
boolean_flag(parser, 'normalize_returns', default=False)
boolean_flag(parser, 'normalize_observations', default=True)
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--critic_l2_reg', type=float, default=1e-2)
parser.add_argument('--actor_lr', type=float, default=1e-4)
parser.add_argument('--critic_lr', type=float, default=1e-3)
boolean_flag(parser, 'popart', default=False)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--reward_scale', type=float, default=1.)
parser.add_argument('--clip-norm', type=float, default=None)
parser.add_argument('--nb_epochs', type=int, default=5)           # number of epochs
                                                                        # with default settings, perform 100K steps total
parser.add_argument('--nb_epoch_cycles', type=int, default=20)      # number of rollouts per epoch, logging
parser.add_argument('--nb_rollout_steps', type=int, default=1000)    # number of timesteps per rollout
                                                                        # per epoch cycle and MPI worker

parser.add_argument('--nb_train_steps', type=int, default=50)  # per epoch cycle and MPI worker
parser.add_argument('--nb_eval_steps', type=int, default=100)  # per epoch cycle and MPI worker
parser.add_argument('--batch_size', type=int, default=64)  # per MPI worker
# parser.add_argument('--noise_type', type=str, default='adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
parser.add_argument('--noise_type', type=str, default='ou_0.2')
boolean_flag(parser, 'evaluation', default=False)
args = parser.parse_args()

sess = U.single_threaded_session()
sess.__enter__()

# Configure things.
rank = MPI.COMM_WORLD.Get_rank()
if rank != 0:
    logger.set_level(logger.DISABLED)

# Create envs.
env = gym.make(str(args.environment))
logger.configure("/tmp/experiments/"+str(args.environment)+"/DDPG/")
env = bench.Monitor(env, logger.get_dir())
if args.evaluation and rank==0:
    eval_env = gym.make(env_id)
    eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
    env = bench.Monitor(env, None)
else:
    eval_env = None

# gym.logger.setLevel(logging.WARN)
# if evaluation and rank==0:
#     eval_env = gym.make(env_id)
#     eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
#     env = bench.Monitor(env, None)

# Parse noise_type
action_noise = None
param_noise = None
nb_actions = env.action_space.shape[-1]
for current_noise_type in args.noise_type.split(','):
    current_noise_type = current_noise_type.strip()
    if current_noise_type == 'none':
        pass
    elif 'adaptive-param' in current_noise_type:
        _, stddev = current_noise_type.split('_')
        param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
    elif 'normal' in current_noise_type:
        _, stddev = current_noise_type.split('_')
        action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
    elif 'ou' in current_noise_type:
        _, stddev = current_noise_type.split('_')
        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
    else:
        raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

# Configure components of DDPG
memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
critic = Critic(layer_norm=args.layer_norm)
actor = Actor(nb_actions, layer_norm=args.layer_norm)
# Seed everything to make things reproducible.
seed = args.seed + 1000000 * rank
logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
# tf.reset_default_graph()
set_global_seeds(seed)
env.seed(seed)
if eval_env is not None:
    eval_env.seed(seed)

# Disable logging for rank != 0 to avoid noise.
if rank == 0:
    start_time = time.time()

# Derive the different numbers for the training process
num_timesteps = args.num_timesteps
nb_rollout_steps = args.nb_rollout_steps
nb_epoch_cycles = args.nb_epoch_cycles
nb_epochs = num_timesteps//(nb_rollout_steps*nb_epoch_cycles)

# Just train
training.train(env=env,
                session=sess,
                nb_epochs=nb_epochs,
                nb_epoch_cycles=nb_epoch_cycles,
                nb_rollout_steps=nb_rollout_steps,
                render_eval=args.render_eval,
                reward_scale=args.reward_scale,
                render=args.render,
                normalize_returns=args.normalize_returns,
                normalize_observations=args.normalize_observations,
                critic_l2_reg=args.critic_l2_reg,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                popart=args.popart,
                gamma=args.gamma,
                clip_norm=args.clip_norm,
                nb_train_steps=args.nb_train_steps,
                # nb_eval_steps=args.nb_eval_steps,
                batch_size=args.batch_size,
                eval_env=eval_env, param_noise=param_noise,
                action_noise=action_noise,
                actor=actor, critic=critic,
                memory=memory,
                job_id="",
                outdir="/tmp/experiments/"+str(args.environment)+"/DDPG/")

env.close()
if eval_env is not None:
    eval_env.close()
if rank == 0:
    logger.info('total runtime: {}s'.format(time.time() - start_time))
