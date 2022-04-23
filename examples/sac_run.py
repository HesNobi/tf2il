import gym
import numpy as np
# from gym.wrappers.flatten_observation import FlattenObservation
# import pybullet_envs
import gym_dmc
import tf2rl

# import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)
from tf2rl.algos.sac import SAC
from tf2rl.experiments.trainer import Trainer
from tf2rl.misc.prepare_output_dir import prep_log_tag
from tf2rl.envs.reward_wrapper import *
from tf2rl.envs.absorbing_wrapper import AbsorbingWrapper
from tf2rl.envs.raw_pixels_gym import ControlPixelWrapper


if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = SAC.get_argument(parser)
    # parser.add_argument('--env-name', type=str, default="HalfCheetah-v2", required=True)
    parser.add_argument('--env-name', type=str, required=True)
    parser.set_defaults(n_warmup=10000)
    # parser.set_defaults(max_steps=1e6)
    args = parser.parse_args()

    units = [int(item) for item in args.units.split(',')]

    args.logdir += prep_log_tag(args, "sac", units, "RefRun")

    env = gym.make(args.env_name)
    if args.pixels:
        env = ControlPixelWrapper(env, n_stack=3, is_gray=True, width=48, height=48)

    if args.reward_dir:
        env = AbsorbingWrapper(env, indicator=0.1)
        # env = GailExternalReward(env, args.reward_dir, [128,128])
        # env = GailEQLExternalReward(env, args.reward_dir, 2, [1,1], [['div'], ['id', 'sin', 'cos', 'sig', 'log', 'mult']], gpu=-1)
        # env = GailEQLExternalReward(env, args.reward_dir, 1, [1], [['log']], gpu=0)
        # env = AirlEQLExternalReward(env, args.reward_dir, 2, [1, 1], [['div','log'], ['id', 'sin', 'cos', 'sig', 'log', 'mult']],
        #                             units=[64,64] , gpu=0, state_only=True)
        # env = AirlExternalReward(env, args.reward_dir, [128,128], state_only=True)
        # env = GaifoExternalReward(env, args.reward_dir, [128,128])
        # env = GaifoEQLExternalReward(env, args.reward_dir, 2, [1,1], [['div','log'], ['id', 'sin', 'cos', 'sig', 'log', 'mult']], gpu=0)
        # env = WeightAtolApply(env, atol=None, drop=0.5, is_show=True)
        # env_visual = MazeRewardVisual(env)
        # [['div','log'],['div','log']]
        #[['div'], ['id', 'sin', 'cos', 'sig', 'log', 'mult']]

    test_env = env

    policy = SAC(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        max_action=env.action_space.high[0],
        gpu=0 if len(env.observation_space.shape) > 2 else args.gpu,
        actor_units=units,
        critic_units=units,
        n_warmup=args.n_warmup,
        batch_size=args.batch_size,
        auto_alpha=args.auto_alpha_disable,
        alpha=args.alpha,
        lr=args.lr,
        lr_alpha=args.lr_alpha,
        is_pixels= True if len(env.observation_space.shape) > 2 else False,
        memory_capacity= int(1e6) if len(env.observation_space.shape) > 2 else args.memory_capacity
        )

    trainer = Trainer(policy, env, args, test_env=test_env, is_episodic=False)
    if args.evaluate:
        trainer.evaluate_policy_continuously()
    else:
        trainer()