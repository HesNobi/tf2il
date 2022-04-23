import gym
import tf2rl
# import numpy as np
# import tensorflow as tf

# tf.random.set_seed(1234)
# np.random.seed(1234)

from tf2rl.algos.td3 import TD3
from tf2rl.experiments.trainer import Trainer
from tf2rl.misc.prepare_output_dir import prep_log_tag
from tf2rl.envs.reward_wrapper import *
from tf2rl.envs.absorbing_wrapper import AbsorbingWrapper


if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = TD3.get_argument(parser)
    # parser.add_argument('--env-name', type=str, default="Walker2d-v2")
    parser.add_argument('--env-name', type=str, required=True)
    parser.set_defaults(n_warmup=10000)
    args = parser.parse_args()

    units = [int(item) for item in args.units.split(',')]
    args.logdir += prep_log_tag(args, "td3", units, "DEBUG_STAGE")

    if "dmc" in args.env_name:
        import gym_dmc
        print("[INFO] Using DeepMind Control Suite.")
    else:
        print("[INFO] Using openAI GYM.")
    env = gym.make(args.env_name)

    if args.reward_dir:
        env = AbsorbingWrapper(env, indicator=0.1)

    test_env = env
    policy = TD3(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        max_action=env.action_space.high[0],
        gpu=0 if len(env.observation_space.shape) > 2 else args.gpu,
        actor_units=units,
        critic_units=units,
        n_warmup=args.n_warmup,
        batch_size=args.batch_size,
        lr_critic=args.lr,
        lr_actor=args.lr,
        is_pixels=True if len(env.observation_space.shape) > 2 else False,
        memory_capacity=int(1e6) if len(env.observation_space.shape) > 2 else args.memory_capacity,
        sigma=args.sigma,
        tau=args.tau,
        actor_update_freq=args.actor_update_freq,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        )

    trainer = Trainer(policy, env, args, test_env=test_env, is_episodic=False)
    if args.evaluate:
        trainer.evaluate_policy_continuously()
    else:
        trainer()