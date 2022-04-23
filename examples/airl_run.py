import gym
import d4rl
import tf2rl

# from tf2rl.algos.sac import SAC
from tf2rl.algos.sac_irl import SAC_IRL as SAC
from tf2rl.algos.airl import AIRL

# from tf2rl.experiments.airl_trainer import AIRLTrainer
from tf2rl.experiments.airl_trainer_v2 import AIRLTrainerV2 as AIRLTrainer
from tf2rl.experiments.utils import restore_latest_n_traj
from tf2rl.misc.prepare_output_dir import prep_log_tag
from tf2rl.envs.absorbing_wrapper import AbsorbingWrapper

if __name__ == '__main__':
    parser = AIRLTrainer.get_argument()
    parser = AIRL.get_argument(parser)
    parser = SAC.get_argument(parser)
    # parser.add_argument('--env-name', type=str, default="Pendulum-v0")
    parser.add_argument('--env-name', type=str, required=True)
    args = parser.parse_args()

    units_ail = [int(item) for item in args.units_ail.split(',')]
    units = [int(item) for item in args.units.split(',')]

    args.logdir += prep_log_tag(args, "airl", "sac", units_ail, units, "DEBUG_STAGE")

    if args.expert_path_dir is None:
        print("Please generate demonstrations first")
        print("python examples/run_sac.py --env-name=Pendulum-v0 --save-test-path --test-interval=50000")
        exit()

    env = gym.make(args.env_name)
    env = AbsorbingWrapper(env, indicator=0.1)
    test_env = env

    policy = SAC(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        max_action=env.action_space.high[0],
        gpu=args.gpu,
        actor_units=units,
        critic_units=units,
        n_warmup=args.n_warmup,
        batch_size=args.batch_size,
        auto_alpha=args.auto_alpha_disable,
        alpha=args.alpha,
        lr=args.lr,
        lr_alpha=args.lr_alpha,
        is_absorbed_state=args.absorbed_state_disable
        )

    irl = AIRL(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        units=units_ail,
        batch_size=args.batch_size_ail,
        gpu=args.gpu,
        lr=args.lr_ail,
        is_debug=args.is_debug,
        state_only=args.state_only,
        # discount=0.9
    )

    # sample_size = int(20*1000)
    # # expert_dataset = d4rl.qlearning_dataset(gym.make('walker2d-expert-v2'))
    # trainer = AIRLTrainer(policy, env, args, irl, expert_dataset["observations"][-sample_size:, :],
    #                      expert_dataset["next_observations"][-sample_size:, :], expert_dataset["actions"][-sample_size:, :],
    #                      test_env, is_episodic=True, discriminator_noise=0.15)

    expert_trajs = restore_latest_n_traj(args.expert_path_dir,
                                         n_path=args.n_expert_trajectories,
                                         max_steps=1000,
                                         is_logps=True,
                                         env = env)
    trainer = AIRLTrainer(policy, env, args, irl, expert_trajs["obses"],
                         expert_trajs["next_obses"], expert_trajs["acts"], expert_trajs["logps"],
                          test_env, is_episodic=args.absorbed_state_disable,
                          discriminator_noise=args.discriminator_noise)
    trainer()