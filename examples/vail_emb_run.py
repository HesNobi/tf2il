import gym
# import d4rl
import tf2rl

# from tf2rl.algos.sac import SAC
from tf2rl.algos.sac_irl import SAC_IRL as SAC
from tf2rl.algos.vail_ebm import VAIL_EBM as VAIL

# from tf2rl.experiments.irl_trainer import IRLTrainer
from tf2rl.experiments.irl_trainer_v2 import IRLTrainerV2 as IRLTrainer
from tf2rl.experiments.utils import restore_latest_n_traj
# from tf2rl.envs.utils import is_discrete, get_act_dim
from tf2rl.misc.prepare_output_dir import prep_log_tag
from tf2rl.envs.absorbing_wrapper import AbsorbingWrapper

if __name__ == '__main__':
    parser = IRLTrainer.get_argument()
    parser = VAIL.get_argument(parser)
    parser = SAC.get_argument(parser)
    # parser.add_argument('--env-name', type=str, default="Pendulum-v0")
    parser.add_argument('--env-name', type=str, required=True)
    args = parser.parse_args()

    units_ail = [int(item) for item in args.units_ail.split(',')]
    units = [int(item) for item in args.units.split(',')]
    units_decoder = [int(item) for item in args.units_decoder.split(',')]

    args.logdir += prep_log_tag(args, "vail_ebm", "sac", units_ail, units, "PC_STAGE")

    if args.expert_path_dir is None:
        print("Please generate demonstrations first")
        # print("python examples/run_sac.py --env-name=Pendulum-v0 --save-test-path --test-interval=50000")
        exit()

    env = gym.make(args.env_name)
    env = AbsorbingWrapper(env, indicator=0.1)
    test_env = env

    # env_expert = gym.make("HalfCheetah-v2")
    env_expert = gym.make(args.env_name)
    env_expert = AbsorbingWrapper(env_expert, indicator=0.1)

    print("[DEBUG] obs: "+str(env.observation_space.shape) +" -act: " + str(env.action_space.shape))

    policy = SAC(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        max_action=env.action_space.high[0],
        gpu=args.gpu,
        actor_units=tuple(units),
        critic_units=tuple(units),
        n_warmup=args.n_warmup,
        batch_size=args.batch_size,
        auto_alpha=args.auto_alpha_disable,
        alpha=args.alpha,
        lr=args.lr,
        lr_alpha=args.lr_alpha,
        is_absorbed_state = args.absorbed_state_disable,
        is_xla = args.xla
        )

    irl = VAIL(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        units=tuple(units_ail),
        decoder_units= tuple(units_decoder),
        n_latent_unit=args.latent_units,
        batch_size=args.batch_size_ail,
        gpu=args.gpu,
        lr=args.lr_ail,
        is_debug=args.debug,
        grad_penalty_coeff=args.gp_coef,
        kl_target=args.kl_target,
        n_training=args.n_training,
    )

    # sample_size = int(20*1000)
    # expert_dataset = d4rl.qlearning_dataset(gym.make('walker2d-expert-v2'), terminate_on_end=True)
    # trainer = IRLTrainer(policy, env, args, irl, expert_dataset["observations"][-sample_size:, :],
    #                      expert_dataset["next_observations"][-sample_size:, :], expert_dataset["actions"][-sample_size:, :],
    #                      test_env, is_episodic=args.absorbed_state, discriminator_noise=args.discriminator_noise)

    expert_trajs = restore_latest_n_traj(args.expert_path_dir,
                                         n_path=args.n_expert_trajectories,
                                         max_steps=1000, env=env_expert)
    trainer = IRLTrainer(policy, env, args, irl, expert_trajs["obses"],
                         expert_trajs["next_obses"], expert_trajs["acts"],
                         test_env, is_episodic=args.absorbed_state_disable,
                         # discriminator_noise=args.discriminator_noise,
                         discriminator_noise=0,
                         random_epoch=args.random_epoch_n)
    trainer()