from runner.maddpgs.configs.base_config import BaseConfig


class MPEConfig(BaseConfig):
    def __init__(self, scenario_name, algorithm_name) -> None:
        super().__init__()
        self.env_name = 'MPE'

        # 来自train_mpe.parse_args
        self.scenario_name = scenario_name
        self.num_landmarks = 3
        self.num_agents = 2
        self.use_same_share_obs = True  # Whether to use available actions
        self.algorithm_name = algorithm_name

        # 来自train_mpe_matd3.sh
        self.episode_length = 25
        self.tau = 0.005
        self.lr = 7e-4
        self.num_env_steps = 10000000
        self.batch_size = 1000
        self.buffer_size = 500000
        self.use_reward_normlization = True
        self.use_wandb = False
  