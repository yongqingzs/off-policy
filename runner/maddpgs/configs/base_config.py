import argparse


class BaseConfig():
    """
    The configuration parser for common hyperparameters of all environment. 
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["rmappo", "mappo", "rmappg", "mappg", "trpo"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch 
        --cuda
            by default True, will use GPU to train; or else will use CPU; 
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.
    
    Env parameters:
        --env_name <str>
            specify the name of environment
        --use_obs_instead_of_state
            [only for some env] by default False, will use global state; or else will use concatenated local obs.
    
    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer. 
    
    Network parameters:
        --share_policy
            by default True, all agents will share the same network; set to make training agents use different policies. 
        --use_centralized_V
            by default True, use centralized training mode; or else will decentralized training mode.
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use PopArt to normalize rewards. 
        --use_valuenorm
            by default True, use running mean and std to normalize rewards. 
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs. 
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0 initialization for biases. or else, will use xavier uniform inilialization.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_naive_recurrent_policy
            by default False, use the whole trajectory to calculate hidden states.
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent layers ( default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.
    
    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)
    
    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss 
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm 
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.  
        --huber_delta <float>
            coefficient of huber loss.  
    
    PPG parameters:
        --aux_epoch <int>
            number of auxiliary epochs. (default: 4)
        --clone_coef <float>
            clone term coefficient (default: 0.01)
    
    Run parameters:
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate
    
    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.
    
    Eval parameters:
        --use_eval
            by default, do not start evaluation. If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.
    
    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.
    
    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    
    def __init__(self) -> None:
        self.algorithm_name = 'matd3'  # "rmatd3", "rmaddpg", "rmasac", "qmix", "vdn", "matd3", "maddpg", "masac", "mqmix", "mvdn"
        self.experiment_name = 'check'
        self.seed = int(1)
        self.cuda = True
        self.cuda_deterministic = True
        self.n_training_threads = int(1)
        self.n_rollout_threads = int(1)  # 32
        self.n_eval_rollout_threads = int(1)
        self.n_render_rollout_threads = int(1)
        self.num_env_steps = int(10e6)
        self.user_name = 'marl'
        self.use_wandb = True
        # env parameters
        self.env_name = 'StarCraft2'
        self.use_obs_instead_of_state = False
        # replay buffer parameters
        self.episode_length = int(80)
        self.buffer_size = int(5000)
        self.use_reward_normalization = False
        self.use_popart = False
        self.popart_update_interval_step = int(2)
                        
        # prioritized experience replay
        self.use_per = False  # Whether to use prioritized experience replay
        self.per_nu = float(0.9)  # Weight of max TD error in formation of PER weights
        self.per_alpha = float(0.6)  # Alpha term for prioritized experience replay
        self.per_eps = float(1e-6)  # Eps term for prioritized experience replay
        self.per_beta_start = float(0.4)  # "Starting beta term for prioritized experience replay

        # network parameters
        self.use_centralized_Q = True  # Whether to use centralized Q function
        self.share_policy = True  # "Whether agents share the same policy
        self.hidden_size = int(64)  # Dimension of hidden layers for actor/critic networks
        self.layer_N = int(1)  # Number of layers for actor/critic networks
        self.use_ReLU = True  # Whether to use ReLU
        self.use_feature_normalization = True  # Whether to apply layernorm to the inputs
        self.use_orthogonal = True  # Whether to use Orthogonal initialization for weights and 0 initialization for biases
        self.gain = float(0.01)  # The gain of last action layer
        self.use_conv1d = False  # Whether to use conv1d
        self.stacked_frames = int(1)  # Dimension of hidden layers for actor/critic networks

        # recurrent parameters
        self.prev_act_inp = False  # Whether the actor input takes in previous actions as part of its input
        self.use_rnn_layer = True  # Whether to use a recurrent policy
        self.use_naive_recurrent_policy = True  # Whether to use a naive recurrent policy
        
        # TODO now only 1 is support
        self.recurrent_N = int(1)
        self.data_chunk_length = int(80)  # Time length of chunks used to train via BPTT
        self.burn_in_time = int(0)  # "Length of burn in time for RNN training, see R2D2 paper")

        # attn parameters
        self.attn = False
        self.attn_N = int(1)
        self.attn_size = int(64)
        self.attn_heads = int(4)
        self.dropout = float(0.0)
        self.use_average_pool = True
        self.use_cat_self = True

        # optimizer parameters
        self.lr = float(5e-4)  # Learning rate for Adam
        self.opti_eps = float(1e-5)  # RMSprop optimizer epsilon (default: 1e-5)
        self.weight_decay = float(0)

        # algo common parameters
        self.batch_size = int(32)  # Number of buffer transitions to train on at once
        self.gamma = float(0.99)  # Discount factor for env
        self.use_max_grad_norm = True
        self.max_grad_norm = float(10.0)  # max norm of gradients (default: 0.5)
        self.use_huber_loss = False  # Whether to use Huber loss for critic update
        self.huber_delta = float(10.0)

        # soft update parameters
        self.use_soft_update = True  # Whether to use soft update
        self.tau = float(0.005)  # Polyak update rate
        # hard update parameters
        self.hard_update_interval_episode = int(200)  # After how many episodes the lagging target should be updated
        self.hard_update_interval = int(200)  # After how many timesteps the lagging target should be updated
        # rmatd3 parameters
        self.target_action_noise_std = 0.2  # Target action smoothing noise for matd3
        # rmasac parameters
        self.alpha = float(1.0)  # Initial temperature
        self.target_entropy_coef = float(0.5)  # Initial temperature
        self.automatic_entropy_tune = True  # Whether use a centralized critic
        # qmix parameters
        self.use_double_q = True  # Whether to use double q learning
        self.hypernet_layers = int(2)  # Number of layers for hypernetworks. Must be either 1 or 2
        self.mixer_hidden_dim = int(32)  # Dimension of hidden layer of mixing network
        self.hypernet_hidden_dim = int(64)  # Dimension of hidden layer of hypernetwork (only applicable if hypernet_layers == 2)

        # exploration parameters
        self.num_random_episodes = int(5)  # Number of episodes to add to buffer with purely random actions
        self.epsilon_start = float(1.0)  # Starting value for epsilon, for eps-greedy exploration
        self.epsilon_finish = float(0.05)  # Ending value for epsilon, for eps-greedy exploration
        self.epsilon_anneal_time = int(50000)  # Number of episodes until epsilon reaches epsilon_finish
        self.act_noise_std = float(0.1)  # Action noise

        # train parameters
        self.actor_train_interval_step = int(2)  # After how many critic updates actor should be updated
        self.train_interval_episode = int(1)  # Number of env steps between updates to actor/critic
        self.train_interval = int(100)  # Number of episodes between updates to actor/critic
        self.use_value_active_masks = False

        # eval parameters
        self.use_eval = True  # Whether to conduct the evaluation
        self.eval_interval = int(10000)  # After how many episodes the policy should be evaled
        self.num_eval_episodes = int(32)  # How many episodes to collect for each eval

        # save parameters
        self.save_interval = int(100000)  # After how many episodes of training the policy model should be saved

        # log parameters
        self.log_interval = int(1000)  # After how many episodes of training the policy model should be saved

        # pretained parameters
        self.model_dir = None 

        # +
        self.use_render = False
