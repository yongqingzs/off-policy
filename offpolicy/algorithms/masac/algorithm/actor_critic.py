import torch
import torch.nn as nn

from offpolicy.utils.util import init, to_torch
from offpolicy.algorithms.utils.mlp import MLPBase
from offpolicy.algorithms.utils.act import ACTLayer

# constants used in baselines implementation, might need to change
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class MASAC_Critic(nn.Module):
    def __init__(self, args, central_obs_dim, central_act_dim, device):
        super(MASAC_Critic, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self.hidden_size = args.hidden_size
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        input_dim = central_obs_dim + central_act_dim

        self.mlp1 = MLPBase(args, input_dim)
        self.mlp2 = MLPBase(args, input_dim)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        self.q1_out = init_(nn.Linear(self.hidden_size, 1))
        self.q2_out = init_(nn.Linear(self.hidden_size, 1))
        
        self.to(device)

    def forward(self, central_obs, central_act):
        # ensure inputs are torch tensors
        central_obs = to_torch(central_obs).to(**self.tpdv)
        central_act = to_torch(central_act).to(**self.tpdv)

        x = torch.cat([central_obs, central_act], dim=1)

        q1 = self.mlp1(x)
        q2 = self.mlp2(x)

        q1_value = self.q1_out(q1)
        q2_value = self.q2_out(q2)

        return q1_value, q2_value

class MASAC_Discrete_Actor(nn.Module):

    def __init__(self, args, obs_dim, act_dim, device):
        super(MASAC_Discrete_Actor, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self._gain = args.gain
        self.hidden_size = args.hidden_size
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        # map observation input into input for rnn
        self.mlp = MLPBase(args, obs_dim)

        # get action from rnn hidden state
        self.act = ACTLayer(act_dim, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, x):
        # make sure input is a torch tensor
        x = to_torch(x).to(**self.tpdv)

        x = self.mlp(x)
        action_logits = self.act(x)

        return action_logits

class MASAC_Gaussian_Actor(nn.Module):
    def __init__(self, args, obs_dim, act_dim, action_space, device):
        super(MASAC_Gaussian_Actor, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self._gain = args.gain
        self.hidden_size = args.hidden_size
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        # map observation input into input for rnn
        self.mlp = MLPBase(args, obs_dim)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), self._gain)

        self.mean_layer = init_(nn.Linear(self.hidden_size, act_dim))
        self.log_std_layer = init_(nn.Linear(self.hidden_size, act_dim))

        self.to(device)

    def forward(self, x):

        x = to_torch(x).to(**self.tpdv)

        x = self.mlp(x)

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)

        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std
