import os
import sys
os.chdir(sys.path[0])

from offpolicy.scripts.train.train_mpe_new import main as train_main
from offpolicy.scripts.render.render_mpe_new import main as render_main
from runner.maddpgs.configs.mpe_config import MPEConfig

TRAIN_ID = 'n'
ENV_ID = 'simple_spread'  # simple_adversary
ALGO_ID = 'rmatd3'  # "rmatd3", "rmaddpg", "rmasac", "qmix", "vdn", "matd3", "maddpg", "masac", "mqmix", "mvdn"

if __name__ == '__main__':
    args = MPEConfig(ENV_ID, ALGO_ID)
    args.num_env_steps = 5000
    args.save_interval = 1000  # only for debug
    # args.buffer_size = 64  # only for debug
    
    if TRAIN_ID == 'y':
        # 改变参数
        # args.cuda = False
        train_main(args)
    elif TRAIN_ID == 'n':
        args.use_render = True  # render也会创建results
        
        args.cuda = False
        args.model_dir = f'./results/MPE/{ENV_ID}/{ALGO_ID}/check/run1/models/'
        render_main(args)
    