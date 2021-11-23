import argparse
import gym
import os

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

gym.envs.register(id="BoxFlipUp-v0",
                  entry_point="manipulation.envs.box_flipup:BoxFlipUpEnv")

parser = argparse.ArgumentParser(
    description='Install ToC and Navigation into book html files.')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

observations = "state"
zip = "data/box_flipup_ppo_{observations}.zip"
log = "/tmp/ppo_box_flipup/"

if __name__ == '__main__':
    num_cpu = 48 if not args.test else 2
    env = make_vec_env("BoxFlipUp-v0",
                       n_envs=num_cpu,
                       seed=0,
                       vec_env_cls=SubprocVecEnv,
                       env_kwargs={'observations': observations})
    #    env = "BoxFlipUp-v0"

    if os.path.exists(zip) and not args.test:
        model = PPO.load(zip, env, verbose=1, tensorboard_log=log)
    else:
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log)

    new_log = True
    while True:
        model.learn(total_timesteps=100000 if not args.test else 10,
                    reset_num_timesteps=new_log)
        if args.test:
            break
        model.save(zip)
        new_log = False
