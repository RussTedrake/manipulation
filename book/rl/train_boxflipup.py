"""
Train a policy for manipuolation.gym.envs.box_flipup
"""

import argparse
import os
import sys

import gymnasium as gym

# `multiprocessing` also provides this method, but empirically `psutil`'s
# version seems more reliable.
from psutil import cpu_count
from pydrake.all import StartMeshcat
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

import manipulation.envs.box_flipup  # no-member
import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--train_single_env", action="store_true")
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--log_path",
        help="path to the logs directory.",
        default="/tmp/BoxFlipUp/",
    )
    args = parser.parse_args()

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 5e5 if not args.test else 5,
        "env_name": "BoxFlipUp-v0",
        "env_time_limit": 10 if not args.test else 0.5,
        "local_log_dir": args.log_path,
        "observations": "state",
    }

    if args.wandb:
        run = wandb.init(
            project="BoxFlipUp-v0",
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload videos
            save_code=True,
        )
    else:
        run = wandb.init(mode="disabled")

    zip = f"data/box_flipup_ppo_{config['observations']}.zip"

    num_cpu = int(cpu_count() / 2) if not args.test else 2
    if args.train_single_env:
        meshcat = StartMeshcat()
        env = gym.make(
            "BoxFlipUp-v0",
            meshcat=meshcat,
            observations=config["observations"],
            time_limit=config["env_time_limit"],
        )
        check_env(env)
        input("Open meshcat (optional). Press Enter to continue...")
    else:
        # Use a callback so that the forked process imports the environment.
        def make_boxflipup():
            pass

            return gym.make(
                "BoxFlipUp-v0",
                observations=config["observations"],
                time_limit=config["env_time_limit"],
            )

        env = make_vec_env(
            make_boxflipup,
            n_envs=num_cpu,
            seed=0,
            vec_env_cls=SubprocVecEnv,
        )

    if args.test:
        model = PPO("MlpPolicy", env, n_steps=4, n_epochs=2, batch_size=8)
    elif os.path.exists(zip):
        model = PPO.load(zip, env, verbose=1, tensorboard_log=f"runs/{run.id}")
    else:
        model = PPO(
            "MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}"
        )

    new_log = True
    while True:
        model.learn(
            total_timesteps=100000 if not args.test else 4,
            reset_num_timesteps=new_log,
            callback=WandbCallback(),
        )
        if args.test:
            break
        model.save(zip)
        new_log = False


if __name__ == "__main__":
    sys.exit(main())
