"""
Train a policy for manipuolation.gym.envs.box_flipup

Example usage:

python solutions/notebooks/rl/train_box_flipup.py --checkpoint_freq 100000 --wandb
"""

import argparse
import os
import sys
from pathlib import Path

import gymnasium as gym

# `multiprocessing` also provides this method, but empirically `psutil`'s
# version seems more reliable.
from psutil import cpu_count
from pydrake.all import StartMeshcat
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EveryNTimesteps,
    ProgressBarCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

import manipulation.envs.box_flipup  # no-member
import wandb


class OffsetCheckpointCallback(BaseCallback):
    """
    Saves checkpoints with a global step count that includes an offset, so that
    resumed training from, e.g., 3,000,000 steps will save checkpoints named
    with accumulated steps (e.g., 4,000,000 after 1,000,000 more steps).

    This callback is intended to be wrapped by EveryNTimesteps for frequency control.
    """

    def __init__(
        self,
        save_path: Path,
        name_prefix: str,
        expected_resume_steps: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.name_prefix = name_prefix
        self.expected_resume_steps = expected_resume_steps

    def _on_step(self) -> bool:
        # Determine effective offset only at save-time to avoid relying on construction order.
        loaded_steps = int(getattr(self.model, "num_timesteps", 0))
        offset = 0
        if (
            self.expected_resume_steps is not None
            and loaded_steps < self.expected_resume_steps
        ):
            offset = int(self.expected_resume_steps)
        total_steps = offset + loaded_steps
        ckpt_path = self.save_path / f"{self.name_prefix}_{total_steps}_steps.zip"
        if self.verbose > 0:
            print(f"Saving checkpoint to {ckpt_path}")
        self.model.save(str(ckpt_path))
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--train_single_env", action="store_true")
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--checkpoint_freq", type=int, default=100_000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--resume_steps",
        type=int,
        default=None,
        help="If set, resume from checkpoint at this timestep (e.g., 3000000).",
    )
    parser.add_argument(
        "--log_path",
        help="path to the logs directory.",
        default="book/rl/BoxFlipUp/logs",
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

    # Where to put checkpoints
    ckpt_dir = Path(args.log_path).parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save a checkpoint when this callback is called.
    # We'll call it via EveryNTimesteps so save_freq can be 1.
    if True:
        checkpoint_cb = OffsetCheckpointCallback(
            save_path=ckpt_dir,
            name_prefix="ppo_boxflipup",
            expected_resume_steps=args.resume_steps,
        )

    # Trigger the checkpoint exactly every 50,000 timesteps (robust to n_envs)
    every_n_timesteps = EveryNTimesteps(
        n_steps=args.checkpoint_freq, callback=checkpoint_cb
    )

    # Combine with your existing Wandb callback
    callbacks = CallbackList(
        [WandbCallback(), every_n_timesteps, ProgressBarCallback()]
    )

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
    elif (
        args.resume_steps is not None
        and (ckpt_dir / f"ppo_boxflipup_{args.resume_steps}_steps.zip").exists()
    ):
        print(f"Loading checkpoint at {args.resume_steps} steps")
        model = PPO.load(
            str(ckpt_dir / f"ppo_boxflipup_{args.resume_steps}_steps.zip"),
            env,
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
            device="cuda",
        )
    elif os.path.exists(zip):
        model = PPO.load(
            zip, env, verbose=1, tensorboard_log=f"runs/{run.id}", device="cuda"
        )
    else:
        model = PPO(
            "MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}", device="cuda"
        )

    model.learn(
        total_timesteps=3e6 if not args.test else 4,
        callback=callbacks,
    )
    model.save(zip)


if __name__ == "__main__":
    sys.exit(main())
