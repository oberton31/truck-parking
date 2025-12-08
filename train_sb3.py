#!/usr/bin/env python3
import argparse
import os
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_env import TruckParkingGymEnv


class CurriculumCallback(BaseCallback):
    """Switch curriculum phases based on global timesteps."""

    def __init__(self, phase2_step: int, phase3_step: int, verbose: int = 0):
        super().__init__(verbose)
        self.phase2_step = phase2_step
        self.phase3_step = phase3_step

    def _on_step(self) -> bool:
        env = self.training_env.envs[0].unwrapped  # single DummyVecEnv
        if self.num_timesteps >= self.phase3_step and env.phase != 3:
            # env.set_phase(3)
            if self.verbose:
                print(f"[Curriculum] Switched to phase 3 at step {self.num_timesteps}")
        elif self.num_timesteps >= self.phase2_step and env.phase != 2:
            # env.set_phase(2)
            if self.verbose:
                print(f"[Curriculum] Switched to phase 2 at step {self.num_timesteps}")
        return True


class SuccessRateCallback(BaseCallback):
    """Track and log rolling success rate based on env info['goal_reached']."""

    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self.episode_successes = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if done:
                success = 1.0 if info.get("goal_reached") or info.get("is_success") else 0.0
                self.episode_successes.append(success)
                if len(self.episode_successes) > self.window_size:
                    self.episode_successes.pop(0)
                success_rate = sum(self.episode_successes) / len(self.episode_successes)
                self.logger.record("metrics/success_rate", success_rate)
                if self.verbose:
                    print(f"[Success] rate over last {len(self.episode_successes)} eps: {success_rate:.3f}")
        return True


class EpisodeLoggerCallback(BaseCallback):
    """Print per-episode reward and termination type for quick monitoring."""

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if done:
                ep_info = info.get("episode", {})
                ep_rew = ep_info.get("r", None)
                ep_len = ep_info.get("l", None)
                success = info.get("goal_reached") or info.get("is_success")
                truncated = info.get("TimeLimit.truncated", False)
                msg_parts = []
                if ep_len is not None:
                    msg_parts.append(f"len={ep_len}")
                if ep_rew is not None:
                    msg_parts.append(f"reward={ep_rew:.3f}")
                msg_parts.append(f"success={bool(success)}")
                msg_parts.append(f"truncated={bool(truncated)}")
                msg = "[Episode] " + " ".join(msg_parts)
                if self.verbose:
                    print(msg)
        return True


def make_env(args):
    def _init():
        env = TruckParkingGymEnv(
            max_episode_steps=args.max_episode_steps,
            phase=1,
            decision_period=4,
            stack_size=5,
            use_cameras=args.use_cameras,
            npv_max=20,
            map_location=0,
        )
        return Monitor(env)

    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on truck parking (SB3).")
    parser.add_argument("--total-steps", type=int, default=10_000_000, help="Total training timesteps.")
    parser.add_argument("--n-steps", type=int, default=2048, help="Rollout steps per update.")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO batch size.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="PPO learning rate.")
    parser.add_argument("--phase2-step", type=int, default=250_000, help="When to start curriculum phase 2.")
    parser.add_argument("--phase3-step", type=int, default=500_000, help="When to start curriculum phase 3.")
    parser.add_argument("--max-episode-steps", type=int, default=1000, help="Episode cap (matching paper).")
    parser.add_argument("--use-cameras", action="store_true", help="Enable RGB cameras from env.py.")
    parser.add_argument("--output", type=str, default="checkpoints/sb3_ppo_truck", help="Checkpoint prefix.")
    parser.add_argument(
        "--bc-init",
        type=str,
        default=None,
        help="Path to a pretrained policy (e.g., behavior cloning) to warm start PPO. "
        "Accepts SB3 .zip or torch .ckpt/.pt with a policy state_dict.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    env = DummyVecEnv([make_env(args)])

    policy_kwargs = dict(net_arch=[512, 512])
    tb_log_dir = "runs/ppo_sb3"
    os.makedirs(tb_log_dir, exist_ok=True)

    if args.bc_init:
        print(f"Loading pretrained policy from {args.bc_init} for warm start...")
        try:
            # Preferred path: SB3-compatible archive (any extension)
            model = PPO.load(
                args.bc_init,
                env=env,
                custom_objects={"learning_rate": args.learning_rate},
                tensorboard_log=tb_log_dir,
            )
        except Exception as e:
            print(f"SB3 load failed ({e}), trying torch state_dict fallback...")
            # Fallback: create fresh PPO then load torch state_dict
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                gamma=0.999,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.001,
                verbose=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log=tb_log_dir,
                vf_coef=0.2,
            )
            ckpt = torch.load(args.bc_init, map_location="cpu")
            state = (
                ckpt.get("state_dict")
                or ckpt.get("model_state_dict")
                or ckpt.get("model_state")
                or ckpt
            )
            if not isinstance(state, dict):
                raise ValueError("Checkpoint does not contain a state_dict-like mapping.")
            missing, unexpected = model.policy.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"Warning: state_dict keys unmatched. missing={missing}, unexpected={unexpected}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.001,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log_dir,
        )

    callback = CallbackList(
        [
            CurriculumCallback(
                phase2_step=args.phase2_step,
                phase3_step=args.phase3_step,
                verbose=1,
            ),
            SuccessRateCallback(window_size=100, verbose=0),
            EpisodeLoggerCallback(verbose=1),
        ]
    )

    model.learn(total_timesteps=args.total_steps, callback=callback)
    model.save(args.output)
    env.close()


if __name__ == "__main__":
    main()
