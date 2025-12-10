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
        self.num_episodes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if done:
                self.num_episodes += 1
                success = 1.0 if info.get("goal_reached") or info.get("is_success") else 0.0
                self.episode_successes.append(success)
                if len(self.episode_successes) > self.window_size:
                    self.episode_successes.pop(0)
                success_rate = sum(self.episode_successes) / len(self.episode_successes)
                self.logger.record("metrics/success_rate", success_rate)
                if self.verbose:
                    print(f"[Success] rate over last {len(self.episode_successes)} eps: {success_rate:.3f}")
        if (len(self.episode_successes) > 0 and self.num_episodes > 200 and sum(self.episode_successes) / len(self.episode_successes) > 0.8):
            env = self.training_env.envs[0].unwrapped 
            env.increase_difficulty()
            print(f"Increased Difficulty to {env.base_env.difficulty}")
            self.episode_successes = []
            self.num_episodes = 0
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
    
class ActionStatsCallback(BaseCallback):
    """
    Logs mean/min/max of actions, log_probs, and PPO ratios to TensorBoard.
    Works with SB3 PPO.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Access the PPO rollout buffer
        buf = getattr(self.model, "rollout_buffer", None)
        if buf is None or buf.pos == 0:
            return True

        # Slice only the filled part of the rollout buffer
        actions = buf.actions[:buf.pos]

        throttle = actions[:, 0, 0]
        brake = actions[:, 0, 1]
        steer = actions[:, 0, 2]
        reverse = actions[:, 0, 3]

        throttle_mean = throttle.mean().item()
        steer_mean = steer.mean().item()
        reverse_mean = reverse.mean().item()
        brake_mean = brake.mean().item()


        # --- Log to TensorBoard ---
        self.logger.record("policy/throttle_mean", throttle_mean)
        self.logger.record("policy/steer_mean", steer_mean)
        self.logger.record("policy/brake_mean", brake_mean)
        self.logger.record("policy/reverse_mean", reverse_mean)

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

def linear_lr(progress_remaining: float) -> float:
    # 1e-4 at start -> 1e-5 at end
    return 1e-5 + (1e-4 - 1e-5) * progress_remaining

def ent_schedule(progress_remaining: float) -> float:
    # 0.02 -> 0.001 over training
    return 0.0005 + (0.01 - 0.0005) * progress_remaining

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on truck parking (SB3).")
    parser.add_argument("--total-steps", type=int, default=10_000_000, help="Total training timesteps.")
    parser.add_argument("--n-steps", type=int, default=2048, help="Rollout steps per update.")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="PPO learning rate.")
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

    policy_kwargs = dict(net_arch=[1024, 1024, 512])
    tb_log_dir = "runs/ppo_sb3"
    os.makedirs(tb_log_dir, exist_ok=True)

    if args.bc_init:
        print(f"Loading pretrained policy from {args.bc_init} for warm start...")
        try:
            # Preferred path: SB3-compatible archive (any extension)
            model = PPO.load(
                args.bc_init,
                env=env,
                custom_objects={
                    "learning_rate": linear_lr,
                    # "ent_coef": 0.0005,
                    # "vf_coef": 0.25,
                    # "clip_range": 0.2
                },                
                tensorboard_log=tb_log_dir
            )
        except Exception as e:
            print(f"SB3 load failed ({e}), trying torch state_dict fallback...")
            # Fallback: create fresh PPO then load torch state_dict
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=linear_lr,#args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                gamma=0.998,
                gae_lambda=0.98,
                clip_range=0.2,
                ent_coef=0.0005,
                verbose=0,
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
            learning_rate=linear_lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=0.998,
            gae_lambda=0.98,
            clip_range=0.15,
            ent_coef=0.0005,
            vf_coef=0.25,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log_dir,
            normalize_advantage=True,
            n_epochs=6,
            device="cuda"
        )

    callback = CallbackList(
        [
            CurriculumCallback(
                phase2_step=args.phase2_step,
                phase3_step=args.phase3_step,
                verbose=0,
            ),
            SuccessRateCallback(window_size=100, verbose=0),
            EpisodeLoggerCallback(verbose=1),
            ActionStatsCallback(verbose=0)
        ]
    )

    try:
        model.learn(total_timesteps=args.total_steps, callback=callback)
    except KeyboardInterrupt:
        pass
    finally:
        model.save(args.output)
        env.close()


if __name__ == "__main__":
    main()
