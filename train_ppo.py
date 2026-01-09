#!/usr/bin/env python3
import argparse
import os
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize

from gym_env import TruckParkingGymEnv

class ActionStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        buf = getattr(self.model, "rollout_buffer", None)
        if buf is None or buf.pos == 0:
            return True
        
        actions = buf.actions[:buf.pos]
        # Handle both (N,4) and (N,1,4) shapes safely
        if len(actions.shape) == 2:
            throttle_mean = actions[:, 0].mean().item()
            brake_mean = actions[:, 1].mean().item()
            steer_mean = actions[:, 2].mean().item()
            reverse_mean = actions[:, 3].mean().item()
        else:
            throttle_mean = actions[:, 0, 0].mean().item()
            brake_mean = actions[:, 0, 1].mean().item()
            steer_mean = actions[:, 0, 2].mean().item()
            reverse_mean = actions[:, 0, 3].mean().item()
        
        self.logger.record("policy/throttle_mean", throttle_mean)
        self.logger.record("policy/brake_mean", brake_mean)
        self.logger.record("policy/steer_mean", steer_mean)
        self.logger.record("policy/reverse_mean", reverse_mean)
        return True

class SuccessRateCallback(BaseCallback):
    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self.episode_successes = []
        self.num_episodes = 0

    def _on_step(self) -> bool:
        # Only process during rollouts, not PPO updates
        if not self.locals or "infos" not in self.locals:
            return True
            
        infos = self.locals["infos"]
        dones = self.locals.get("dones", [False])
        env = self.training_env.envs[0].unwrapped 
        #env = self.training_env.envs[0].venv.envs[0]
        for i, (done, info) in enumerate(zip(dones, infos)):
            if done:
                success = 1.0 if info.get("goal_reached") or info.get("is_success") else 0.0
                self.episode_successes.append(success)
                self.num_episodes += 1
                
                if len(self.episode_successes) > self.window_size:
                    self.episode_successes.pop(0)
                    
                if self.episode_successes:
                    success_rate = sum(self.episode_successes) / len(self.episode_successes)
                    self.logger.record("metrics/success_rate", success_rate)
        self.logger.record("rollout/difficulty", env.base_env.difficulty)
       
        if (len(self.episode_successes) > 0 and 
            self.num_episodes > 200 and 
            sum(self.episode_successes) / len(self.episode_successes) > 0.8):
            
            env.increase_difficulty()
            print(f"Increased Difficulty to {env.base_env.difficulty}")
            self.episode_successes = []
            self.num_episodes = 0            
        return True
        
class CurriculumCallback(BaseCallback):
    def __init__(self, phase2_step: int, phase3_step: int, verbose: int = 0):
        super().__init__(verbose)
        self.phase2_step = phase2_step
        self.phase3_step = phase3_step

    def _on_step(self) -> bool:
        try:
            env = self.training_env.envs[0].unwrapped
            if self.num_timesteps >= self.phase3_step and env.phase != 3:
                env.set_phase(3)
                if self.verbose:
                    print(f"[Curriculum] Switched to phase 3 at step {self.num_timesteps}")
            elif self.num_timesteps >= self.phase2_step and env.phase != 2:
                env.set_phase(2)
                if self.verbose:
                    print(f"[Curriculum] Switched to phase 2 at step {self.num_timesteps}")
            
            self.logger.record("rollout/difficulty", getattr(env.base_env, 'difficulty', 0.0))
        except:
            pass  # Silent fail during PPO updates
        return True
    
class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if not self.locals or "infos" not in self.locals:
            return True
            
        infos = self.locals["infos"]
        dones = self.locals.get("dones", [False])
        
        for done, info in zip(dones, infos):
            if done:
                ep_info = info.get("episode", {})
                ep_rew = ep_info.get("r", None)
                ep_len = ep_info.get("l", None)
                success = info.get("goal_reached") or info.get("is_success")
                
                msg_parts = []
                if ep_len is not None: msg_parts.append(f"len={ep_len}")
                if ep_rew is not None: msg_parts.append(f"reward={ep_rew:.1f}")
                msg_parts.extend([f"success={bool(success)}"])
                
                if self.verbose:
                    print("[Episode] " + " ".join(msg_parts))
        return True


def make_env(args):
    def _init():
        env = TruckParkingGymEnv(
            max_episode_steps=args.max_episode_steps,
            phase=2,
            decision_period=4,
            stack_size=5,
            use_cameras=args.use_cameras,
            npv_max=20,
            map_location=0,
        )
        # Monitor FIRST (logs episode stats)
        env = Monitor(env)
        return env
    return _init


def linear_lr(progress_remaining: float) -> float:
    # 1e-4 at start -> 1e-5 at end
    return 1e-5 + (1e-4 - 1e-5) * progress_remaining

def ent_schedule(progress_remaining: float) -> float:
    # 0.02 -> 0.001 over training
    return 0.0005 + (0.01 - 0.0005) * progress_remaining

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on truck parking (SB3).")
    parser.add_argument("--total-steps", type=int, default=1_000_000, help="Total training timesteps.")
    parser.add_argument("--n-steps", type=int, default=2048, help="Rollout steps per update.")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="PPO learning rate.")
    parser.add_argument("--phase2-step", type=int, default=250_000, help="When to start curriculum phase 2.")
    parser.add_argument("--phase3-step", type=int, default=500_000, help="When to start curriculum phase 3.")
    parser.add_argument("--max-episode-steps", type=int, default=1000, help="Episode cap (matching paper).")
    parser.add_argument("--use-cameras", action="store_true", help="Enable RGB cameras from env.py.")
    parser.add_argument("--output", type=str, default="checkpoints/sb3_ppo_truck_phase_2", help="Checkpoint prefix.")
    parser.add_argument(
        "--bc-init",
        type=str,
        default=None,
        help="Path to a pretrained policy (e.g., behavior cloning) to warm start PPO. "
        "Accepts SB3 .zip or torch .ckpt/.pt with a policy state_dict.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    from stable_baselines3.common.env_checker import check_env
    test_env = make_env(args)()
    check_env(test_env)  
    env = DummyVecEnv([make_env(args)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=200)

    policy_kwargs = dict(net_arch=dict(pi=[512,512], vf=[512,512]))

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
                    "learning_rate": 5e-5,#linear_lr,
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
                learning_rate=5e-5,#linear_lr,#args.learning_rate,
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
            learning_rate=1e-4, #linear_lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=0.99,
            gae_lambda=0.98,
            clip_range=0.15,
            ent_coef=0.001,
            vf_coef=0.25,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log_dir,
            normalize_advantage=True,
            n_epochs=6,
            device="cuda"
        )

    callback = CallbackList([
        # CurriculumCallback(phase2_step=args.phase2_step, phase3_step=args.phase3_step, verbose=0),
        SuccessRateCallback(window_size=100, verbose=0),
        EpisodeLoggerCallback(verbose=1),
        ActionStatsCallback(verbose=0)
    ])

    try:
        model.learn(total_timesteps=args.total_steps, callback=callback)
    except KeyboardInterrupt:
        pass
    finally:
        model.save(args.output)
        env.close()


if __name__ == "__main__":
    main()
