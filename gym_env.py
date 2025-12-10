#!/usr/bin/env python3
import math
import random
from collections import deque
from time import time
from typing import Deque, Tuple

from sympy import true

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

import env as base_env
from env import TruckEnv

# Use the same workspace bounds normalization used in ppo_train.py
WORKSPACE_BOUNDS = np.array([-69, 157, -30, 200], dtype=np.float32)
RENDER_CARLA = False

class TruckParkingGymEnv(gym.Env):
    """
    Gymnasium wrapper around TruckEnv with paper-style rewards and curriculum phases.
    Observation is a stacked vector of (pos_rel, vel, accel, trailer_angle, reverse).
    Action: [throttle_or_brake, steer, reverse_flag] in [-1, 1] for the first two, [0, 1] for reverse.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_episode_steps: int = 1000,
        phase: int = 1,
        decision_period: int = 4,
        stack_size: int = 5,
        use_cameras: bool = False,
        npv_max: int = 20,
        map_location: int = 0,
    ):
        super().__init__()
        self.stack_size = stack_size
        self.decision_period = decision_period
        self.npv_max = npv_max

        # Keep the same map and spawn logic as env.py
        self.base_env: TruckEnv = TruckEnv(
            max_steps=max_episode_steps,
            phase=phase,
            map_location=map_location,
            use_cameras=False,
        )
        # Ensure rendering follows use_cameras choice
        settings = self.base_env.world.get_settings()
        settings.no_rendering_mode = not RENDER_CARLA
        self.base_env.world.apply_settings(settings)

        self.phase = phase
        self.max_episode_steps = max_episode_steps
        self._episode_steps = 0
        self._stack: Deque[np.ndarray] = deque(maxlen=stack_size)
        self._npv_actors = []
        self._rng = np.random.default_rng()

        # Observation: pos_rel(3) + vel(2) + accel(2) + trailer_angle(1) + reverse(1)
        self._obs_dim = 9 + 8
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim * stack_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0, 0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.prev_dist = None
        #self.base_env.difficulty = 0.0


    def _normalize_angle(self, angle: float) -> float:
        """Wrap angle to [-180, 180]."""
        angle = angle % 360
        if angle > 180:
            angle -= 360
        elif angle < -180:
            angle += 360
        return angle

    def _compute_trailer_angle(self) -> float:
        cab = self.base_env.player.get_transform()
        trailer = self.base_env.playerTrailer.get_transform()
        cab_yaw = self._normalize_angle(cab.rotation.yaw)
        trailer_yaw = self._normalize_angle(trailer.rotation.yaw)
        angle = cab_yaw - trailer_yaw
        return self._normalize_angle(angle)


    def _build_observation(self) -> np.ndarray:
        pos = self.base_env.player.get_transform()
        goal = self.base_env.goal_pos
        vel = self.base_env.player.get_velocity()
        accel = self.base_env.player.get_acceleration()

        map_offset = self.base_env.map_location * 11000 / 100
        pos_xy = np.array([pos.location.x - map_offset, pos.location.y], dtype=np.float32)
        goal_xy = np.array([goal.location.x, goal.location.y], dtype=np.float32)

        pos_norm = 2 * (pos_xy - WORKSPACE_BOUNDS[[0, 2]]) / (
            WORKSPACE_BOUNDS[[1, 3]] - WORKSPACE_BOUNDS[[0, 2]]
        ) - 1
        goal_norm = 2 * (goal_xy - WORKSPACE_BOUNDS[[0, 2]]) / (
            WORKSPACE_BOUNDS[[1, 3]] - WORKSPACE_BOUNDS[[0, 2]]
        ) - 1

        pos_yaw = self._normalize_angle(pos.rotation.yaw)
        goal_yaw = self._normalize_angle(goal.rotation.yaw)
        pos_rel = np.concatenate(
            [
                goal_norm - pos_norm,
                np.array([(goal_yaw - pos_yaw) / 180.0], dtype=np.float32),
            ]
        )

        vel_xy = np.array([vel.x, vel.y], dtype=np.float32)
        accel_xy = np.array([accel.x, accel.y], dtype=np.float32)
        trailer_angle = self._compute_trailer_angle() / 180.0
        reverse_flag = 1.0 if self.base_env.control.reverse else 0.0

        cab_rays = self.base_env._compute_rays_from_transform(self.base_env.player.get_transform(), self.base_env.player)
        trailer_rays = self.base_env._compute_rays_from_transform(self.base_env.playerTrailer.get_transform(), self.base_env.playerTrailer)

        ray_obs = np.concatenate([cab_rays, trailer_rays])

        obs = np.concatenate(
            [pos_rel, vel_xy, accel_xy, np.array([trailer_angle, reverse_flag], dtype=np.float32), ray_obs]
        )
        # if self._episode_steps % 100 == 0 :
        #     print(ray_obs)
            #print(f"Trailer angle: {trailer_angle}, Pos rel: {pos_rel}, Vel: {vel_xy}, Accel: {accel_xy}, Reverse: {reverse_flag}")
        return obs.astype(np.float32)

    def _heading_error(self) -> float:
        player_yaw = self.base_env.player.get_transform().rotation.yaw
        goal_yaw =self.base_env.goal_pos.rotation.yaw
        # player_tf = self.base_env.player.get_transform()
        # goal_tf = self.base_env.goal_pos
        # map_offset = self.base_env.map_location * 11000 / 100
        # goal_loc = base_env.carla.Location(
        #     x=goal_tf.location.x + map_offset, y=goal_tf.location.y, z=goal_tf.location.z
        # )
        # forward = player_tf.get_forward_vector()
        # to_goal = goal_loc - player_tf.location
        # heading = math.degrees(
        #     math.atan2(
        #         forward.y * to_goal.x - forward.x * to_goal.y,
        #         forward.x * to_goal.x + forward.y * to_goal.y,
        #     )
        # )
        player_yaw = self._normalize_angle(player_yaw)
        goal_yaw = self._normalize_angle(goal_yaw)
        return abs(self._normalize_angle(player_yaw - goal_yaw))

    def _distance_to_goal(self) -> float:
        player_tf = self.base_env.player.get_transform()
        goal_tf = self.base_env.goal_pos
        map_offset = self.base_env.map_location * 11000 / 100
        dx = goal_tf.location.x + map_offset - player_tf.location.x
        dy = goal_tf.location.y - player_tf.location.y
        return float(math.hypot(dx, dy))

    def _at_goal(self) -> bool:
        """Check goal using tight tolerances similar to paper alignment criteria."""
        player_tf = self.base_env.player.get_transform()
        goal_tf = self.base_env.goal_pos
        map_offset = self.base_env.map_location * 11000 / 100
        dx = goal_tf.location.x + map_offset - player_tf.location.x
        dy = goal_tf.location.y - player_tf.location.y
        pos_dist = math.hypot(dx, dy)

        yaw_diff = self._normalize_angle(player_tf.rotation.yaw - goal_tf.rotation.yaw)
        angle_dist = abs(yaw_diff)
        trailer_angle = abs(self._compute_trailer_angle())
        vel = self.base_env.player.get_velocity()
        vel_mag = math.hypot(vel.x, vel.y)

        return (
            pos_dist < 3.5
            and angle_dist < 9.0
            and trailer_angle < 12.0
            and vel_mag < 0.1
        )

    def _spawn_npvs(self, count: int) -> None:
        if count <= 0:
            return
        blueprint_library = self.base_env.world.get_blueprint_library().filter("vehicle.*")
        attempts = 0
        spawned = 0
        map_offset = self.base_env.map_location * 11000 / 100
        while spawned < count and attempts < count * 5:
            attempts += 1
            bp = random.choice(blueprint_library)
            if bp.has_attribute("color"):
                color = random.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)
            x = self._rng.uniform(self.base_env.spawn_bounds[0], self.base_env.spawn_bounds[2])
            y = self._rng.uniform(self.base_env.spawn_bounds[1], self.base_env.spawn_bounds[3])
            loc = base_env.carla.Location(x=x + map_offset, y=y, z=1.0)
            yaw = random.choice([0, 90, 180, -90])
            tf = base_env.carla.Transform(loc, base_env.carla.Rotation(yaw=yaw))
            actor = self.base_env.world.try_spawn_actor(bp, tf)
            if actor:
                actor.set_autopilot(False)
                actor.apply_control(base_env.carla.VehicleControl(hand_brake=True))
                self._npv_actors.append(actor)
                self.base_env.actor_list.append(actor)
                spawned += 1

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, seed = seeding.np_random(seed)
            self._rng = np.random.default_rng(seed)
        super().reset(seed=seed)
        self._episode_steps = 0
        self.base_env.phase = self.phase
        self._destroy_npvs()
        obs_tuple = self.base_env.reset()

        if self.phase >= 2:
            npv_count = self._rng.integers(0, self.npv_max + 1)
            self._spawn_npvs(int(npv_count))

        obs_vec = self._build_observation()
        self._stack.clear()
        zero_obs = np.zeros_like(obs_vec)  # Pad with zeros (standard)
        for _ in range(self.stack_size - 1):
            self._stack.append(zero_obs)
        self._stack.append(obs_vec) 
        stacked = np.concatenate(list(self._stack))
        self.prev_dist = float(self._distance_to_goal())
        self.near_flag = False
        self._dist_window = deque(maxlen=8)
        self._dist_window.append(self.prev_dist)
        # for optional alignment incremental reward bookkeeping
        self._last_alignment = 0.0
        self._alignment_accum = 0.0
        return stacked, {"phase": self.phase}

    def _destroy_npvs(self):
        for actor in self._npv_actors:
            try:
                if hasattr(actor, "is_listening") and actor.is_listening:
                    actor.stop()
                actor.destroy()
            except Exception:
                pass
        self._npv_actors = []

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        #print(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # action = np.concatenate([action, np.ones(1)])
        # Reuse TruckEnv control path (throttle/brake, steer, reverse)
        self.base_env.apply_control(action)
        #print(action)

        for _ in range(self.decision_period):
            self.base_env.world.tick()
            # if RENDER_CARLA:
            #     time.sleep(1)

        self._episode_steps += 1
        obs_vec = self._build_observation()
        self._stack.popleft()  # Remove frame 0 (oldest)
        self._stack.append(obs_vec)  # Add new frame at end
        
        stacked_obs = np.concatenate(list(self._stack))

        reward, terminated, truncated, info = self._compute_reward()
        self.prev_dist = float(self._distance_to_goal())

        return stacked_obs, reward, terminated, truncated, info

    def _compute_reward(self) -> Tuple[float, bool, bool, dict]:
        terminated = False
        truncated = False
        info = {}

        dist = self._distance_to_goal()
        heading_err = self._heading_error()
        trailer_angle = abs(self._compute_trailer_angle())
        at_goal = self._at_goal()

        # update dist window for trend detection
        prev_dist = getattr(self, "prev_dist", dist)
        self._dist_window.append(dist)
        # trend = previous average - current average (positive if getting closer)
        if len(self._dist_window) >= 2:
            trend = (self._dist_window[0] - self._dist_window[-1])  # positive -> approaching
        else:
            trend = prev_dist - dist

        reward = 0.0

        # -----------------------------
        # 1) Small constant time penalty (prevents infinite stalling)
        # -----------------------------
        reward += -0.005   # -5 over 1000 steps if agent does nothing

        # -----------------------------
        # 2) Gated progress reward (only when there is clear approaching trend)
        # -----------------------------
        # require a little rolling approach to grant progress reward (prevents micro-farming)
        if trend > 0.02:  # empirical; agent must reduce distance by ~0.02m over window to qualify
            progress = max(min(prev_dist - dist, 1.0), -1.0)  # clamp per-step delta
            reward += 0.25 * progress   # meaningful per-step incentive

            # small extra bonus for steady approach speed (discourage tiny micro-steps)
            reward += 0.02 * min(1.0, (trend / 0.1))
            allow_alignment = True
        else:
            # no progress reward if not truly approaching
            allow_alignment = False

        # -----------------------------
        # 3) Soft safety penalties (always apply)
        # -----------------------------
        # clamp angles so penalty can't explode
        reward -= 0.002 * min(abs(heading_err), 30.0)
        reward -= 0.004 * min(abs(trailer_angle), 30.0)

        # jackknife: terminate + moderate penalty (bounded)
        if trailer_angle > 55.0:
            reward += -3.0
            terminated = True

        # collisions (phase-sensitive)
        if self.base_env.collision:
            if self.phase == 1:
                pass
            elif self.phase == 2:
                reward += -0.2
            else:
                reward += -1.0
                terminated = True
            self.base_env.collision = False

        # -----------------------------
        # 4) Gated alignment reward (only when making progress)
        # -----------------------------
        if allow_alignment:
            # continuous bounded alignment metric in [0,1]
            alignment = (1.0 - min(heading_err / 90.0, 1.0))
            trailer_alignment = (1.0 - min(trailer_angle / 90.0, 1.0))
            align_score = 0.5 * (alignment + trailer_alignment)  # in [0,0.5]
            # give small incremental alignment reward, but cap total acc per episode
            align_inc = 0.05 * align_score
            # track accumulated alignment to avoid farming
            self._alignment_accum = getattr(self, "_alignment_accum", 0.0)
            max_align_total = 2.0
            allowed = max(0.0, max_align_total - self._alignment_accum)
            inc = min(align_inc, allowed)
            reward += inc
            self._alignment_accum += inc

        # -----------------------------
        # 5) Encourage motion modestly (not exploitable)
        # -----------------------------
        vel = self.base_env.player.get_velocity()
        vel_mag = math.hypot(vel.x, vel.y)
        if 0.05 < vel_mag < 5.0:
            reward += 0.01

        # -----------------------------
        # 6) Success reward (large, but bounded)
        # -----------------------------
        if at_goal:
            terminated = True
            # strong success reward but not astronomically large (suitable for 1000 steps)
            reward += 150.0
            # small extra shaping for better alignment (bounded)
            close_dist_bonus = max(0.0, 5.0 * (3.5 - dist) / 3.5)      # up to +5
            good_heading_bonus = max(0.0, 5.0 * (9.0 - heading_err) / 9.0)
            good_trailer_bonus = max(0.0, 5.0 * (12.0 - trailer_angle) / 12.0)
            reward += (close_dist_bonus + good_heading_bonus + good_trailer_bonus)

        # -----------------------------
        # 7) timeout (truncated)
        # -----------------------------
        if self._episode_steps >= self.max_episode_steps:
            print(f"Distance to goal: {dist}, Heading Error: {float(heading_err)}, Trailer Angle: {float(trailer_angle)}")
            print(f"Trailer Yaw: {self.base_env.playerTrailer.get_transform().rotation.yaw}, Cab Yaw: {self.base_env.player.get_transform().rotation.yaw}, Goal Yaw: {self.base_env.goal_pos.rotation.yaw}")
            truncated = True

        # Save prev_dist for next step
        self.prev_dist = dist

        info.update(
            {
                "distance": dist,
                "heading_error": heading_err,
                "trailer_angle": trailer_angle,
                "phase": self.phase,
                "goal_reached": at_goal,
                "is_success": at_goal,
                "dist_trend": float(trend),
                "alignment_accum": float(getattr(self, "_alignment_accum", 0.0)),
            }
        )

        return float(reward), terminated, truncated, info


    # def _compute_reward(self) -> Tuple[float, bool, bool, dict]:
    #     terminated = False
    #     truncated = False
    #     info = {}

    #     dist = self._distance_to_goal()
    #     heading_err = self._heading_error()
    #     trailer_angle = abs(self._compute_trailer_angle())
    #     at_goal = self._at_goal()
    #     reward = 0
    #     # Paper-style dense terms: distance and heading in [-0.1, 0]
    #     # max_dist = 50.0  # parking area size
    #     # distance_reward = -0.5 * min(dist / max_dist, 1.0)
    #     # reward += distance_reward
    #     # heading_reward = -0.1 * min(heading_err / 180.0, 1.0)
    #     # reward = distance_reward + heading_reward

    #     # # Jackknife discouragement: penalize large cab-trailer angle
    #     # if trailer_angle > 30.0:
    #     #     reward += -0.1 * min((trailer_angle - 30.0) / 60.0, 1.0)

    #     # brake = self.base_env.control.brake
    #     # reward -= 0.01 * brake

    #     prev_dist = getattr(self, "prev_dist", dist)
    #     progress = prev_dist - dist  # positive if moving toward the goal
    #     reward += 5.0 * float(progress)
    #     HEADING_COEFF = 0.002  # ~0.36 worst-case over 180 deg
    #     reward -= HEADING_COEFF * float(heading_err)

    #     TRAILER_COEFF = 0.002
    #     reward -= TRAILER_COEFF * float(trailer_angle)
        
    #     # --- 5) Jackknife discouragement: stronger past threshold but still small
    #     if trailer_angle > 45.0:
    #         reward += -2 * min((trailer_angle - 45.0) / 45.0, 1.0)

    #     vel = self.base_env.player.get_velocity()

    #     vel_mag = math.hypot(vel.x, vel.y)
    #     if vel_mag > 0.1 and vel_mag < 5.0:
    #         reward += 0.2 
        

    #     if dist < 4 and not self.near_flag:
    #         self.near_flag = True
    #         alignment = 1.0 - min(heading_err / 90.0, 1.0)  # 1.0=perfect, 0=bad
    #         trailer_alignment = 1.0 - min(float(trailer_angle) / 90, 1.0)
    #         reward += 2 * (alignment + 1.25 * trailer_alignment)

    #     # Collision penalty per curriculum phase
    #     if self.base_env.collision:
    #         if self.phase == 1:
    #             pass  # no penalty
    #         elif self.phase == 2:
    #             reward += -0.1
    #         else:
    #             reward += -1.0
    #             terminated = True
    #         self.base_env.collision = False

    #     # Goal and alignment bonuses
    #     if at_goal:
    #         terminated = True
    #         reward += 1000.0
    #         close_dist_bonus = max(0.0, 50.0 * (3.5 - dist) / 3.5)      # up to +50 if very close
    #         good_heading_bonus = max(0.0, 50.0 * (9.0 - heading_err) / 9.0)
    #         good_trailer_bonus = max(0.0, 50.0 * (12.0 - trailer_angle) / 12.0)

    #         reward += close_dist_bonus + good_heading_bonus + good_trailer_bonus
    #         # if trailer_angle < 8.0 and heading_err < 5.0:
    #         #     reward += 10.0  # alignment bonus
    #         print(f"Goal bonuses: Dist {close_dist_bonus}, Heading {good_heading_bonus}, Trailer {good_trailer_bonus}")

    #     if self._episode_steps >= self.max_episode_steps:
    #         print(f"Distance to goal: {dist}, Heading Error: {float(heading_err)}, Trailer Angle: {float(trailer_angle)}")
    #         print(f"Trailer Yaw: {self.base_env.playerTrailer.get_transform().rotation.yaw}, Cab Yaw: {self.base_env.player.get_transform().rotation.yaw}, Goal Yaw: {self.base_env.goal_pos.rotation.yaw}")
    #         truncated = True
    #     reward *= 0.05
    #     info.update(
    #         {
    #             "distance": dist,
    #             "heading_error": heading_err,
    #             "trailer_angle": trailer_angle,
    #             "phase": self.phase,
    #             "goal_reached": at_goal,
    #             "is_success": at_goal,
    #         }
    #     )
    #     return float(reward), terminated, truncated, info

    def set_phase(self, phase: int):
        """Update curriculum phase."""
        self.phase = int(phase)
        self.base_env.phase = int(phase)
    
    def increase_difficulty(self):
        self.base_env.difficulty = min(self.base_env.difficulty+0.05, 1)

    def close(self):
        self._destroy_npvs()
        try:
            self.base_env.destroy()
        except Exception:
            pass
