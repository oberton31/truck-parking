from env import TruckEnv
import time
import os
import pygame

SAVE_PATH = "data/"
CHUNK_SIZE = 15  # number of frames per save chunk (reduced to limit memory per chunk)

import numpy as np

import multiprocessing as mp
import queue
import sys
import math

if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser

class AsyncSaver:
    def __init__(self, max_queue=10, n_workers=None):
        self.queue = mp.Queue(max_queue)
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.workers = []
        for _ in range(self.n_workers):
            p = mp.Process(target=self._worker, daemon=True)
            p.start()
            self.workers.append(p)

    def _worker(self):
        import numpy as _np
        import gc as _gc
        while True:
            item = self.queue.get()
            if item is None:
                break
            save_path, data = item
            try:
                _np.savez_compressed(save_path, **data)
            except Exception as e:
                print("Save error:", e)
            try:
                del data
            except Exception:
                pass
            _gc.collect()

    def save(self, save_path, data):
        try:
            self.queue.put_nowait((save_path, data))
        except queue.Full:
            print("Save queue full, skipping")

    def close(self):
        # send poison pills for each worker
        for _ in range(len(self.workers)):
            try:
                self.queue.put(None)
            except Exception:
                pass
        for p in self.workers:
            p.join()

class PygameController:

    def __init__(self, tick_hz=20, window_size=(200, 100)):
        pygame.init()
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please Connect Just One Joystick")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()
        pygame.display.set_mode(window_size)
        pygame.display.set_caption("Truck Keyboard Controller")

        self._parser = ConfigParser()
        self._parser.read('wheel_config.ini')
        self._steer_idx = int(
            self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(
            self._parser.get('G29 Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
        self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
        self._handbrake_idx = int(
            self._parser.get('G29 Racing Wheel', 'handbrake'))
        self._clock = pygame.time.Clock()
        self._tick_hz = tick_hz

        # reverse toggle pulse state (set when JOYBUTTONDOWN for reverse index is seen)
        self._reverse_pulse = False

    def get_action(self):
        # Process events (keep the event queue from filling)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == self._reverse_idx:
                    self._reverse_pulse = True
            

        action = [0] * 5

        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        action[0] = throttleCmd
        action[1] = brakeCmd

        action[2] = steerCmd
        action[4] = bool(jsButtons[self._handbrake_idx])

        if self._reverse_pulse:
            action[3] = 1
            self._reverse_pulse = False

        self._clock.tick(self._tick_hz)

        return action


if __name__ == "__main__":
    env = TruckEnv(max_steps=1000000)
    obs = env.reset()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(SAVE_PATH, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    controller = PygameController()
    saver = AsyncSaver(max_queue=20, n_workers=3)

    buffer = []
    chunk_idx = 0
    episode = 0

    try:
        episode_dir = os.path.join(save_dir, f"episode_{episode:03d}")
        os.makedirs(episode_dir, exist_ok=True)

        while True:
            state = obs

            action = controller.get_action()
            if action is None:
                print("Quit requested by user")
                break


            next_obs, reward, terminated, truncated = env.step(action)

            buffer.append(dict(
                images=np.stack(state[0], axis=0),
                pos=np.array(state[1], dtype=np.float32),
                vel=np.array(state[2], dtype=np.float32),
                trailer_angle=np.array(state[3], dtype=np.float32),
                goal=np.array(state[4], dtype=np.float32),

                actions=np.array(action, dtype=np.float32),
                reward=np.array(reward, dtype=np.float32),
                done=np.array(terminated or truncated, dtype=np.uint8),
                timestamp=np.array(time.time(), dtype=np.float64),

                # next_images=np.stack(next_obs[0], axis=0),
                # next_pos=np.array(next_obs[1], dtype=np.float32),
                # next_vel=np.array(next_obs[2], dtype=np.float32),
                # next_trailer_angle=np.array(next_obs[3], dtype=np.float32),
                # next_goal=np.array(next_obs[4], dtype=np.float32),
            ))

            if len(buffer) >= CHUNK_SIZE or terminated or truncated:
                save_path = os.path.join(episode_dir, f"chunk_{chunk_idx:04d}.npz")
                saver.save(save_path, {"frames": buffer})
                # delete buffer contents from memory
                buffer = []
                chunk_idx += 1

            if terminated or truncated:
                print("Episode finished, resetting env")
                # wait until all saves are done
                while not saver.queue.empty():
                    time.sleep(0.1)
                print("All saves complete for episode")
                obs = env.reset()
                episode += 1
                episode_dir = os.path.join(save_dir, f"episode_{episode:03d}")
                os.makedirs(episode_dir, exist_ok=True)
                chunk_idx = 0
            else:
                obs = next_obs
    finally:
        env.destroy()
        pygame.quit()