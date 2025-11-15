#!/usr/bin/env python

from __future__ import print_function

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import cv2
import numpy as np
import time
import math

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
RENDER_CARLA = True # SET TO FALSE FOR TRAINING

class TruckEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT

    # Phase 0: no additional vehicles, but penalized if collides with building or itself
    # Phase 1: add other vehicles but minor penalty for collisions
    # Phase 3: full penalties for collisions, and episode terminates
    def __init__(self, max_steps=10000, goal_idx=None, phase=1):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world('Town10HD')

        # Taken from example file
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713

        self._actor_generation = "2"
        # Get a random blueprint.
        self.blueprint = self._get_actor_blueprints(self.world, "dafxf", self._actor_generation)[0]
        self.blueprint.set_attribute('role_name', 'truck')
        if self.blueprint.has_attribute('color'):
            color = random.choice(self.blueprint.get_attribute('color').recommended_values)
            self.blueprint.set_attribute('color', color)
        if self.blueprint.has_attribute('driver_id'):
            driver_id = random.choice(self.blueprint.get_attribute('driver_id').recommended_values)
            self.blueprint.set_attribute('driver_id', driver_id)
        if self.blueprint.has_attribute('is_invincible'):
            self.blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if self.blueprint.has_attribute('speed'):
            self.player_max_speed = float(self.blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(self.blueprint.get_attribute('speed').recommended_values[2])
        # get trailer blueprint
        self.blueprintTrailer = self._get_actor_blueprints(self.world, "trailer", self._actor_generation)[0]
        self.blueprintTrailer.set_attribute('role_name', 'hero-trailer')
    
        self.actor_list = []
        self.image_list = []

        self.control = carla.VehicleControl()
        self.steer_cache = 0.0 # stores the prior steering values
        self.max_steps = max_steps
        self.curr_steps = 0

        self.action_ndim = 5 # If change action space, need to change this

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 Hz control steps
        settings.no_rendering_mode = not RENDER_CARLA
        self.world.apply_settings(settings)

        self.goal_pos = None

        self.goal_points = [
            (carla.Transform(carla.Location(-43.5, 188.5, 0.06), carla.Rotation(0, -90, 0)), "Bottom Right"),
            (carla.Transform(carla.Location(-54, 188.5, 0.06), carla.Rotation(0, -90, 0)), "Bottom Center"),
            (carla.Transform(carla.Location(-63.5, 188.5, 0.06), carla.Rotation(0, -90, 0)), "Bottom Left"),
            (carla.Transform(carla.Location(-43.5, 182.5, 0.06), carla.Rotation(0, -180, 0)), "Right Top"),
            (carla.Transform(carla.Location(-43.5, 192.5, 0.06), carla.Rotation(0, -180, 0)), "Right Bottom")
        ]

        self.goal_idx = goal_idx
        self.spawn_bounds = [-65, 157, -40, 188]  # x_min, y_min, x_max, y_max
        self.collision = False
        self.phase = phase

    def reset(self):
        # TODO: define spawn point and goal point
        print("Resetting Env")
        self.curr_steps = 0
        self.destroy()


        if (self.goal_idx is not None):
            self.goal_pos = self.goal_points[self.goal_idx][0]
            print(f"Using provided goal: {self.goal_points[self.goal_idx][1]}")
        else:
            rand_goal = random.choice(self.goal_points)
            self.goal_pos = rand_goal[0]
            print(f"Using random goal: {rand_goal[1]}")

        self.actor_list = []

        self.player = None
        self.playerTrailer = None
        while self.playerTrailer is None or self.player is None:
            if self.player is not None: 
                self.player.destroy()
            if self.playerTrailer is not None:
                self.playerTrailer.destroy()
            self.world.tick()
            self.player = None
            self.playerTrailer = None
            spawn_point = carla.Transform()
            spawn_point.location.x = random.uniform(self.spawn_bounds[0], self.spawn_bounds[2])
            spawn_point.location.y = random.uniform(self.spawn_bounds[1], self.spawn_bounds[3])
            spawn_point.location.z = 1.0
            spawn_point.rotation.yaw = random.uniform(-180, 180)

            self.playerTrailer = self.world.try_spawn_actor(self.blueprintTrailer, spawn_point)

            forwardVector = spawn_point.get_forward_vector() * 5.2
            spawn_point.location += forwardVector

            self.player = self.world.try_spawn_actor(self.blueprint, spawn_point) 
            self.world.tick()

        print(f"Spawned Truck and Trailer at ({self.player.get_transform().location.x:.2f}, {self.player.get_transform().location.y:.2f})")
        self.actor_list.append(self.playerTrailer)
        self.actor_list.append(self.player)

        self.collision = False

        # spawn cameras
        self.rgb_sensors = self._spawn_cameras()

        pos = self.player.get_transform() # it appears the position for both of these is the same, but the rotation varies
        pos_trailer = self.playerTrailer.get_transform()

        trailer_angle = pos.rotation.yaw - pos_trailer.rotation.yaw # right hand rule from cab to trailer
        vel = self.player.get_velocity()

        collision_blueprint = self.world.get_blueprint_library().find('sensor.other.collision')
        self.player_collision_sensor = self.world.spawn_actor(collision_blueprint, carla.Transform(), attach_to=self.player)


        self.player_collision_sensor.listen(lambda event: self._on_collision(event))
        self.actor_list.append(self.player_collision_sensor)

        self.playerTrailer_collision_sensor = self.world.spawn_actor(collision_blueprint, carla.Transform(), attach_to=self.playerTrailer)

        self.playerTrailer_collision_sensor.listen(lambda event: self._on_collision(event))
        self.actor_list.append(self.playerTrailer_collision_sensor)

        self.player.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        while not all(img is not None for img in self.image_list):
            self.world.tick()

        pos_list = [pos.location.x, pos.location.y, pos.location.z, pos.rotation.pitch, pos.rotation.yaw, pos.rotation.roll]
        vel_list = [vel.x, vel.y, vel.z]

        reverse = False


        return (self.image_list, pos_list, vel_list, trailer_angle, reverse, self.goal_pos) # TODO: talk about this some more

    def step(self, action):
        # TODO: Discuss action space
        # For now, discrete action space that mirror the manual control
        # 0: throttle, 1: brake, 2: steering command, 3: reverse toggle, 4: handbrake toggle
        self.curr_steps += 1

        self.control.steer = action[2]
        self.control.brake = action[1]
        self.control.throttle = action[0]

        if (bool(action[3])):
            self.control.reverse = not self.control.reverse

        self.control.hand_brake = bool(action[4])

        self.player.apply_control(self.control)

        # advance env
        self.world.tick()

        reward = 0
        terminated = False
        truncated = False


        if self._at_goal():
            terminated = True
            if RENDER_CARLA:
                print("At Goal!")
            reward += 15.0

        collision = self.collision
        if self.collision:
            if self.phase == 2:
                terminated = True
                reward = -1
            elif self.phase == 1:
                reward += -0.1
            self.collision = False # reset collision flag for next step
        
        # dense rewards for distance, orientation, trailer angle
        player_pos = self.player.get_transform()
        dist_to_goal = math.sqrt((self.goal_pos.location.x - player_pos.location.x)**2 + (self.goal_pos.location.y - player_pos.location.y)**2)
        reward += min(0, -dist_to_goal / 200) # should roughly be between 0 and -0.2 (max values of dist are around 40)

        player_yaw = player_pos.rotation.yaw % 360
        if player_yaw > 180: player_yaw -= 360
        elif player_yaw < -180: player_yaw += 360

        goal_yaw = self.goal_pos.rotation.yaw % 360
        if goal_yaw > 180: goal_yaw -= 360
        elif goal_yaw < -180: goal_yaw += 360

        yaw_diff = player_yaw - goal_yaw
        if yaw_diff > 180: yaw_diff -= 360
        elif yaw_diff < -180: yaw_diff += 360
        yaw_diff = abs(yaw_diff)
        reward += min(0, yaw_diff / 180 / 10) # should be between 0 and -0.1
        trailer_pos = self.playerTrailer.get_transform()
        trailer_angle_error = np.abs(player_pos.rotation.yaw % 360 - trailer_pos.rotation.yaw % 360) # this error should be between 0 and 90
        trailer_angle_error = min(trailer_angle_error, 360 - trailer_angle_error)  # account for wrap-around
        reward += min(0, -trailer_angle_error / 90 / 10)

        if self.curr_steps >= self.max_steps:
            truncated = True        

        if self.SHOW_CAM:
            for i in range(len(self.image_list)):
                cv2.imshow(f"img{i}", self.image_list[i])
                cv2.waitKey(1)
        
        pos = self.player.get_transform()
        pos_trailer = self.playerTrailer.get_transform()

        # should be between -180 and 180
        cab_yaw = pos.rotation.yaw % 360
        if cab_yaw > 180: cab_yaw -= 360
        elif cab_yaw < -180: cab_yaw += 360

        trailer_yaw = pos_trailer.rotation.yaw % 360
        if trailer_yaw > 180: trailer_yaw -= 360
        elif trailer_yaw < -180: trailer_yaw += 360

        trailer_angle = cab_yaw - trailer_yaw
        if trailer_angle > 180: trailer_angle -= 360
        elif trailer_angle < -180: trailer_angle += 360

        vel = self.player.get_velocity()

        pos_list = [pos.location.x, pos.location.y, pos.location.z, pos.rotation.pitch, pos.rotation.yaw, pos.rotation.roll]
        vel_list = [vel.x, vel.y, vel.z]

        obs = (self.image_list, pos_list, vel_list, trailer_angle, self.control.reverse, self.goal_pos)

        if RENDER_CARLA and self.curr_steps % 100 == 0:
            print(f"Step: {self.curr_steps}, Reward: {reward:.3f}, Pos: ({pos.location.x:.2f}, {pos.location.y:.2f}), Dist to Goal: {dist_to_goal:.2f}, Yaw: {pos.rotation.yaw:.2f}, Yaw Diff: {yaw_diff:.2f}, Trailer Angle: {trailer_angle:.2f}, Collision: {collision}")

        return obs, reward, terminated, truncated # obs, goal, reward, terminated, truncated


    # Need to tune these
    def _at_goal(self, pos_tol=1.5, angle_tol=5, trailer_tol=8, vel_tol = 0.0001):
    # def _at_goal(self, pos_tol=1.0, angle_tol=0.1, trailer_tol=15.0, vel_tol = 0.1):
        # pos euclidian distance (going to ignore z)
        player_pos = self.player.get_transform()
        pos_dist = np.sqrt((self.goal_pos.location.x - player_pos.location.x)**2 + (self.goal_pos.location.y - player_pos.location.y)**2)
        
        # only care about yaw
        player_yaw = player_pos.rotation.yaw % 360
        if player_yaw > 180: player_yaw -= 360
        elif player_yaw < -180: player_yaw += 360

        goal_yaw = self.goal_pos.rotation.yaw % 360
        if goal_yaw > 180: goal_yaw -= 360
        elif goal_yaw < -180: goal_yaw += 360
        angle_dist = player_yaw - goal_yaw
        if angle_dist > 180: angle_dist -= 360
        elif angle_dist < -180: angle_dist += 360

        angle_dist = np.abs(angle_dist)
        angle_dist = min(angle_dist, 360 - angle_dist)  # account for wrap-around

        trailer_pos = self.playerTrailer.get_transform()
        # tractor angle
        cab_yaw = player_pos.rotation.yaw % 360
        if cab_yaw > 180: cab_yaw -= 360
        elif cab_yaw < -180: cab_yaw += 360

        trailer_yaw = trailer_pos.rotation.yaw % 360
        if trailer_yaw > 180: trailer_yaw -= 360
        elif trailer_yaw < -180: trailer_yaw += 360
        trailer_angle = cab_yaw - trailer_yaw
        if trailer_angle > 180: trailer_angle -= 360
        elif trailer_angle < -180: trailer_angle += 360

        trailer_angle_error = np.abs(trailer_angle)

        trailer_angle_error = min(trailer_angle_error, 360 - trailer_angle_error)  # account for wrap-around

        # vel
        vel = self.player.get_velocity()
        vel_magnitude = np.sqrt(vel.x**2 + vel.y**2) # not going to consider z
        #print(f"Pos dist: {pos_dist}, Angle dist: {angle_dist}, Trailer angle error: {trailer_angle_error}, Vel magnitude: {vel_magnitude}")
        if (pos_dist < pos_tol and angle_dist < angle_tol and trailer_angle_error < trailer_tol and vel_magnitude < vel_tol):
            return True
        else:
            return False

    def _on_collision(self, event):
        # filtering out collisions with the ground which seem to be erroneous
        if (event.other_actor.type_id != "static.ground"):           
            self.collision = True
    
    def _spawn_cameras(self):
        # Should initialize cameras and get them to start listening

        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', '140') # talk to Rodrigo about changing the size + FOV

        Attachment = carla.AttachmentType

        player_bound_x = 0.5 + self.player.bounding_box.extent.x
        player_bound_y = 0.5 + self.player.bounding_box.extent.y
        player_bound_z = 0.5 + self.player.bounding_box.extent.z

        playerTrailer_bound_x = 0.5 + self.playerTrailer.bounding_box.extent.x
        playerTrailer_bound_y = 0.5 + self.playerTrailer.bounding_box.extent.y
        playerTrailer_bound_z = 0.5 + self.playerTrailer.bounding_box.extent.z

        # It appears that the cab and trailer have the same location. This may be due to both of them being defined with the zero at the spawn point?
        camera_transforms = [
                (carla.Transform(carla.Location(x=+0.8*player_bound_x, y=+0.0*player_bound_y, z=1.3*player_bound_z)), Attachment.Rigid), # on cab
                (carla.Transform(carla.Location(x=0, y=0.8*playerTrailer_bound_y, z=1.3*player_bound_z), carla.Rotation(yaw=140)), Attachment.Rigid), # on trailer
                (carla.Transform(carla.Location(x=0, y=-0.8*playerTrailer_bound_y, z=1.3*playerTrailer_bound_z), carla.Rotation(yaw=-140)), Attachment.Rigid), # on trailer
                (carla.Transform(carla.Location(x=-2.1* playerTrailer_bound_x, y=0, z=1.3*playerTrailer_bound_y), carla.Rotation(yaw=180)), Attachment.Rigid), # on trailer
        ]
        
        self.image_list = [None] * len(camera_transforms)

        camera_list = []
        for i in range(len(camera_transforms)):
            transform = camera_transforms[i]

            sensor = self.world.spawn_actor(self.rgb_cam, transform[0], attach_to=self.player if i == 0 else self.playerTrailer, attachment_type=transform[1])

            sensor.listen(lambda data, idx=i: self._process_img(data, idx))
            camera_list.append(sensor)
            self.actor_list.append(sensor)

        
        return camera_list
    
    def _process_img(self, data, id):
        image = np.array(data.raw_data)

        i2 = image.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
                
        self.image_list[id] = i3

    def _get_actor_blueprints(self, world, filter, generation):
        bps = world.get_blueprint_library().filter(filter)

        if generation.lower() == "all":
            return bps

        # If the filter returns only one bp, we assume that this one needed
        # and therefore, we ignore the generation
        if len(bps) == 1:
            return bps

        try:
            int_generation = int(generation)
            # Check if generation is in available generations
            if int_generation in [1, 2]:
                bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
                return bps
            else:
                print("   Warning! Actor Generation is not valid. No actor will be spawned.")
                return []
        except:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []

    def destroy(self):
        print("Destroying actors...")
        for actor in self.actor_list:
            try:
                # Stop sensors if they are listening
                if hasattr(actor, 'is_listening') and actor.is_listening:
                    actor.stop()
                actor.destroy()
            except Exception as e:
                print(f"Error destroying actor: {e}")
        self.actor_list = []

        # Tick the world a few times to make sure actors are removed
        for _ in range(5):
            self.world.tick()
            time.sleep(0.05)

if __name__ == "__main__":
    env = TruckEnv()
    env.reset()
    action = [0] * env.action_ndim
    action[0] = 1
    action[4] = 1
    #action[3] = 1
    try:
        step = 0
        while True:
            _, reward, terminated, truncated = env.step(action)
            #print(reward)
            action[4] = 0
            #action[2]
            # if step > 100:
            #     action[0] = 0
            #     action[1] = 1
            # step += 1
            if terminated or truncated:
                env.reset()
    except Exception as e:
        print(f"Caught Exception: {e}")
    finally:
        print("Shutdown Triggered...")
        env.destroy()