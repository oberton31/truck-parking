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

SHOW_PREVIEW = -1
IM_WIDTH = 640
IM_HEIGHT = 480
RENDER_CARLA = True # SET TO FALSE FOR TRAINING

class TruckEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT

    def __init__(self, max_steps=1000):
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
        self.collision_hist = []
        self.image_list = []

        self.control = carla.VehicleControl()
        self.steer_cache = 0.0 # stores the prior steering values
        self.max_steps = max_steps
        self.curr_steps = 0

        # TODO: would be nice to have a couple different goal positions
        self.goal_pos = carla.Transform(carla.Location(-43.39, 191.2, 0.06), carla.Rotation(0, -90, 0))

        self.action_ndim = 5 # If change action space, need to change this

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 Hz control steps
        settings.no_rendering_mode = not RENDER_CARLA
        self.world.apply_settings(settings)

    def reset(self):
        # TODO: define spawn point and goal point
        print("Resetting Env")
        self.curr_steps = 0
        self.destroy()


        self.actor_list = []
        spawn_point = carla.Transform(carla.Location(-41, 171, 1), carla.Rotation(0, -90, 0))

        num_attempts = 0
        self.playerTrailer = None
        while self.playerTrailer is None and num_attempts < 5:
            self.playerTrailer = self.world.try_spawn_actor(self.blueprintTrailer, spawn_point)
            num_attempts += 1
            self.world.tick()

        if (self.playerTrailer) is None: raise Exception("Could Not Spawn Trailer. Validate Spawn Point...")
        self.actor_list.append(self.playerTrailer)
        forwardVector = spawn_point.get_forward_vector() * 5.2
        spawn_point.location += forwardVector

        num_attempts = 0
        self.player = None
        while self.player is None and num_attempts < 5:
            self.player = self.world.try_spawn_actor(self.blueprint, spawn_point) 
            num_attempts += 1
            self.world.tick()

        if (self.player) is None: raise Exception("Could Not Spawn Cab. Validate Spawn Point...")
        self.actor_list.append(self.player)

        # spawn cameras
        self.rgb_sensors = self._spawn_cameras()

        pos = self.player.get_transform() # it appears the position for both of these is the same, but the rotation varies
        pos_trailer = self.playerTrailer.get_transform()

        trailer_angle = pos.rotation.yaw - pos_trailer.rotation.yaw # right hand rule from cab to trailer
        vel = self.player.get_velocity()

        self.collision_hist = []
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


        return (self.image_list, pos_list, vel_list, trailer_angle, reverse) # TODO: talk about this some more

    def step(self, action):
        # TODO: Discuss action space
        # For now, discrete action space that mirror the manual control
        # 0: throttle, 1: brake, 2: turn steering wheel left, 3: turn steering wheel right, 4: toggle forward/reverse
        self.curr_steps += 1
        if action[0]:
            self.control.throttle = min(self.control.throttle + 0.01, 1.00)
        else:
            self.control.throttle = 0.0

        if action[1]:
            self.control.brake = min(self._control.brake + 0.2, 1)
        else:
            self.control.brake = 0.0
        
        # steer cache behaviour taken from manual control. Idea is that changes in steering applied
        # gradually (do not jump instantly to full turn)
        # Think of it this way: steer_cache holds steering wheel input, and steer holds actual tire steering

        steer_increment = 5e-4 * self.world.get_settings().fixed_delta_seconds * 0.001 # put into milliseconds
        if action[2]:
            if self.steer_cache > 0:
                self.steer_cache = 0
            else:
                self.steer_cache -= steer_increment
        elif action[3]:
            if self.steer_cache < 0:
                self.steer_cache = 0
            else:
                self.steer_cache += steer_increment
        else:
            self.steer_cache = 0.0
        self.steer_cache = min(0.7, max(-0.7, self.steer_cache))            
        
        if action[4]:
            self.control.gear = 1 if self.control.reverse else -1
            print(self.control.reverse)

        self.player.apply_control(self.control)

        # advance env
        self.world.tick()

        reward = 0
        terminated = False
        truncated = False

        if len(self.collision_hist) != 0:
            terminated = True
            reward = -10
        
        if self.curr_steps >= self.max_steps:
            truncated = True
        
        if self._at_goal():
            terminated = True
            reward = 10
        
        pos = self.player.get_transform()
        pos_trailer = self.playerTrailer.get_transform()

        trailer_angle = pos.rotation.yaw - pos_trailer.rotation.yaw
        vel = self.player.get_velocity()

        pos_list = [pos.location.x, pos.location.y, pos.location.z, pos.rotation.pitch, pos.rotation.yaw, pos.rotation.roll]
        vel_list = [vel.x, vel.y, vel.z]

        obs = (self.image_list, pos_list, vel_list, trailer_angle, self.control.reverse)

        return obs, reward, terminated, truncated


    def _at_goal(self, pos_tol=0.01, angle_tol=0.1, trailer_tol=0.1, vel_tol = 0.01):
        # pos euclidian distance (going to ignore z)
        player_pos = self.player.get_transform()
        pos_dist = np.sqrt((self.goal_pos.location.x - player_pos.location.x)**2 + (self.goal_pos.location.y - player_pos.location.y))**2
        
        # only care about yaw
        angle_dist = np.abs(self.goal_pos.rotation.yaw - player_pos.rotation.yaw)

        trailer_pos = self.playerTrailer.get_transform()
        # tractor angle
        trailer_angle_error = np.abs(player_pos.rotation.yaw - trailer_pos.rotation.yaw)

        # vel
        vel = self.player.get_velocity()
        vel_magnitude = np.sqrt(vel.x**2 + vel.y**2) # not going to consider z

        if (pos_dist < pos_tol and angle_dist < angle_tol and trailer_angle_error < trailer_tol and vel_magnitude < vel_tol):
            return True
        else:
            return False

    def _on_collision(self, event):
        # filtering out collisions with the ground which seem to be erroneous
        if (event.other_actor.type_id != "static.ground"):
            self.collision_hist.append(event)
    
    def _spawn_cameras(self):
        # Should initialize cameras and get them to start listening

        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', '110')

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
                (carla.Transform(carla.Location(x=0, y=0.8*playerTrailer_bound_y, z=1.3*player_bound_z), carla.Rotation(yaw=90)), Attachment.Rigid), # on trailer
                (carla.Transform(carla.Location(x=0, y=-0.8*playerTrailer_bound_y, z=1.3*playerTrailer_bound_z), carla.Rotation(yaw=-90)), Attachment.Rigid), # on trailer
                (carla.Transform(carla.Location(x=-2.1*player_bound_x, y=0, z=1.3*playerTrailer_bound_y), carla.Rotation(yaw=180)), Attachment.Rigid), # on trailer
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
        
        if id == self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        
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
    try:
        while True:
            _, reward, terminated, truncated = env.step(action)
            #print(reward)

            if terminated or truncated:
                env.reset()
    except Exception as e:
        print(f"Caught Exception: {e}")
    finally:
        print("Shutdown Triggered...")
        env.destroy()