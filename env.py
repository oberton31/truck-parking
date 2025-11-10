#!/usr/bin/env python


from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


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


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
import random
import cv2
import numpy as np
import time

SHOW_PREVIEW = 0
IM_WIDTH = 640
IM_HEIGHT = 480

class TruckEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT

    def __init__(self):
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
        self._collision_hist = []
        self.image_list = []

    def reset(self):
        # TODO: define spawn point and goal point
        spawn_point = carla.Transform(carla.Location(-41, 171, 1), carla.Rotation(0, 90, 0))
        self.playerTrailer = self.world.try_spawn_actor(self.blueprintTrailer, spawn_point)

        if (self.playerTrailer) is None: raise Exception("Could Not Spawn Trailer. Validate Spawn Point...")
        
        forwardVector = spawn_point.get_forward_vector() * 5.2
        spawn_point.location += forwardVector

        self.player = self.world.try_spawn_actor(self.blueprint, spawn_point) 
        self.actor_list.append(self.player)

        # spawn cameras
        self.rgb_sensors = self._spawn_cameras()

        # TODO: some way to get positon data from vehicle
        pos = self.player.get_transform() # it appears the position for both of these is the same, but the rotation varies
        pos_trailer = self.playerTrailer.get_transform()

        angle_trailer = pos.rotation.yaw - pos_trailer.rotation.yaw # right hand rule from cab to trailer
        vel = self.player.get_velocity()
        # print(angle_trailer)
        # spawn collision sensor
        self.collision_hist = []
        collision_blueprint = self.world.get_blueprint_library().find('sensor.other.collision')
        self.player_collision_sensor = self.world.spawn_actor(collision_blueprint, carla.Transform(), attach_to=self.player)


        self.player_collision_sensor.listen(lambda event: self._on_collision(event))
        self.actor_list.append(self.player_collision_sensor)

        self.playerTrailer_collision_sensor = self.world.spawn_actor(collision_blueprint, carla.Transform(), attach_to=self.playerTrailer)

        self.playerTrailer_collision_sensor.listen(lambda event: self._on_collision(event))
        self.actor_list.append(self.playerTrailer_collision_sensor)

        self.player.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        while True:
            images_present = True
            for img in self.image_list:
                if img is None:
                    images_present = False
                    break
            if images_present: break
            time.sleep(0.1)

        return (self.image_list, pos, vel, angle_trailer)

    def _on_collision(self, event):
        self._collision_hist.append(event)
    
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

        playerTrailer_bound_x = 0.5 + self.player.bounding_box.extent.x
        playerTrailer_bound_y = 0.5 + self.player.bounding_box.extent.y
        playerTrailer_bound_z = 0.5 + self.player.bounding_box.extent.z

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
        for actor in self.actor_list:
            actor.stop()
            actor.destroy()

if __name__ == "__main__":
    env = TruckEnv()
    env.reset()
    try:
        while True:
            pass
    except Exception as e:
        print(f"Caught Exception: {e}")
    finally:
        print("Shutdown Triggered...")
        env.destroy()