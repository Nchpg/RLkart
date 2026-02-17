import math
import time
from enum import Enum

import numpy as np
import pybullet as p

from gen_track import Track
from car import Car


class SimulationMode(Enum):
    MANUAL = 0


class SimulatorGUI:
    def __init__(self, track: Track, nb_cars: int = 1):
        self.track = track
        self.nb_cars = nb_cars
        self.track_id = None
        self.cars = []
        self.selected_car = 0
        self.do_action_count = 0
        self.mode = SimulationMode.MANUAL

        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

        p.setGravity(0, 0, 0)

        p.setTimeStep(1 / 120)
        p.setRealTimeSimulation(0)

        # Compute camera position
        window_width, window_height = p.getDebugVisualizerCamera()[:2]
        camera_fov = 90

        x_coords = [pt[0] for pt in self.track.centerline]
        y_coords = [pt[1] for pt in self.track.centerline]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        x_mid = x_min + (x_max - x_min) / 2
        y_mid = y_min + (y_max - y_min) / 2
        width = x_max - x_min
        height = y_max - y_min

        if height / window_height > width / window_width:
            max_dim = height
        else:
            max_dim = width
        max_dim += self.track.width + 2

        p.resetDebugVisualizerCamera(
            cameraDistance=(max_dim / 2.0) / math.tan(math.radians(camera_fov / 2.0)),
            cameraYaw=0,
            cameraPitch=-89.999,  # presque -90 = vue du dessus
            cameraTargetPosition=[x_mid, y_mid, 0],
        )

        # Plane
        plane_size = window_width / window_height * max_dim / 2
        vis = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[plane_size, plane_size, 0.01],
            rgbaColor=[0.227, 0.616, 0.137, 1],
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=[x_mid, y_mid, 0],
        )

    def draw_track(self):
        # track
        track_height = 0.05
        vertices = []
        indices = []

        n = self.track.nb_points

        for i in range(n):
            l0 = self.track.left[i]
            l1 = self.track.left[(i + 1) % n]
            r0 = self.track.right[i]
            r1 = self.track.right[(i + 1) % n]

            v_idx = len(vertices)
            vertices.extend(
                [
                    [l0[0], l0[1], track_height],
                    [l1[0], l1[1], track_height],
                    [r1[0], r1[1], track_height],
                    [r0[0], r0[1], track_height],
                ]
            )

            indices.extend([v_idx, v_idx + 2, v_idx + 1, v_idx, v_idx + 3, v_idx + 2])

        visual = p.createVisualShape(
            p.GEOM_MESH,
            vertices=vertices,
            indices=indices,
            rgbaColor=[0.306, 0.239, 0.157, 1],
        )

        self.track_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual)

        wall_thickness = 0.2
        wall_height = 0.1
        for border in [self.track.left, self.track.right]:
            for i in range(len(border)):
                p1 = border[i]
                p2 = border[(i + 1) % len(border)]
                mid = (p1 + p2) / 2
                length = float(np.linalg.norm(p2 - p1))
                yaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

                half_length = length / 2
                half_width = wall_thickness / 2
                half_height = wall_height / 2

                col = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=[half_length, half_width, 1]
                )
                vis = p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[half_length, half_width, half_height],
                    rgbaColor=[0, 0, 0, 1],
                )
                wall = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=vis,
                    baseCollisionShapeIndex=col,
                    basePosition=[mid[0], mid[1], half_height],
                    baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
                )

                for car in self.cars:
                    p.setCollisionFilterPair(
                        car.obj_id, wall, -1, -1, enableCollision=0
                    )

    def draw_cars(self):
        x, y = self.track.centerline[0]
        tangent = self.track.centerline[1] - self.track.centerline[0]
        theta = math.atan2(tangent[1], tangent[0])

        for i in range(self.nb_cars):
            self.cars.append(Car(x, y, theta))

    def run(self):
        dt = 1 / 120

        while True:
            t0 = time.time()
            p.stepSimulation()
            self._do_action()
            time.sleep(max(0.0, dt - (time.time() - t0)))

    def _do_action(self):
        self.do_action_count += 1
        self.do_action_count %= 5

        if self.do_action_count != 0:
            return

        for car in self.cars:
            car.compute_state(self.track)

        print(self.cars[0].ray_fractions)

        actions = {SimulationMode.MANUAL: self._handle_keyboard_input}
        actions.get(self.mode, lambda: print(f"Unknow mode: {self.mode.name}"))()

    def _handle_keyboard_input(self):
        keys = p.getKeyboardEvents()

        # avancer/reculer
        if p.B3G_UP_ARROW in keys:
            self.cars[self.selected_car].forward()
        if p.B3G_DOWN_ARROW in keys:
            self.cars[self.selected_car].backward()
        if p.B3G_LEFT_ARROW in keys:
            self.cars[self.selected_car].left()
        if p.B3G_RIGHT_ARROW in keys:
            self.cars[self.selected_car].right()


if __name__ == "__main__":
    s = SimulatorGUI(Track.generate(samples_per_segment=30, tension=0.8), nb_cars=2)
    s.draw_cars()
    s.draw_track()
    s.run()
