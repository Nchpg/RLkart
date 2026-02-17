import math

import numpy as np
import pybullet as p

from gen_track import Track


class Car:
    acceleration = 10
    torque = 15
    ray_angle = [0, 45, 90, 135, 180, 225, 270, 315]
    ray_length = 5

    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.yaw = theta
        self.orn = p.getQuaternionFromEuler([0, 0, theta])

        self.speed = 0
        self.vx = 0
        self.vy = 0

        self.angular_speed = 0
        self.progress = 0
        self.on_track = True

        self.ray_fractions = []

        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.3, 0.01])
        vis = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.5, 0.3, 0.01],
            rgbaColor=[0.5, 0.5, 0.5, 1],
        )
        self.obj_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, 0.25],
            baseOrientation=p.getQuaternionFromEuler([0, 0, theta]),
        )
        p.changeDynamics(self.obj_id, -1, linearDamping=0.5, angularDamping=2)

    def compute_state(self, track: Track):
        # speed
        (vx, vy, _), ang_vel = p.getBaseVelocity(self.obj_id)
        self.vx = vx
        self.vy = vy
        self.speed = np.sqrt(self.vx**2 + self.vy**2)
        self.angular_speed = ang_vel[2]

        # pos & orientation
        pos, orn = p.getBasePositionAndOrientation(self.obj_id)
        x, y, _ = pos
        self.x = x
        self.y = y
        self.orn = orn
        self.yaw = math.degrees(p.getEulerFromQuaternion(orn)[2]) % 360

        # track
        self._compute_from_track(track)

        # ray
        self._compute_ray()

        if self.on_track:
            p.changeVisualShape(self.obj_id, -1, rgbaColor=[0.5, 0.5, 0.5, 1])
        else:
            p.changeVisualShape(self.obj_id, -1, rgbaColor=[1, 0, 0, 1])

    def _compute_from_track(self, track: Track):
        distances = np.linalg.norm(
            track.centerline - np.array([self.x, self.y]), axis=1
        )
        idx = np.argmin(distances)
        self.progress = idx / track.nb_points

        dist = distances[idx]
        self.on_track = dist <= track.width / 2

    def _compute_ray(self):
        origin = (self.x, self.y, 0.5)

        angles = np.radians(Car.ray_angle).reshape(-1, 1)
        direction_vect = np.cos(angles) * np.array([1, 0, 0]) + np.sin(
            angles
        ) * np.array([0, 1, 0])

        rot_matrix = np.array(p.getMatrixFromQuaternion(self.orn)).reshape(3, 3)
        transformed_direction = direction_vect @ rot_matrix.T
        ray_to = origin + transformed_direction * Car.ray_length

        hit = p.rayTestBatch([origin] * len(ray_to), ray_to)

        self.ray_fractions = [h[2] for h in hit]

    def _compute_acceleration(self):
        return Car.acceleration * max(3, 5 * self.speed)

    def forward(self):
        p.applyExternalForce(
            self.obj_id,
            -1,
            [self._compute_acceleration(), 0, 0],
            [0, 0, 0],
            p.LINK_FRAME,
        )

    def backward(self):
        p.applyExternalForce(
            self.obj_id,
            -1,
            [-self._compute_acceleration(), 0, 0],
            [0, 0, 0],
            p.LINK_FRAME,
        )

    def left(self):
        p.applyExternalTorque(self.obj_id, -1, [0, 0, Car.torque], p.LINK_FRAME)

    def right(self):
        p.applyExternalTorque(self.obj_id, -1, [0, 0, -Car.torque], p.LINK_FRAME)

    def __str__(self):
        return f"Car(x={self.x:.2f}, y={self.y:.2f}, v={self.speed:.2f}, a_v={self.angular_speed:.2f}, yaw={self.yaw:.2f}, progress={self.progress:.2f}, on_track={self.on_track}, hits={self.ray_fractions})"
