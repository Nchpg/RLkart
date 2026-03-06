import math

import numpy as np
import pybullet as p

from rl.GenTrack import Track
from rl.GUI import GUI


class Car:
    """Car class representing a car in the simulation."""

    # Car parameters
    acceleration = 10
    torque = 15
    ray_angle = [0, 22.5, 45, 90, 180, 270, 315, 337.5]
    ray_length = 15

    # Car dimensions
    length = 0.5
    width = 0.3

    def __init__(self, track: Track):
        """
        :param track: Track on which the car will drive
        """
        self.track = track

        # Get car position and orientation on the track at the origin point (always 0)
        x, y, yaw = track.get_pos_and_direction_at(0)
        self.x = x  # x position of the car center
        self.y = y  # y position of the car center
        self.yaw = yaw  # z axis orientation of the car
        self.orn = p.getQuaternionFromEuler(
            [0, 0, yaw]
        )  # orientation of the car as a quaternion

        # Velocity parameters
        self.velocity = 0  # velocity
        self.vx = 0  # velocity in x direction
        self.vy = 0  # velocity in y direction
        self.wz = 0  # angular velocity around z axis
        self.vl = 0  # velocity in the car lateral direction
        self.vf = 0  # velocity in the car forward direction

        # Progress parameters
        self.progress = 0
        self.prev_progress = self.progress
        self.last_distance = 0
        self.last_centerline_idx = 0
        self.dist_to_centerline = 0
        self.nb_lap = 0

        # Boolean to know if the car is on the track or not
        self.on_track = True

        # Parameter that will store the fraction distances of all the car rays
        self.ray_fractions = []

        # Draw the car in GUI
        self.obj_id = GUI.draw_car(self)

        # Compute initial state
        self.compute_state()

    def compute_state(self):
        """Compute the car state at the current time step"""
        # Velocity
        (vx, vy, _), (_, _, wz) = p.getBaseVelocity(self.obj_id)
        self.velocity = np.sqrt(vx**2 + vy**2)
        self.vx = vx
        self.vy = vy
        self.wz = wz

        # Position & Orientation
        (x, y, _), orn = p.getBasePositionAndOrientation(self.obj_id)
        _, _, yaw = p.getEulerFromQuaternion(orn)
        self.x = x
        self.y = y
        self.orn = orn
        self.yaw = yaw

        # Project velocity on car forward and lateral directions
        self.vf = vx * math.cos(self.yaw) + vy * math.sin(self.yaw)
        self.vl = -vx * math.sin(self.yaw) + vy * math.cos(self.yaw)

        # Compute track progression
        idx, dist = self.track.get_closest_centerline_point_idx_distance_on_track(
            self.x, self.y
        )  # Get the closest centerline point on track
        self.dist_to_centerline = dist
        self.on_track = (
            dist <= (self.track.width - min(Car.length, Car.width)) / 2
        )  # Check if the car is on the track
        self.last_centerline_idx = idx
        if self.on_track:
            distance = self.track.get_distance_on_track_from_origin(idx)
        else:
            distance = self.last_distance

        # Check lap count
        distance_bounds = self.track.total_distance * 0.05
        if (
            self.last_distance > self.track.total_distance - distance_bounds
            and distance < distance_bounds
        ):
            self.nb_lap += 1
        if (
            distance > self.track.total_distance - distance_bounds
            and self.last_distance < distance_bounds
        ):
            self.nb_lap -= 1
        self.progress = (
            self.nb_lap + distance / self.track.total_distance
        )  # Update progression with lap count and distance on track
        self.last_distance = distance

        # Ray
        ray_origin = (self.x, self.y, 0.5)
        angles = np.radians(Car.ray_angle).reshape(
            -1, 1
        )  # Convert angles to radians in a column vector
        direction_vect = np.cos(angles) * np.array([1, 0, 0]) + np.sin(
            angles
        ) * np.array(
            [0, 1, 0]
        )  # Compute direction vector from angles
        rot_matrix = np.array(p.getMatrixFromQuaternion(self.orn)).reshape(
            3, 3
        )  # Get rotation matrix from orientation quaternion
        transformed_direction = (
            direction_vect @ rot_matrix.T
        )  # Rotate direction vector by rotation matrix
        ray_to = (
            ray_origin + transformed_direction * Car.ray_length
        )  # Get ray endpoints
        hit = p.rayTestBatch([ray_origin] * len(ray_to), ray_to)
        self.ray_fractions = [h[2] for h in hit]

        # Update GUI car color
        if self.on_track:
            p.changeVisualShape(self.obj_id, -1, rgbaColor=[0.5, 0.5, 0.5, 1])
        else:
            p.changeVisualShape(self.obj_id, -1, rgbaColor=[1, 0, 0, 1])

    def _compute_acceleration(self, f=1):
        """
        Compute the acceleration to apply based on the current velocity.

        :param f: factor to apply to acceleration
        """
        return Car.acceleration * max(3.0, 5.0 * self.velocity / f)

    def forward(self, amount=1.0):
        """
        Apply a forward force to the car

        :param amount: throttle amount to apply (between 0 and 1)
        """
        p.applyExternalForce(
            self.obj_id,
            -1,
            [self._compute_acceleration() * amount, 0, 0],
            [0, 0, 0],
            p.LINK_FRAME,
        )

    def backward(self, amount=1.0):
        """
        Apply a backward force to the car

        :param amount: throttle amount to apply (between 0 and 1)
        """
        p.applyExternalForce(
            self.obj_id,
            -1,
            [-self._compute_acceleration(f=2) * amount, 0, 0],
            [0, 0, 0],
            p.LINK_FRAME,
        )

    def left(self, amount=1.0):
        """
        Apply a left torque to the car

        :param amount: steering amount to apply (between 0 and 1)
        """
        p.applyExternalTorque(
            self.obj_id, -1, [0, 0, Car.torque * amount], p.LINK_FRAME
        )

    def right(self, amount=1.0):
        """
        Apply a right torque to the car

        :param amount: steering amount to apply (between 0 and 1)
        """
        p.applyExternalTorque(
            self.obj_id, -1, [0, 0, -Car.torque * amount], p.LINK_FRAME
        )

    def get_obs(self):
        """
        Compute the car observation vector. The observation includes:
        - Velocity in forward and lateral directions
        - Angular velocity
        - Relative yaw to track direction (cos and sin)
        - Distance to centerline
        - Look-ahead track angles (cos and sin of relative angles at 10, 20 and 40 points ahead)
        - Ray fractions
        """
        # Current track direction
        track_yaw = self.track.get_direction_at(round(self.last_centerline_idx))

        # Relative yaw
        rel_yaw = self.yaw - track_yaw
        while rel_yaw > math.pi:
            rel_yaw -= 2 * math.pi
        while rel_yaw < -math.pi:
            rel_yaw += 2 * math.pi

        # Look-ahead track angles (relative to current track angle)
        look_aheads = []
        for offset in [10, 20, 40]:
            target_idx = self.last_centerline_idx + offset
            future_yaw = self.track.get_direction_at(target_idx)
            rel_future_yaw = future_yaw - track_yaw
            while rel_future_yaw > math.pi:
                rel_future_yaw -= 2 * math.pi
            while rel_future_yaw < -math.pi:
                rel_future_yaw += 2 * math.pi
            look_aheads.extend([math.cos(rel_future_yaw), math.sin(rel_future_yaw)])

        return np.hstack(
            [
                np.array(
                    [
                        self.vf,
                        self.vl,
                        self.wz,
                        math.cos(rel_yaw),
                        math.sin(rel_yaw),
                        self.dist_to_centerline,
                    ]
                ),
                #np.array(look_aheads),
                np.array(self.ray_fractions),
            ]
        ).astype(np.float64)

    def compute_reward(self):
        """
        Compute the reward for the current state of the car. The reward is composed of:
        - Progress reward: based on the progress made since the last step (scaled to be large
        - Survival / Time penalty: small penalty each step to encourage speed
        - Off-track penalty: large penalty if the car goes off track
        - Centering bonus: reward for staying near the centerline (only if moving forward)
        - Stability penalty: penalty for excessive shaking/wobbling
        - Velocity bonus: reward for speed in the right direction
        """
        # Calculate progress made since last step
        delta_progress = self.progress - self.prev_progress
        self.prev_progress = self.progress

        # 1. Main Reward: Progress (scaled to be large)
        reward = delta_progress * 2000

        # 2. Survival / Time Penalty: Small penalty each step to encourage speed
        reward -= 0.1

        # 3. Off-track Penalty
        if not self.on_track:
            reward -= 1000
            return reward

        # 4. Centering Bonus: Reward staying near the centerline (only if moving forward)
        if self.vf > 0.5:
            # 1.0 when on center, 0.0 when at edge
            centering = 1.0 - (self.dist_to_centerline / (self.track.width / 2.0))
            reward += max(0, centering) * 0.2

        # 5. Stability Penalty: Penalize excessive shaking/wobbling
        reward -= abs(self.wz) * 0.02

        # 6. Velocity Bonus: Directly reward speed in the right direction
        reward += self.vf * 0.05

        return reward

    def get_move(self):
        """Get the move to perform for this car. This method should be overridden by subclasses (e.g. ManualCar, RLCar) to return the desired action based on user input or RL model prediction."""
        return {"up": 0, "down": 0, "left": 0, "right": 0}

    def move(self, direction: dict):
        """
        Perform the desired move for this car based on the given direction dictionary.

        :param direction: Dictionary with keys "up", "down", "left" and "right" and values between 0 and 1 representing the throttle/steering amount for each direction.
        """
        if up := direction.get("up", 0):
            self.forward(up)
        if down := direction.get("down", 0):
            self.backward(down)
        if left := direction.get("left", 0):
            self.left(left)
        if right := direction.get("right", 0):
            self.right(right)

    def __str__(self):
        """String representation of the car state for debugging purposes."""
        return f"Car(x={self.x:.2f}, y={self.y:.2f}, v={self.velocity:.2f}, vf={self.vf}, vl={self.vl}, a_v={self.wz:.2f}, yaw={self.yaw:.2f}, progress={self.progress:.2f}, on_track={self.on_track}, hits={self.ray_fractions})"


class ManualCar(Car):
    """Car controlled by user keyboard input in GUI mode. Use arrow keys to control the car."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_move(self, **kwargs):
        """Get the move to perform based on user keyboard input. Use arrow keys to control the car."""
        keys = p.getKeyboardEvents()
        return {
            "up": 1.0 if keys.get(p.B3G_UP_ARROW) else 0.0,
            "down": 1.0 if keys.get(p.B3G_DOWN_ARROW) else 0.0,
            "left": 1.0 if keys.get(p.B3G_LEFT_ARROW) else 0.0,
            "right": 1.0 if keys.get(p.B3G_RIGHT_ARROW) else 0.0,
        }


class RLCar(Car):
    """Car controlled by a Reinforcement Learning model. The model predicts the desired action based on the car observation vector."""

    model = None
    vec_env = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_move(self, action=None):
        """Get the move to perform based on the RL model prediction. If action is None, use the model to predict the action based on the current observation."""
        if action is None:
            assert (
                RLCar.model is not None and RLCar.vec_env is not None
            ), "Model and VecNormalize environment must be set for RLCar"

            obs = self.get_obs()
            obs = np.array([obs])
            obs = RLCar.vec_env.normalize_obs(obs)

            action, _ = RLCar.model.predict(obs, deterministic=True)
            action = action[0]

        throttle, steering = action
        return {
            "up": max(0.0, throttle),
            "down": max(0.0, -throttle),
            "left": max(0.0, -steering),
            "right": max(0.0, steering),
        }

    @staticmethod
    def set_model(full_model):
        """Set the RL model and VecNormalize environment for all RLCar instances."""
        model, vec_env = full_model
        RLCar.model = model
        RLCar.vec_env = vec_env
