import time
from typing import Callable

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

from rl.Car import Car
from rl.GenTrack import TrackGenerator
from rl.GUI import GUI


class BaseSimulator(gym.Env):
    """Base class for all simulators."""

    def __init__(
        self,
        cars_type,
        mode=p.GUI,
        fps=120,
        frame_gap_action=5,
        track_generator=TrackGenerator(),
        wait=True,
    ):
        """
        :param mode: Pybullet mode (GUI, DIRECT, ...)
        :param fps: Number of frames per second use in the simulation
        :param frame_gap_action:  Number of frames between two actions
        :param track_generator: Track generator to use.
        :param cars_type: List of car types to use
        :param wait: If True, the simulator waits for the next frame. If False, the simulator runs at the maximum possible frame rate.
        """
        # Ensure this class is never instantiated directly
        if type(self) == BaseSimulator:
            raise RuntimeError(
                "BaseSimulator is an abstract class and cannot be instantiated."
            )

        # Initialize Pybullet GUI
        GUI.init(mode, fps)

        # Initialize simulator parameters
        self.mode = mode
        self.fps = fps
        self.dt = 1 / fps
        self.frame_gap_action = frame_gap_action
        self.track = track_generator.generate()
        self.track_generator = track_generator
        self.cars = []
        self.cars_type = cars_type
        self.wait = wait

        # Create a temporary car to get observation dimension
        car = Car(self.track)
        obs_dim = len(car.get_obs())

        # Create observation and action spaces
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32,
        )

    def init(self):
        """
        Initialize the simulation. This method must be called before performing any simulation step.
        """

        # Reset Pybullet simulation (remove all objects)
        p.resetSimulation()

        # Set camera position to the center of the track in GUI mode
        if self.mode == p.GUI:
            GUI.set_camera_position_to(self.track)

        # Generate cars
        self.cars = []
        for car_type in self.cars_type:
            car = car_type(self.track)
            # Remove collisions between cars
            for c in self.cars:
                p.setCollisionFilterPair(
                    car.obj_id, c.obj_id, -1, -1, enableCollision=0
                )
            self.cars.append(car)

        # Draw track lines
        GUI.draw_track_line(self.track, self.cars)

    def perform_action(self, action=None):
        """
        Performs the desired action for all cars on the track with additional frame steps between each action if needed.

        :param action: Action given by the training algorithm (Optional)
        """

        def do_frame_step(callback: Callable = lambda: None):
            """
            Executes a simulation frame, optionally invoking a callback function to do action (move).
            Updates the state of all cars on the track after executing the simulation step.
            Wait for the next frame if `wait` is True.

            :param callback: Callback function to execute at this frame.
            """
            # Get current time
            t0 = time.time()

            # Do action
            callback()

            # Perform simulation step
            p.stepSimulation()

            # Update state of all cars on the track
            for car in self.cars:
                car.compute_state()

            # If wait is True, wait for the next frame according to the simulation frame rate
            if self.wait:
                time.sleep(max(0.0, self.dt - (time.time() - t0)))

        # Create a lambda function to move all cars
        move_all_cars = lambda: [
            car.move(car.get_move(action=action)) for car in self.cars
        ]

        # Perform the action and simulation step to move all cars
        do_frame_step(move_all_cars)
        for _ in range(self.frame_gap_action - 1):
            # Do the simulation step without moving cars
            do_frame_step()
