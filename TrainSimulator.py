import gymnasium as gym
import pybullet as p

from BaseSimulator import BaseSimulator
from Car import RLCar
from GenTrack import Track, TrackGenerator
from RLModels import RLModelHandler


class TrainSimulator(BaseSimulator):
    """Simulator for training RL model"""

    def __init__(
        self,
        mode=p.GUI,
        fps=120,
        frame_gap_action=5,
        track_generator=TrackGenerator(origin=Track.Origin.RANDOM),
    ):
        """
        :param mode: Pybullet mode (GUI, DIRECT, ...)
        :param fps: Number of frames per second use in the simulation
        :param frame_gap_action:  Number of frames between two actions
        """
        gym.Env.__init__(self)
        BaseSimulator.__init__(
            self, [RLCar], mode, fps, frame_gap_action, track_generator, wait=False
        )

        # Training variables
        self.step_count = 0
        self.max_steps = 2000

    @property
    def rl_car(self):
        """
        :return: RLCar instance
        """
        return self.cars[0]

    def reset(self, seed=None, options=None):
        """Reset the simulation. Method called by gym.Env.reset()."""
        gym.Env.reset(self, seed=seed)

        # Generate new track
        self.track = self.track_generator.generate()

        # Reset simulation
        self.init()

        # Reset step count
        self.step_count = 0

        # Return initial observation and empty dictionary of metadata (required by gym.Env)
        return self.rl_car.get_obs(), {}

    def step(self, action):
        """
        Perform a simulation step. Method called by gym.Env.step().

        :param action: Action given by the RL model.
        """

        # Perform an action with the given action
        self.perform_action(action=action)
        self.step_count += 1

        # Compute observations, reward and done flag
        obs = self.rl_car.get_obs()
        reward = self.rl_car.compute_reward()
        done = not self.rl_car.on_track
        truncated = self.step_count >= self.max_steps

        if done:
            print("/!\ Car go off track!")

        # return observation, reward, done flag, truncated flag and empty dictionary of metadata (required by gym.Env)
        return obs, reward, done, truncated, {}


if __name__ == "__main__":
    handler = RLModelHandler(lambda: TrainSimulator(mode=p.DIRECT))
    handler.train("ppo_car12", n_envs=8)
