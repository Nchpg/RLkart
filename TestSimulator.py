import pybullet as p

from BaseSimulator import BaseSimulator
from Car import RLCar
from GenTrack import Track, TrackGenerator
from RLModels import RLModelHandler


class TestSimulator(BaseSimulator):
    """Test simulator"""

    def __init__(
        self,
        cars_type,
        mode=p.GUI,
        fps=120,
        frame_gap_action=5,
        track_generator=TrackGenerator(origin=Track.Origin.RANDOM),
    ):
        """
        :param mode: Pybullet mode (GUI, DIRECT, ...)
        :param fps: Number of frames per second use in the simulation
        :param frame_gap_action:  Number of frames between two actions
        :param track_generator: Track generator to use.
        :param cars_type: List of car types to use
        """
        BaseSimulator.__init__(
            self, cars_type, mode, fps, frame_gap_action, track_generator, wait=True
        )
        self.init()

    def run(self):
        """
        Run the simulation.
        """
        # Perform actions continuously
        while True:
            self.perform_action()


class TestBenchmarkSimulator(BaseSimulator):
    """Test simulator with benchmarking for RL model training"""

    def __init__(
        self,
        mode=p.GUI,
        fps=120,
        frame_gap_action=5,
        track_generator=TrackGenerator(origin=Track.Origin.RANDOM),
        nb_episodes=100,
    ):
        """
        :param mode: Pybullet mode (GUI, DIRECT, ...)
        :param fps: Number of frames per second use in the simulation
        :param frame_gap_action:  Number of frames between two actions
        :param track_generator: Track generator to use.
        :param nb_episodes: Number of episodes to run for benchmarking
        """
        BaseSimulator.__init__(
            self, [RLCar], mode, fps, frame_gap_action, track_generator, wait=True
        )

        # benchmarking variables
        self.epi_count = 0
        self.epi_max = nb_episodes
        self.step_count = 0
        self.max_steps = 2000

    @property
    def rl_car(self):
        """
        :return: RLCar instance
        """
        return self.cars[0]

    def run(self):
        """Run the simulation and benchmark the RL model"""

        def reset():
            """Reset the simulation"""

            # Generate new track
            self.track = self.track_generator.generate()

            # Reset simulation
            self.init()

            # Reset step count
            self.step_count = 0

        def step():
            """Perform a simulation step"""
            self.perform_action()
            self.step_count += 1

        score = 0

        # Run episodes
        for _ in range(self.epi_max):
            # Reset simulation
            reset()

            # Run the episode until the car is off track or max steps reached
            while self.step_count < self.max_steps and self.rl_car.on_track:
                step()
            self.epi_count += 1

            print(
                f"episode {self.epi_count}: {self.step_count} steps | on_track: {self.rl_car.on_track} | progress: {self.rl_car.progress}"
            )
            score += self.rl_car.on_track

            # Save track if the car is off track
            if not self.rl_car.on_track:
                self.track.save_track(f"track/epi_{self.epi_count}.npz")

        print("=" * 50)
        print(f"Average score: {score/self.epi_count}")
        print("=" * 50)


if __name__ == "__main__":
    handler = RLModelHandler(
        lambda: TestSimulator(
            mode=p.GUI, fps=120, frame_gap_action=5, track_generator=TrackGenerator(origin=Track.Origin.ZERO, file_path="track/api_generated_track.npz"), cars_type=[RLCar]
        )
    )
    handler.test(rl_model_filename="Models/ppo_car")
