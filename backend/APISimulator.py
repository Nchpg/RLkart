import pybullet as p
import numpy as np
from rl.BaseSimulator import BaseSimulator
from rl.Car import RLCar, Car
from rl.GenTrack import Track, generate_centerline, get_left_right_track, TrackGenerator
from rl.GUI import GUI

class CustomTrack(Track):
    """Custom Track class that allows manual initialization with given points."""
    def __init__(self, centerline, left, right, width):
        # We don't call super() because Track prevents direct instantiation
        self._initialize(centerline, left, right, width, Track.Origin.ZERO)

class CustomTrackGenerator(TrackGenerator):
    """Track generator that returns a pre-calculated track."""
    def __init__(self, track: Track):
        self.track = track

    def generate(self):
        return self.track

class APISimulator(BaseSimulator):
    """Simulator specifically for API usage, running in DIRECT mode and returning trajectory."""

    def __init__(
        self,
        control_points: list,
        width: int = 4,
        samples_per_segment: int = 20,
        tension: float = 0.5,
        fps: int = 120,
        frame_gap_action: int = 5,
    ):
        # Create track from control points
        pts = np.array(control_points)
        centerline = generate_centerline(pts, samples_per_segment, tension)
        left, right = get_left_right_track(centerline, width)
        
        # Instantiate track
        track = object.__new__(CustomTrack)
        track._initialize(centerline, left, right, width, Track.Origin.ZERO)
        
        # We use a custom track generator to bypass the default random generation
        track_generator = CustomTrackGenerator(track)
        
        # We also need to skip GUI.init(mode, fps) if we want to handle multiple calls correctly without re-connecting? 
        # Actually BaseSimulator.init() handles things.
        
        # Initialize BaseSimulator in DIRECT mode
        super().__init__(
            cars_type=[RLCar],
            mode=p.DIRECT,
            fps=fps,
            frame_gap_action=frame_gap_action,
            track_generator=track_generator,
            wait=False # No need to wait in API mode
        )
        self.init()
        
        self.max_steps = 4000 # Enough to complete a lap
        self.trajectory = []


    def run_simulation(self):
        """Run one episode and return the trajectory."""
        car = self.cars[0]
        step_count = 0
        
        while step_count < self.max_steps and car.dist_to_centerline <= self.track.width / 2:
            # Check if car has an action method (RLCar has get_move)
            # BaseSimulator perform_action uses car.get_move(action=action)
            self.perform_action()
            self.trajectory.append({
                "x": float(car.x),
                "y": float(car.y),
                "yaw": float(car.yaw)
            })
            step_count += 1
            
            if car.progress > 1:
                break

        print(f"Simulation completed: steps={step_count}, progress={car.progress:.2f}, on_track={car.on_track}")
                
        return {
            "trajectory": self.trajectory,
            "track": {
                "centerline": self.track.centerline.tolist(),
                "left": self.track.left.tolist(),
                "right": self.track.right.tolist(),
                "width": float(self.track.width)
            },
            "success": bool(car.progress > 0.9),
            "progress": float(car.progress)
        }
