from __future__ import annotations

import numpy as np


def catmull_rom(
    P0: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    P3: np.ndarray,
    nb_points: int = 20,
    tension: float = 0.5,
) -> np.ndarray:
    """
    Compute a Catmull-Rom spline between P1 and P2 using P0 and P3 for tangents.

    :param P0:
    :param P1:
    :param P2:
    :param P3:
        => Control points of the spline. The curve passes through P1 and P2.

    :param nb_points:  Number of points to generate along the curve
    :param tension: Amount of tension to apply to the curve. Values between 0 and 1 increase the amount of tension
    :return: Array of shape (n_points, dim) with points on the spline.
    """
    # Generate nb_points evenly spaced values from 0 to 1
    t = np.linspace(0, 1, nb_points)

    # Construct the cubic polynomial basis matrix for each t.
    T = np.stack([t**3, t**2, t, np.ones_like(t)], axis=1)

    # Define the Catmull-Rom spline coefficient matrix
    M = np.array(
        [
            [-tension, 2 - tension, tension - 2, tension],
            [2 * tension, tension - 3, 3 - 2 * tension, -tension],
            [-tension, 0, tension, 0],
            [0, 1, 0, 0],
        ]
    )

    # Collect P0, P1, P2, P3 into a matrix for vectorized computation
    G = np.stack([P0, P1, P2, P3])

    # Compute the spline points
    return T @ M @ G


def generate_centerline(
    control_points: np.ndarray, samples_per_segment: int = 20, tension: float = 0.5
) -> np.ndarray:
    """
    Generate a closed Catmull-Rom spline passing through all given points.

    :param control_points: List or array of control points
    :param samples_per_segment: Number of points to sample per segment between consecutive control points
    :param tension: Amount of tension to apply to the curve. Values between 0 and 1 increase the amount of tension
    :return: Array of shape (len(control_points) * samples_per_segment, dim) containing the points on the centerline.
    """
    centerline = []
    n = len(control_points)

    # Loop through each point to generate spline segments connecting them.
    for i in range(n):
        P0 = control_points[(i - 1) % n]
        P1 = control_points[i]
        P2 = control_points[(i + 1) % n]
        P3 = control_points[(i + 2) % n]

        segment = catmull_rom(P0, P1, P2, P3, samples_per_segment, tension)
        centerline.append(segment[:-1])

    # Combine all segments into a single array representing the centerline.
    return np.vstack(centerline)


def generate_track_control_points(
    nb_control_points: int = 10, base_radius: int = 20, noise: int = 10
) -> np.ndarray:
    """
    Generate control points arranged roughly in a circular shape with radial noise.

    The control points are evenly distributed in angle around a circle, and
    their radius is randomly perturbed to create an irregular track shape.

    :param nb_control_points: Number of control points to generate
    :param base_radius: Radius of the circle from which the control points are generated
    :param noise: Maximum radial deviation added/subtracted from the base radius
    :return: Array of shape (nb_control_points, 2) containing the control points coordinates
    """
    # Generate evenly spaced angles between 0 and 2π
    angles = np.linspace(0, 2 * np.pi, nb_control_points, endpoint=False)

    # Generate a radius for each control point in the range [base_radius - noise, base_radius + noise]
    r = base_radius + np.random.uniform(-noise, noise, nb_control_points)

    # Convert polar coordinates(radius, angle) to Cartesian coordinates(x, y)
    return np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)


def compute_normals(points: np.ndarray) -> np.ndarray:
    """
    Compute 2D normal vectors for a sequence of points.

    :param points: Array of 2D points
    :return: Array of normal vectors for each point
    """
    # Compute approximate tangent vectors at each point
    tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

    # Compute 2D normals by rotating tangents 90 degrees counterclockwise.
    normals = np.zeros_like(tangents)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]

    return normals


def get_left_right_track(centerline, width=4):
    """
    Generate left and right track boundaries from a centerline.

    :param centerline: Array of 2D points representing the centerline of a track
    :param width: Width of the track
    :return: Tuple of left and right track boundaries, each represented as an array of 2D points.
    """
    # Compute unit normal vectors perpendicular to the centerline at each point.
    normals = compute_normals(centerline)

    # Shift the centerline along the normals by half the width to get the left and right boundary.
    left = centerline + normals * width / 2
    right = centerline - normals * width / 2
    return left, right


class Track:
    def __new__(cls, *args, **kwargs):
        """
        Prevent direct instantiation of the clas
        """
        raise RuntimeError(
            "Do not instantiate directly.\n Use from_file() or generate() instead."
        )

    def _initialize(
        self, centerline: np.ndarray, left: np.ndarray, right: np.ndarray, width: int
    ):
        """
        Initialize the track attributes.

        :param centerline: Array of 2D points representing the centerline of the track.
        :param left: Array of 2D points representing the left boundary of the track.
        :param right: Array of 2D points representing the right boundary of the track.
        :param width: Width of the track.
        """
        self.centerline = centerline
        self.left = left
        self.right = right
        self.nb_points = len(centerline)
        self.width = width

    @classmethod
    def load(cls, file_path: str) -> Track:
        """
        Load a Track instance from a saved .npz file

        :param file_path: Path to the .npz file containing saved track data.
        :return: A new Track instance, initialized with the loaded data.
        """
        # Load track
        data = np.load(file_path)

        # Instantiate track
        self = object.__new__(cls)
        self._initialize(**data)
        return self

    @classmethod
    def generate(
        cls,
        width: int = 4,
        base_radius: int = 20,
        noise: int = 10,
        nb_control_points: int = 10,
        samples_per_segment: int = 20,
        tension: float = 0.5,
    ) -> Track:
        """
        Generate a new Track instance

        :param width: Width of the track.
        :param base_radius: Average radius for generating control points
        :param noise: Maximum radial deviation added/subtracted from the base radius
        :param nb_control_points: Number of control points to generate
        :param samples_per_segment: Number of points to sample per segment between consecutive control points
        :param tension: Amount of tension to apply to the curve. Values between 0 and 1 increase the amount of tension
        :return: A new Track instance with generated centerline, left, and right boundaries
        """
        # Generate track
        control_pts = generate_track_control_points(
            nb_control_points, base_radius, noise
        )
        centerline = generate_centerline(control_pts, samples_per_segment, tension)
        left, right = get_left_right_track(centerline, width)

        # Instantiate track
        self = object.__new__(cls)
        self._initialize(centerline, left, right, width)
        return self

    def save_track(self, file_path: str):
        """
        Save the Track instance to a .npz file.

        :param file_path: Path to the file where the track data will be saved.
        """
        np.savez(
            file_path,
            centerline=self.centerline,
            left=self.left,
            right=self.right,
            width=self.width,
        )
