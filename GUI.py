import math

import numpy as np
import pybullet as p

from GenTrack import Track, filter_left_right_track


class GUI:
    """Class responsible for handling the Pybullet GUI, including initialization and drawing of the track and cars."""

    @staticmethod
    def init(mode=p.GUI, fps=120):
        """Initialize the Pybullet GUI with the specified mode and frames per second."""
        p.connect(mode)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

        p.setGravity(0, 0, 0)

        p.setTimeStep(1 / fps)
        p.setRealTimeSimulation(0)

    @staticmethod
    def set_camera_position_to(track: Track):
        """Set the camera position in the Pybullet GUI to have a top-down view of the track."""
        window_width, window_height = p.getDebugVisualizerCamera()[:2]
        camera_fov = 90

        x_coords = [pt[0] for pt in track.centerline]
        y_coords = [pt[1] for pt in track.centerline]

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
        max_dim += track.width + 2

        p.resetDebugVisualizerCamera(
            cameraDistance=(max_dim / 2.0) / math.tan(math.radians(camera_fov / 2.0)),
            cameraYaw=0,
            cameraPitch=-89.999,  # presque -90 = vue du dessus
            cameraTargetPosition=[x_mid, y_mid, 0],
        )

    @staticmethod
    def draw_track_line(track: Track, cars: list):
        """Draw the track lines in the Pybullet GUI"""
        wall_thickness = 0.2
        wall_height = 0.1
        for border in [track.left, track.right]:
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

                for car in cars:
                    p.setCollisionFilterPair(
                        car.obj_id, wall, -1, -1, enableCollision=0
                    )

    @staticmethod
    def draw_car(car):
        """Draw a car in the Pybullet GUI at its current position and orientation."""
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.3, 0.1])
        vis = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.5, 0.3, 0.01],
            rgbaColor=[0.5, 0.5, 0.5, 1],
        )

        # --- Pointe de la flèche (cône)
        arrow_head_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.3, 0.01, 0.1],
            rgbaColor=[1, 0, 0, 1],
        )

        # Orientation pour mettre cylindre/cone dans le plan XY
        rot = p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 2])

        obj_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[car.x, car.y, 0.25],
            baseOrientation=p.getQuaternionFromEuler([0, 0, car.yaw]),
            linkMasses=[0],
            linkCollisionShapeIndices=[-1],  # PAS de collision
            linkVisualShapeIndices=[arrow_head_vis],
            linkPositions=[
                [1, 0, 0],  # pointe
            ],
            linkOrientations=[rot],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0]],
        )

        p.changeDynamics(obj_id, -1, linearDamping=0.5, angularDamping=2)
        return obj_id
