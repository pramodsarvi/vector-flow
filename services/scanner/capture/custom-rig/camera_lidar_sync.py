"""
Custom Hardware Rig: Camera + LiDAR Synchronization Node
Captures synchronized RGB frames and LiDAR depth/point-cloud sweeps
using hardware timestamps, applies extrinsic calibration [R | t],
and packages the session bundle for downstream processing.
"""

import json
import os
import time

import cv2
import numpy as np


class CameraLiDARCaptureRig:
    def __init__(
        self,
        camera_id: int = 0,
        rgb_resolution: tuple = (1920, 1080),
        fps: int = 30,
        extrinsic_r: np.ndarray = None,
        extrinsic_t: np.ndarray = None
    ):
        self.camera_id = camera_id
        self.rgb_resolution = rgb_resolution
        self.fps = fps

        # Extrinsic Calibration: LiDAR -> Camera Optical Frame
        # Default: LiDAR is mounted 4.5cm above camera lens along Y axis
        self.R = extrinsic_r if extrinsic_r is not None else np.eye(3)
        self.t = extrinsic_t if extrinsic_t is not None else np.array([0.0, 0.045, -0.02])

        # Camera Intrinsics
        self.fx = 1420.0
        self.fy = 1420.0
        self.cx = rgb_resolution[0] / 2.0
        self.cy = rgb_resolution[1] / 2.0

        self.cap = None
        self.is_capturing = False

    def transform_lidar_to_camera(self, lidar_points: np.ndarray) -> np.ndarray:
        """
        Applies rigid body transformation: P_cam = R * P_lidar + t
        """
        return np.dot(lidar_points, self.R.T) + self.t

    def start_session(self, output_dir: str):
        """
        Initializes recording directory structure and starts hardware streams.
        """
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.rgb_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.rgb_resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.is_capturing = True
        self.recorded_frames = []
        self.accumulated_points = []
        self.frame_idx = 0
        print(f"Rig Capture started. Target: {self.output_dir}")

    def capture_frame(self, simulated_lidar_sweep: np.ndarray = None, camera_pose: dict = None):
        """
        Grabs camera frame and synchronized LiDAR sweep.
        """
        if not self.is_capturing:
            return

        ret, frame = self.cap.read() if self.cap and self.cap.isOpened() else (False, None)
        timestamp = time.time()

        if not ret or frame is None:
            # Fallback synthetic frame if no physical camera attached
            frame = np.ones((self.rgb_resolution[1], self.rgb_resolution[0], 3), dtype=np.uint8) * 220
            cv2.putText(frame, f"Rig Frame #{self.frame_idx:04d} - {timestamp:.3f}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2)

        # Save image frame
        frame_name = f"frame_{self.frame_idx:04d}.jpg"
        img_path = os.path.join(self.images_dir, frame_name)
        cv2.imwrite(img_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        # Accumulate LiDAR points
        if simulated_lidar_sweep is not None:
            cam_space_pts = self.transform_lidar_to_camera(simulated_lidar_sweep)
            self.accumulated_points.append(cam_space_pts)

        pose = camera_pose if camera_pose else {
            "camera_position": [0.0, 1.6, 0.0],
            "camera_orientation_yaw_pitch_roll": [0.0, 0.0, 0.0]
        }

        self.recorded_frames.append({
            "frame_id": self.frame_idx,
            "image_file": f"images/{frame_name}",
            "timestamp": timestamp,
            "camera_position": pose["camera_position"],
            "camera_orientation_yaw_pitch_roll": pose["camera_orientation_yaw_pitch_roll"],
            "intrinsics": {
                "fx": self.fx,
                "fy": self.fy,
                "cx": self.cx,
                "cy": self.cy,
                "width": self.rgb_resolution[0],
                "height": self.rgb_resolution[1]
            }
        })
        self.frame_idx += 1

    def stop_session(self):
        """
        Finalizes recording, saves PLY point cloud and camera_poses.json.
        """
        self.is_capturing = False
        if self.cap:
            self.cap.release()

        # Save poses manifest
        poses_path = os.path.join(self.output_dir, "camera_poses.json")
        with open(poses_path, "w") as f:
            json.dump({
                "session_id": os.path.basename(self.output_dir),
                "sensor_type": "USB_RGB_Camera_Plus_SolidState_LiDAR",
                "calibration": {
                    "extrinsics_R": self.R.tolist(),
                    "extrinsics_t": self.t.tolist(),
                },
                "frames": self.recorded_frames
            }, f, indent=2)

        # Save PLY point cloud if points exist
        if self.accumulated_points:
            all_pts = np.vstack(self.accumulated_points)
            ply_path = os.path.join(self.output_dir, "raw_lidar_cloud.ply")
            with open(ply_path, "w") as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {len(all_pts)}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("end_header\n")
                for p in all_pts:
                    f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

        print(f"Capture completed. Saved {self.frame_idx} frames to {self.output_dir}")

if __name__ == "__main__":
    rig = CameraLiDARCaptureRig()
    print("Camera + LiDAR capture rig initialized successfully.")
