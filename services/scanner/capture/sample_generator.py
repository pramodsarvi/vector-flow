"""
Synthetic Room Data Generator for Camera + LiDAR Spatial Scanner
Generates:
1. Synthetic 3D LiDAR Point Cloud with noise and realistic room geometry (walls, floor, door opening, window, furniture)
2. Camera trajectory (6-DoF poses) and synthetic multi-view RGB images
3. Ground truth metadata for evaluation and automated testing
"""

import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw


def generate_synthetic_room(
    output_dir: str,
    room_length: float = 6.0,   # X axis in meters
    room_width: float = 4.5,    # Z axis in meters
    room_height: float = 2.8,   # Y axis in meters
    point_density: int = 40000,
    noise_std: float = 0.015,   # 1.5 cm LiDAR noise
):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    points = []
    colors = []

    # 1. Floor points (Y = 0) with wood/tile color pattern
    n_floor = int(point_density * 0.3)
    fx = np.random.uniform(0, room_length, n_floor)
    fz = np.random.uniform(0, room_width, n_floor)
    fy = np.random.normal(0, noise_std, n_floor)
    for x, y, z in zip(fx, fy, fz):
        points.append([x, y, z])
        # wood plank color pattern
        c = 140 + int(20 * math.sin(x * 6) + 10 * math.cos(z * 4))
        colors.append([c, int(c * 0.7), int(c * 0.4)])

    # 2. Ceiling points (Y = room_height)
    n_ceil = int(point_density * 0.15)
    cx = np.random.uniform(0, room_length, n_ceil)
    cz = np.random.uniform(0, room_width, n_ceil)
    cy = room_height + np.random.normal(0, noise_std, n_ceil)
    for x, y, z in zip(cx, cy, cz):
        points.append([x, y, z])
        colors.append([235, 235, 240])

    # 3. Walls
    n_wall_pts = int(point_density * 0.1)

    # Wall 1: Z = 0 (Front wall, with door opening at X in [3.8, 4.7], Y in [0, 2.1])
    door_min_x, door_max_x, door_h = 3.8, 4.7, 2.1
    w1_x = np.random.uniform(0, room_length, n_wall_pts)
    w1_y = np.random.uniform(0, room_height, n_wall_pts)
    w1_z = np.random.normal(0, noise_std, n_wall_pts)
    for x, y, z in zip(w1_x, w1_y, w1_z):
        if not (door_min_x <= x <= door_max_x and y <= door_h):
            points.append([x, y, z])
            colors.append([220, 225, 230])

    # Wall 2: X = room_length (Right wall)
    w2_z = np.random.uniform(0, room_width, n_wall_pts)
    w2_y = np.random.uniform(0, room_height, n_wall_pts)
    w2_x = room_length + np.random.normal(0, noise_std, n_wall_pts)
    for x, y, z in zip(w2_x, w2_y, w2_z):
        points.append([x, y, z])
        colors.append([218, 222, 226])

    # Wall 3: Z = room_width (Back wall, with window opening at X in [1.5, 3.5], Y in [0.9, 2.2])
    win_min_x, win_max_x, win_sill, win_top = 1.5, 3.5, 0.9, 2.2
    w3_x = np.random.uniform(0, room_length, n_wall_pts)
    w3_y = np.random.uniform(0, room_height, n_wall_pts)
    w3_z = room_width + np.random.normal(0, noise_std, n_wall_pts)
    for x, y, z in zip(w3_x, w3_y, w3_z):
        if not (win_min_x <= x <= win_max_x and win_sill <= y <= win_top):
            points.append([x, y, z])
            colors.append([222, 226, 230])

    # Wall 4: X = 0 (Left wall)
    w4_z = np.random.uniform(0, room_width, n_wall_pts)
    w4_y = np.random.uniform(0, room_height, n_wall_pts)
    w4_x = np.random.normal(0, noise_std, n_wall_pts)
    for x, y, z in zip(w4_x, w4_y, w4_z):
        points.append([x, y, z])
        colors.append([215, 220, 225])

    # 4. Furniture Obstacles (e.g. Sofa and Table on floor)
    # Sofa bounding box: X in [0.5, 2.5], Z in [0.5, 1.4], Y in [0, 0.85]
    n_sofa = int(point_density * 0.08)
    sx = np.random.uniform(0.5, 2.5, n_sofa)
    sz = np.random.uniform(0.5, 1.4, n_sofa)
    sy = np.random.uniform(0.0, 0.85, n_sofa)
    for x, y, z in zip(sx, sy, sz):
        points.append([x, y, z])
        colors.append([60, 90, 130])

    # Table bounding box: X in [3.0, 4.2], Z in [2.2, 3.2], Y in [0, 0.75]
    n_table = int(point_density * 0.05)
    tx = np.random.uniform(3.0, 4.2, n_table)
    tz = np.random.uniform(2.2, 3.2, n_table)
    ty = np.random.uniform(0.0, 0.75, n_table)
    for x, y, z in zip(tx, ty, tz):
        points.append([x, y, z])
        colors.append([160, 110, 70])

    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)

    # Save point cloud as PLY
    ply_path = os.path.join(output_dir, "raw_lidar_cloud.ply")
    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")

    # 5. Generate Camera Trajectory (Circular walkthrough inside the room)
    # Standing eye height Y = 1.6m
    n_frames = 12
    poses = []
    center_x = room_length / 2.0
    center_z = room_width / 2.0
    radius = min(room_length, room_width) * 0.3

    for i in range(n_frames):
        angle = (2.0 * math.pi * i) / n_frames
        cam_x = center_x + radius * math.cos(angle)
        cam_z = center_z + radius * math.sin(angle)
        cam_y = 1.6  # eye level in meters

        # Camera looks tangentially around the room
        look_angle = angle + math.pi / 2.0

        frame_name = f"frame_{i:04d}.jpg"
        img_path = os.path.join(images_dir, frame_name)

        # Generate a stylized synthetic camera frame
        img = Image.new("RGB", (640, 480), color=(220, 225, 230))
        draw = ImageDraw.Draw(img)
        # draw floor horizon
        draw.rectangle([(0, 260), (640, 480)], fill=(150, 105, 60))
        # draw room label
        draw.text((20, 20), f"Camera View #{i:02d} | Yaw: {math.degrees(look_angle):.1f}°", fill=(30, 30, 30))
        draw.text((20, 45), f"Pos: ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f})m", fill=(50, 50, 50))
        img.save(img_path, quality=90)

        poses.append({
            "frame_id": i,
            "image_file": f"images/{frame_name}",
            "timestamp": 1700000000.0 + i * 0.5,
            "camera_position": [round(cam_x, 4), round(cam_y, 4), round(cam_z, 4)],
            "camera_orientation_yaw_pitch_roll": [round(math.degrees(look_angle), 2), -5.7, 0.0],
            "intrinsics": {
                "fx": 525.0,
                "fy": 525.0,
                "cx": 320.0,
                "cy": 240.0,
                "width": 640,
                "height": 480
            }
        })

    # Save poses JSON
    poses_path = os.path.join(output_dir, "camera_poses.json")
    with open(poses_path, "w") as f:
        json.dump({
            "session_id": "synthetic_demo_01",
            "sensor_type": "RGB_Camera_Plus_LiDAR",
            "point_count": len(points),
            "ground_truth_dimensions": {
                "length_m": room_length,
                "width_m": room_width,
                "height_m": room_height,
                "area_sqm": room_length * room_width
            },
            "frames": poses
        }, f, indent=2)

    print(f"Generated synthetic scan at: {output_dir}")
    print(f" - Point Cloud: {ply_path} ({len(points)} points)")
    print(f" - Camera frames: {len(poses)} in {images_dir}")
    print(f" - Poses & Intrinsics: {poses_path}")

if __name__ == "__main__":
    import sys
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample-data")
    out_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    generate_synthetic_room(out_dir)
