#!/usr/bin/env python3
"""Convert the HDF5 simulation dataset into a ROS 2 bag (mcap or sqlite3).

Requires ROS 2 (rclpy + rosbag2_py) and is therefore intentionally kept
out of the main test pipeline. From a sourced ROS 2 environment::

    python scripts/h5_to_rosbag.py \
        --dataset datasets/sim_run/dataset.h5 \
        --out     datasets/sim_run/bag

Topics produced (mirrors what ``ros_ws/src/g2t_bringup`` expects):

    /scan             sensor_msgs/msg/LaserScan
    /odom             nav_msgs/msg/Odometry
    /imu/data         sensor_msgs/msg/Imu
    /ground_truth/g2t g2t_msgs/msg/G2TState         (custom)
    /tf               tf2_msgs/msg/TFMessage        (tractor/trailer poses)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path


def _require_ros() -> None:
    try:
        import rclpy  # noqa: F401
        import rosbag2_py  # noqa: F401
        import sensor_msgs.msg  # noqa: F401
        import nav_msgs.msg  # noqa: F401
        import geometry_msgs.msg  # noqa: F401
        import tf2_msgs.msg  # noqa: F401
    except ImportError as e:  # pragma: no cover
        print(f"This script requires ROS 2 to be installed and sourced: {e}",
              file=sys.stderr)
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--storage", choices=["mcap", "sqlite3"], default="mcap")
    args = ap.parse_args()
    _require_ros()

    import h5py
    import numpy as np
    import rclpy
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry
    from rclpy.serialization import serialize_message
    import rosbag2_py
    from sensor_msgs.msg import Imu, LaserScan
    from tf2_msgs.msg import TFMessage

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(args.out), storage_id=args.storage),
        rosbag2_py.ConverterOptions("", ""))

    def add_topic(name: str, type_str: str) -> None:
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=name, type=type_str, serialization_format="cdr"))

    add_topic("/scan", "sensor_msgs/msg/LaserScan")
    add_topic("/odom", "nav_msgs/msg/Odometry")
    add_topic("/imu/data", "sensor_msgs/msg/Imu")
    add_topic("/tf", "tf2_msgs/msg/TFMessage")

    def to_time(t: float) -> tuple[int, int]:
        sec = int(t); nsec = int((t - sec) * 1e9)
        return sec, nsec

    def stamp(t: float) -> Time:
        s, n = to_time(t)
        out = Time(); out.sec = s; out.nanosec = n; return out

    with h5py.File(args.dataset, "r") as f:
        angles = f["scan/angles"][:]
        ranges = f["scan/ranges"][:]
        scan_t = f["scan/time"][:]
        odom_t = f["odom/time"][:]; odom_v = f["odom/v_noisy"][:]; odom_w = f["odom/omega_noisy"][:]
        imu_t = f["imu/time"][:];  imu_g = f["imu/gyro_z"][:]
        gt_p0 = f["gt/tractor_pose"][:]
        gt_p1 = f["gt/trailer1_pose"][:]
        gt_p2 = f["gt/trailer2_pose"][:]
        gt_t = f["time"][:]
        range_min = float(f["scan/range_min"][()])
        range_max = float(f["scan/range_max"][()])

    # --- LaserScans ---------------------------------------------------------
    for i, t in enumerate(scan_t):
        msg = LaserScan()
        msg.header.stamp = stamp(float(t))
        msg.header.frame_id = "lidar"
        msg.angle_min = float(angles[0])
        msg.angle_max = float(angles[-1])
        msg.angle_increment = float(angles[1] - angles[0])
        msg.time_increment = 0.0
        msg.scan_time = 1.0 / 40.0
        msg.range_min = range_min
        msg.range_max = range_max
        r = ranges[i].copy()
        r[~np.isfinite(r)] = range_max + 1.0
        msg.ranges = r.astype(np.float32).tolist()
        s, n = to_time(float(t))
        writer.write("/scan", serialize_message(msg), s * 10**9 + n)

    # --- Odometry -----------------------------------------------------------
    for i, t in enumerate(odom_t):
        msg = Odometry()
        msg.header.stamp = stamp(float(t)); msg.header.frame_id = "odom"
        msg.child_frame_id = "tractor"
        msg.twist.twist.linear.x = float(odom_v[i])
        msg.twist.twist.angular.z = float(odom_w[i])
        s, n = to_time(float(t))
        writer.write("/odom", serialize_message(msg), s * 10**9 + n)

    # --- IMU ---------------------------------------------------------------
    for i, t in enumerate(imu_t):
        msg = Imu()
        msg.header.stamp = stamp(float(t)); msg.header.frame_id = "imu"
        msg.angular_velocity.z = float(imu_g[i])
        s, n = to_time(float(t))
        writer.write("/imu/data", serialize_message(msg), s * 10**9 + n)

    # --- TF (ground truth) -------------------------------------------------
    def make_tf(parent: str, child: str, x: float, y: float, yaw: float,
                t: float) -> TransformStamped:
        ts = TransformStamped()
        ts.header.stamp = stamp(t); ts.header.frame_id = parent
        ts.child_frame_id = child
        ts.transform.translation.x = float(x)
        ts.transform.translation.y = float(y)
        ts.transform.rotation.z = math.sin(yaw / 2.0)
        ts.transform.rotation.w = math.cos(yaw / 2.0)
        return ts

    for i, t in enumerate(gt_t):
        msg = TFMessage()
        msg.transforms = [
            make_tf("map", "tractor", *gt_p0[i], float(t)),
            make_tf("tractor", "trailer1", float(gt_p1[i, 0] - gt_p0[i, 0]),
                    float(gt_p1[i, 1] - gt_p0[i, 1]),
                    float(gt_p1[i, 2] - gt_p0[i, 2]), float(t)),
            make_tf("trailer1", "trailer2", float(gt_p2[i, 0] - gt_p1[i, 0]),
                    float(gt_p2[i, 1] - gt_p1[i, 1]),
                    float(gt_p2[i, 2] - gt_p1[i, 2]), float(t)),
        ]
        s, n = to_time(float(t))
        writer.write("/tf", serialize_message(msg), s * 10**9 + n)

    print(f"Wrote bag to {args.out} ({len(scan_t)} scans, {len(odom_t)} odom, "
          f"{len(imu_t)} imu, {len(gt_t)} tf frames).")


if __name__ == "__main__":
    main()
