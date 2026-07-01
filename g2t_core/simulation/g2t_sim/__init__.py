"""G2T simulator package.

Self-contained Python simulator that produces a synthetic LaserScan dataset
for a generalized tractor-with-two-trailers (G2T) articulated vehicle. The
purpose of this package is *not* to replace CoppeliaSim/Gazebo but to make
the rest of the project (perception, fusion, evaluation) fully reproducible
without any robotics middleware installed.

A thin ROS 2 bridge (`ros_ws/src/g2t_bringup`) is provided separately so
that the very same kinematic and sensor models can be run inside a real
ROS environment when one is available.
"""

from .kinematics import G2TKinematics, G2TState           # noqa: F401
from .lidar import Lidar2D, LidarParams                   # noqa: F401
from .trajectories import build_trajectory                # noqa: F401
from .world import World, Obstacle, Pedestrian            # noqa: F401
