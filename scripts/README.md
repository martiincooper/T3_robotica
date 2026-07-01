# Scripts Auxiliares

Este directorio contiene herramientas auxiliares complementarias para la integración del pipeline de simulación con otros entornos middleware como ROS 2.

## Contenido del directorio

- **`h5_to_rosbag.py`**: Exporta los conjuntos de datos HDF5 generados en la simulación a un formato de grabación de tópicos nativo de ROS 2 (`sqlite3` o `mcap`). Mapea y serializa los datos como mensajes estándar:
  - `/scan` -> `sensor_msgs/msg/LaserScan`
  - `/odom` -> `nav_msgs/msg/Odometry`
  - `/imu/data` -> `sensor_msgs/msg/Imu`
  - `/tf` -> `tf2_msgs/msg/TFMessage` (Verdad terreno del tractor y remolques para visualización en RViz 2).
