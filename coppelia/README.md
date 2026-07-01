# CoppeliaSim Integration

Este directorio contiene las herramientas para la interfaz de simulación con CoppeliaSim, incluyendo la generación programática de escenarios, la recolección automatizada de datasets y el seguimiento de trayectoria en lazo cerrado.

## Contenido del directorio

- **`build_scene.py`**: Genera el entorno en CoppeliaSim (corredor en S, ubicación de landmarks y posiciones de inicio/meta) utilizando la API remota ZeroMQ.
- **`record_dataset.py`**: Conduce el robot de manera preprogramada a través del escenario para recolectar datos brutos de sensores (LiDAR nativo SICK S300, odometría, IMU y verdad terreno) y guardarlos en un archivo HDF5 (`dataset.h5`).
- **`run_navigation.py`**: Controlador de seguimiento en lazo cerrado. Carga la ruta calculada por el RRT* y ejecuta un seguimiento de trayectoria mediante **Pure Pursuit**, utilizando la pose estimada por el filtro en línea.
- **`vehicle.py`**: Implementa la interfaz del robot articulado G2T en CoppeliaSim, imponiendo las restricciones cinemáticas calculadas mediante integración RK4 sobre los cuerpos simulados.
- **`_client.py`**: Envoltura para establecer la conexión con la API ZeroMQ de CoppeliaSim.
- **`scenario.py`**: Definiciones geométricas del escenario y utilidades para su generación.
- **`lidar_calibrate.py`**, **`lidar_diag.py`**, **`lidar_probe.py`**, **`lidar_probe2.py`**, **`lidar_read.py`**: Scripts de diagnóstico y calibración para alinear el LiDAR de CoppeliaSim con el sistema de coordenadas del robot (determina el signo de la transformación $s=-1$).

## Requisitos de Ejecución

Para ejecutar cualquier script de esta carpeta, primero debes abrir la escena correspondiente (`g3_scene.ttt`) en CoppeliaSim para que la API ZeroMQ esté activa:

```bash
/Applications/coppeliaSim.app/Contents/MacOS/coppeliaSim coppelia/g3_scene.ttt
```

Luego, puedes ejecutar el seguimiento de lazo cerrado (luego de haber corrido la planificación):

```bash
python3 -m coppelia.run_navigation
```
