# Estimación: EKF-SLAM y EKF de Fusión Sensorial

Este directorio contiene los estimadores basados en filtros de Kalman extendidos (EKF) para la localización del vehículo, mapeo de landmarks y fusión de sensores.

## Contenido del directorio

- **`ekf_slam.py`**: Filtro de Kalman Extendido para la localización simultánea y mapeo (EKF-SLAM). El vector de estado se aumenta dinámicamente al detectar nuevos landmarks cilíndricos en el entorno.
- **`run_slam.py`**: Ejecuta el pipeline de EKF-SLAM de forma offline sobre el dataset recolectado. Genera métricas de RMSE de posición y rumbo, y guarda gráficos de desempeño.
- **`ekf_fusion.py`**: Implementa el EKF de fusión sensorial de 6 dimensiones que estima el estado $[x_t, y_t, \theta_t, \psi_1, \psi_2, \omega_t]$. Combina las lecturas de odometría, IMU, ángulo de articulación medido y la pose global corregida por el EKF-SLAM.
- **`run_fusion.py`**: Corre el filtro de fusión de forma batch y evalúa el error de estimación de los ángulos de articulación ($\psi_1, \psi_2$) contra la verdad terreno de la simulación.
- **`landmarks.py`**: Algoritmo de detección y ajuste de círculos mediante el método de Kasa en la nube de puntos LiDAR. Implementa la asociación de datos con compuerta de Mahalanobis ($\chi^2$) y landmarks provisionales.
- **`inputs.py`**: Módulo utilitario encargado de cargar y parsear la información de sensores e IMU guardada en el formato HDF5.

## Ejecución

Puedes ejecutar cualquiera de los estimadores de forma offline utilizando los datos guardados en `datasets/`:

```bash
# Ejecutar EKF-SLAM
python3 -m estim.run_slam

# Ejecutar EKF de Fusión Sensorial
python3 -m estim.run_fusion
```
