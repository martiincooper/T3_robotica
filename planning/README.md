# Planificación de Trayectoria: RRT*

Este directorio contiene la implementación del algoritmo RRT* (Rapidly-exploring Random Trees Star) adaptado para el vehículo articulado G2T.

## Contenido del directorio

- **`rrt_star.py`**: Algoritmo de planificación de trayectorias óptimas RRT*. Considera:
  - Mapeo directo sobre los landmarks y paredes estimados por EKF-SLAM.
  - Inflación geométrica de obstáculos ($ clearance = 0.65\text{ m}$) para acomodar el ancho del tractor y el barrido lateral de los remolques.
  - Suavizado de trayectoria mediante método *shortcut*.
  - Verificación cinemática usando el radio mínimo de giro admisible ($R_{\min} = L_0 / \tan(35^\circ)$).
- **`run_planning.py`**: Script ejecutor que corre el planificador sobre el mapa estimado del escenario base, visualiza la trayectoria generada y exporta la ruta óptima a `planning/rrt_path.csv` para su posterior seguimiento.

## Ejecución

Para generar una trayectoria óptima desde la posición inicial a la meta:

```bash
python3 -m planning.run_planning
```
