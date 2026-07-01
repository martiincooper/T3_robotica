# G2T Core (Código Base de Guía 2)

Este directorio contiene el código base importado (*vendored*) de la **Guía 2** de la asignatura. Proporciona las librerías cinemáticas fundamentales y los estimadores preliminares que sustentan la simulación e integración de esta guía.

## Contenido del directorio

- **`simulation/g2t_sim/kinematics.py`**: Implementa las ecuaciones del modelo cinemático generalizado $N$-trailer *off-axle* de Altafini e integrador Runge-Kutta de 4to orden (RK4). Utilizado para simular las poses en bucle cerrado.
- **`perception/`**: Algoritmos de segmentación y detección geométrica basados en RANSAC para estimar los ángulos de articulación del remolque a partir de nubes de puntos LiDAR 2D limpias.
- **`fusion/`**: Contiene el EKF básico de la Guía 2 encargado de estimar la pose del tractor y los ángulos de articulación a partir de mediciones relativas.
- **`evaluation/`**: Métricas de error geométrico y funciones de apoyo para el cálculo de diferencias de pose y rumbo.
