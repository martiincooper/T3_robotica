# Herramientas y Evaluación de Robustez

Este directorio agrupa los scripts utilitarios, validaciones del dataset y las herramientas de simulación de robustez Monte Carlo.

## Contenido del directorio

- **`monte_carlo.py`**: Motor de simulación Monte Carlo. Corre el pipeline de estimación (EKF-SLAM y Fusión) de forma repetida variando semillas aleatorias (8 semillas) y niveles de ruido (nominal vs. ruido duplicado) sobre dos escenarios diferentes. Guarda los resultados en `evaluation/monte_carlo_results.csv` y genera gráficos comparativos en formato de boxplot.
- **`evaluate_all.py`**: Computa de forma global el rendimiento de estimación y control del lazo cerrado. Calcula RMSE de posición, rumbo y ángulos de articulación ($\psi_1, \psi_2$), distancias a obstáculos y el error final a la meta.
- **`validate_dataset.py`**: Valida que la convención de reconstrucción de la nube de puntos LiDAR sea correcta contra la verdad terreno de las paredes y cilindros.
- **`preview_scenario.py`**: Genera un mapa gráfico del escenario de simulación mostrando las posiciones relativas de inicio, meta y los 12 cilindros de referencia.

## Ejecución

Para correr las evaluaciones del sistema completo:

```bash
# Validar el dataset recolectado
python3 -m tools.validate_dataset

# Ejecutar el análisis completo de métricas
python3 -m tools.evaluate_all

# Ejecutar el barrido de robustez Monte Carlo
python3 -m tools.monte_carlo
```
