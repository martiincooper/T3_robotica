# Parámetros y Configuración

Este directorio almacena los archivos YAML con los parámetros geométricos del vehículo G2T, configuraciones de los niveles de ruido y especificaciones físicas del escenario de simulación.

## Contenido del directorio

- **`scenario_g3.yaml`**: Parámetros nominales de la simulación para el Escenario 1 (S1). Define las dimensiones físicas del tractor y remolques, topes de articulación, ruido en actuadores/sensores, y la posición de los 12 cilindros de referencia.
- **`scenario_g3_alternative.yaml`**: Parámetros de configuración del Escenario 2 (S2), utilizado para evaluar la robustez y la generalización de la estimación probabilística del EKF sobre un mapa no visto previamente.
