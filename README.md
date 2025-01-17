Maestría en Inteligencia Artificial - UNI
 
Integrantes G8:
- Fernando Boza Gutarra
- Johan Manuel Callomamani Buendia
- Kevin Gómez Villanueva	
- Umbert Lewis de la Cruz Rodriguez
- Yovany Romero Ramos


# Deep Q-Network Training

Este proyecto implementa un agente **Deep Q-Network (DQN)** para jugar MsPacman utilizando PyTorch y Gymnasium. El agente emplea experiencia de repetición, redes Q local y objetivo, y preprocesamiento de imágenes.

## Instalación

1. **Crear y activar un entorno virtual**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuración del archivo `hyperparameters.txt`

Crea un archivo llamado `hyperparameters.txt` en el directorio raíz del proyecto. Este archivo debe contener los hiperparámetros necesarios en el siguiente formato:

```plaintext
# Hiperparámetros del entrenamiento
env_name = 'ALE/MsPacman-v5'                # Nombre del entorno en Gymnasium
number_episodes = 2000                      # Número total de episodios
maximum_number_timesteps_per_episode = 10000 # Pasos máximos por episodio
learning_rate = 0.0005                      # Tasa de aprendizaje
minibatch_size = 256                        # Tamaño del minibatch
discount_factor = 0.99                      # Factor de descuento (gamma)
max_memory = 20000                          # Tamaño máximo de memoria de experiencia
patience = 1000                             # Episodios sin mejora para early stopping
earlystop_threshold = 0.01                  # Umbral de mejora mínima para early stopping
epsilon_starting_value = 1.0                # Valor inicial de epsilon
epsilon_ending_value = 0.01                 # Valor mínimo de epsilon
epsilon_decay_value = 0.98                  # Factor de decaimiento de epsilon
```

### Descripción de los hiperparámetros

- **`env_name`**: Nombre del entorno que se utilizará.
- **`number_episodes`**: Número total de episodios para el entrenamiento.
- **`maximum_number_timesteps_per_episode`**: Máximo de pasos por episodio antes de finalizar.
- **`learning_rate`**: Tasa de aprendizaje para el optimizador Adam.
- **`minibatch_size`**: Cantidad de muestras usadas en cada actualización de la red.
- **`discount_factor`**: Factor de descuento para calcular valores futuros.
- **`max_memory`**: Capacidad máxima de la memoria de experiencia.
- **`patience`**: Número de episodios sin mejora antes de detener el entrenamiento.
- **`earlystop_threshold`**: Mejora mínima necesaria para reiniciar el contador de early stopping.
- **`epsilon_starting_value`**: Valor inicial para la política epsilon-greedy.
- **`epsilon_ending_value`**: Valor mínimo para epsilon.
- **`epsilon_decay_value`**: Factor de decaimiento para reducir epsilon en cada episodio.

## Ejecución

1. **Asegúrate de haber configurado el archivo `hyperparameters.txt`.**
2. Ejecuta el código principal:
   ```bash
   python Pac-Man-training.py
   ```

## Outputs

El proceso de entrenamiento generará los siguientes resultados en el directorio de experimentos:

1. **Checkpoints de los modelos**:
   - Guardados cada 100 episodios en `./exp_<UUID>/checkpoints/`.

2. **Gráficas**:
   - `scores_plot.png`: Evolución del puntaje por episodio.
   - `loss_plot.png`: Pérdida acumulada durante el entrenamiento.
   - `epsilon_plot.png`: Decaimiento de epsilon.
   - `td_error_plot.png`: Evolución del error TD.
   - `steps_plot.png`: Pasos por episodio.
   - `action_distribution_plot.png`: Distribución de acciones por episodio.
   - `heatmap_visits.png`: Mapa de calor de visitas acumuladas.

3. **Estadísticas en consola**:
   - Puntaje por episodio.
   - Promedio móvil de los últimos 100 episodios.
   - Notificaciones de early stopping o resolución del entorno.

## Notas
- Asegúrate de contar con una GPU disponible para acelerar el entrenamiento.
- Para modificar el entorno o hiperparámetros, edita el archivo `hyperparameters.txt` y reinicia el proceso de entrenamiento.

