"""
-------------------------------------------------------------------------------
Trabajo Final - G8 Sección A

Nombre del código  : Deep Q-Network Training
Curso: APRENDIZAJE POR REFORZAMIENTO
Fecha de creación   : 14/01/2025

Descripción :
Este código implementa y entrena un agente Deep Q-Network para jugar MsPacman usando PyTorch y Gymnasium.
El agente utiliza técnicas como experiencia de repetición, redes Q local y objetivo, y preprocesamiento de frames.

Integrantes:
* Boza Gutarra, Fernando
* Callomamani Buendia, Johan Manuel
* De La Cruz Rodríguez, Lewis Umbert
* Gomez Villanueva, Kevin
* Romero Ramos, Yovany
-------------------------------------------------------------------------------
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import uuid

# ------------------- Leer Hiperparámetros desde un archivo ------------------- #
# ----------- Desarrollado por De La Cruz Rodríguez, Lewis Umbert ------------- #
def load_hyperparameters(file_path):
    """
    Carga los hiperparámetros desde un archivo de texto y devuelve un diccionario.
    """
    hyperparameters = {}
    with open(file_path, "r") as file:
        for line in file:
            if line.strip() and not line.startswith("#"):
                key, value = line.split("=")
                key = key.strip()
                value = value.strip()
                try:
                    value = eval(value)  # Intentar convertir a tipo numérico o lista
                except:
                    pass  # Mantener como cadena si no se puede convertir
                hyperparameters[key] = value
    return hyperparameters

# Ruta del archivo de hiperparámetros
HYPERPARAMS_FILE = "hyperparameters.txt"
if not os.path.exists(HYPERPARAMS_FILE):
    raise FileNotFoundError(f"El archivo de hiperparámetros '{HYPERPARAMS_FILE}' no existe.")

hyperparameters = load_hyperparameters(HYPERPARAMS_FILE)

# Asignar valores de hiperparámetros a variables
env_name = hyperparameters["env_name"]
number_episodes = hyperparameters["number_episodes"]
maximum_number_timesteps_per_episode = hyperparameters["maximum_number_timesteps_per_episode"]
learning_rate = hyperparameters["learning_rate"]
minibatch_size = hyperparameters["minibatch_size"]
discount_factor = hyperparameters["discount_factor"]
MAX_MEMORY = hyperparameters["max_memory"]
PATIENCE = hyperparameters["patience"]
EARLYSTOP_THRESHOLD = hyperparameters["earlystop_threshold"]
epsilon_starting_value = hyperparameters["epsilon_starting_value"]
epsilon_ending_value = hyperparameters["epsilon_ending_value"]
epsilon_decay_value = hyperparameters["epsilon_decay_value"]

# ------------------- Configuración del entorno ------------------- #
env = gym.make(env_name, full_action_space=False)
state_shape = env.observation_space.shape
number_actions = env.action_space.n

print("State shape:", state_shape)
print("Number of actions:", number_actions)

# ------------------- Carpeta de experimentos ------------------- #
EXPERIMENT_FOLDER = f"exp_{str(uuid.uuid4())}"

try:
    os.mkdir(EXPERIMENT_FOLDER)
    print(f"Directory '{EXPERIMENT_FOLDER}' created successfully.")
except FileExistsError:
    print(f"Directory '{EXPERIMENT_FOLDER}' already exists.")

try:
    os.mkdir(f'./{EXPERIMENT_FOLDER}/checkpoints')
    print(f"Directory checkpoints created successfully.")
except FileExistsError:
    print(f"Directory checkpoints' already exists.")

# ------------------- Definición de Red Neuronal ------------------- #
# --------- Desarrollado por Boza Gutarra, Fernando ---------------- #

class Network(nn.Module):
    """
    Red neuronal convolucional para estimar los valores Q.
    """
    def __init__(self, action_size, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(10 * 10 * 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, state):
        """Realiza el pase hacia adelante en la red."""
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def preprocess_frame(frame):
    """
    Preprocesa el frame del juego para el modelo.
    Cambia la resolución a 128x128 y convierte a tensor.
    """
    frame = Image.fromarray(frame)
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    return preprocess(frame).unsqueeze(0)

# ------------------- Definición de Agente ------------------- #
# ---- Desarrollado por Callomamani Buendia, Johan Manuel ---- #

class Agent:
    """
    Define un agente que utiliza una red Q para aprender y jugar.
    """
    def __init__(self, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size

        # Redes Q local y objetivo
        self.local_qnetwork = Network(action_size).to(self.device)
        self.target_qnetwork = Network(action_size).to(self.device)

        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=MAX_MEMORY)

        # Para registro de pérdidas (loss) en cada batch de aprendizaje
        self.losses = []

        # Para registro de TD-errors (guardaremos la diferencia q_expected - q_targets)
        self.td_errors = []

        # Para ver la distribución de Q-values de forma periódica
        self.q_values_samples = []

    def step(self, state, action, reward, next_state, done):
        """
        Almacena la experiencia y, si hay suficientes muestras, entrena la red.
        """
        state = preprocess_frame(state)
        next_state = preprocess_frame(next_state)
        self.memory.append((state, action, reward, next_state, done))

        # Cada vez que tenemos minibatch_size o más en memoria, entrenamos
        if len(self.memory) > minibatch_size:
            experiences = random.sample(self.memory, k=minibatch_size)
            self.learn(experiences, discount_factor)

    def act(self, state, epsilon=0.0):
        """
        Selecciona acción con política epsilon-greedy.
        """
        state = preprocess_frame(state).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()

        # Epsilon-greedy
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Actualiza los parámetros de la red Q usando un batch de experiencias.
        """
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.cat(states).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(1).to(self.device)
        next_states = torch.cat(next_states).float().to(self.device)
        dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float().unsqueeze(1).to(self.device)

        # Q valores objetivo
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * next_q_targets * (1 - dones))

        # Q valores esperados (de la red local)
        q_expected = self.local_qnetwork(states).gather(1, actions)

        # Cálculo de la pérdida (MSE)
        loss = F.mse_loss(q_expected, q_targets)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Guardamos la pérdida para graficar
        self.losses.append(loss.item())

        # Guardamos la TD-error (podemos tomar la diferencia)
        td_error = (q_expected - q_targets).detach().cpu().numpy()  # shape: [batch_size, 1]
        self.td_errors.extend(td_error.flatten())  # lo aplanamos y lo guardamos

        # Actualizamos la red objetivo con la local cada cierto tiempo
        self.soft_update(tau=1e-3)

    def soft_update(self, tau=1e-3):
        """
        Actualiza los pesos de la red objetivo con la red local
        usando factor tau (soft update).
        """
        for target_param, local_param in zip(self.target_qnetwork.parameters(),
                                             self.local_qnetwork.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def sample_q_values(self, sample_states):
        """
        Dado un lote de estados, extrae los Q-values con la red local.
        Se usa para visualizar la distribución de Q-values cada cierto número de episodios.
        """
        self.local_qnetwork.eval()
        with torch.no_grad():
            q_vals = self.local_qnetwork(sample_states.to(self.device))
        self.local_qnetwork.train()
        return q_vals.cpu().numpy()

    def load(self, file_name):
        """
        Carga pesos de un archivo .pth.
        """
        checkpoint = torch.load(file_name, map_location=self.device)
        self.local_qnetwork.load_state_dict(checkpoint)
        self.target_qnetwork.load_state_dict(checkpoint)

# ------------------- Entrenamiento del Agente ------------------- #
# -------- Desarrollado por Gomez Villanueva, Kevin -------------- #
epsilon = epsilon_starting_value

scores_on_100_episodes = deque(maxlen=100)

# Listas para registro de métricas
all_scores = []  # Score por episodio
avg_scores = []  # Promedio móvil de score (100 últimos episodios)
eps_history = []  # Epsilon por episodio
all_steps = []  # Cantidad de pasos por episodio
action_distribution_history = []  # Frecuencia (proporción) de acciones por episodio

# Para el heatmap global de visitas
track_visits = True
accumulated_heatmap = np.zeros((128, 128), dtype=np.int64) if track_visits else None

# Para early stopping
best_avg_score = -np.inf
episodes_no_improvement = 0

for episode in range(1, number_episodes + 1):
    state, _ = env.reset()
    score = 0

    steps_in_episode = 0
    action_counts = np.zeros(number_actions, dtype=np.int64)

    # Mapa de visitas en este episodio
    episode_heatmap = np.zeros((128, 128), dtype=np.int64) if track_visits else None

    for t in range(maximum_number_timesteps_per_episode):
        # Elegir acción
        action = agent.act(state, epsilon)
        action_counts[action] += 1

        # Pasar un timestep
        next_state, reward, done, _, info = env.step(action)

        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        steps_in_episode += 1

        # Actualizar mapa de visitas del episodio
        if track_visits:
            state_tensor = preprocess_frame(state).squeeze(0).numpy()  # shape (3,128,128)
            channel_0 = state_tensor[0]
            episode_heatmap[channel_0 > 0] += 1

        if done:
            break

    # Agregar heatmap del episodio al heatmap acumulado
    if track_visits:
        accumulated_heatmap += episode_heatmap

    # Registro de métricas de cada episodio
    scores_on_100_episodes.append(score)
    all_scores.append(score)
    current_avg_score = np.mean(scores_on_100_episodes)
    avg_scores.append(current_avg_score)
    eps_history.append(epsilon)
    all_steps.append(steps_in_episode)

    # Distribución de acciones (en proporción)
    action_dist = action_counts / steps_in_episode
    action_distribution_history.append(action_dist)

    # Decaimiento de epsilon
    epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)

    print(f"\rEpisode {episode}\tScore: {score}\tAvg Score(últ. 100): {current_avg_score:.2f}", end="")

    # Cada 100 episodios, mostramos info, guardamos checkpoint y muestreamos Q-values
    if episode % 100 == 0:
        print(f"\rEpisode {episode}\tAverage Score (últ. 100): {current_avg_score:.2f}")
        torch.save(agent.local_qnetwork.state_dict(), f'./{EXPERIMENT_FOLDER}/checkpoints/checkpoint_ep{episode}.pth')

        # Muestreamos Q-values para visualización
        # Tomamos 10 estados aleatorios del entorno (manualmente) o de la memoria
        sample_size = 10
        if len(agent.memory) > sample_size:
            sample_experiences = random.sample(agent.memory, sample_size)
            sample_states = [exp[0] for exp in sample_experiences]  # exp[0] = state
            # Concatenamos en un solo tensor
            sample_states_tensor = torch.cat(sample_states).float()
            q_vals_sample = agent.sample_q_values(sample_states_tensor)
            # Guardamos en la lista para graficar la distribución de Q-values
            agent.q_values_samples.append(q_vals_sample)

    # Early Stopping manual
    if current_avg_score > (best_avg_score + EARLYSTOP_THRESHOLD):
        best_avg_score = current_avg_score
        episodes_no_improvement = 0
    else:
        episodes_no_improvement += 1

    if episodes_no_improvement >= PATIENCE:
        print(f"\nNo hubo mejora en {PATIENCE} episodios consecutivos. Deteniendo entrenamiento.")
        break

    # Si consideras un umbral para "resolver" el entorno
    if current_avg_score >= 800.0:
        print(f"\n¡Entorno solucionado en {episode} episodios! Avg Score: {current_avg_score:.2f}")
        torch.save(agent.local_qnetwork.state_dict(), f'./{EXPERIMENT_FOLDER}/checkpoints/checkpoint_solved_1.pth')
        break

# ------------------- Termina el entrenamiento. Guardar gráficas ------------------- #
# ------------------- Desarrollado por Romero Ramos, Yovany ------------------------ #

# 1) Gráfica de Score por episodio y promedio móvil de los últimos 100 episodios
plt.figure(figsize=(12, 5))
plt.plot(all_scores, label='Score por episodio')
plt.plot(avg_scores, label='Promedio móvil (100)')
plt.title('Evolución de Score en MsPacman')
plt.xlabel('Episodio')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.savefig(f'./{EXPERIMENT_FOLDER}/scores_plot.png')
plt.close()

# 2) Gráfica de pérdidas (loss) acumuladas en cada batch de aprendizaje
plt.figure(figsize=(12, 5))
plt.plot(agent.losses, label='Loss')
plt.title('Pérdida (Loss) durante el entrenamiento')
plt.xlabel('Actualizaciones de la red')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(f'./{EXPERIMENT_FOLDER}/loss_plot.png')
plt.close()

# 3) Gráfica de epsilon
plt.figure(figsize=(12, 5))
plt.plot(eps_history, label='Epsilon')
plt.title('Evolución de Epsilon')
plt.xlabel('Episodio')
plt.ylabel('Epsilon')
plt.legend()
plt.grid(True)
plt.savefig(f'./{EXPERIMENT_FOLDER}/epsilon_plot.png')
plt.close()

# 4) Gráfica de TD-error
plt.figure(figsize=(12, 5))
plt.plot(agent.td_errors, label='TD Error (q_expected - q_targets)')
plt.title('Evolución de TD-error')
plt.xlabel('Actualizaciones de la red')
plt.ylabel('TD-error')
plt.legend()
plt.grid(True)
plt.savefig(f'./{EXPERIMENT_FOLDER}/td_error_plot.png')
plt.close()

# 5) Gráfica de pasos por episodio
plt.figure(figsize=(12, 5))
plt.plot(all_steps, label='Steps por episodio')
plt.title('Cantidad de pasos por episodio')
plt.xlabel('Episodio')
plt.ylabel('Steps')
plt.legend()
plt.grid(True)
plt.savefig(f'./{EXPERIMENT_FOLDER}/steps_plot.png')
plt.close()

# 6) Distribución de acciones
#    Podemos graficar cómo varía la proporción de cada acción en el tiempo.
#    Para cada acción, sacaremos una lista con su proporción en cada episodio.
action_distribution_history = np.array(action_distribution_history)  # shape (num_episodes, num_actions)
plt.figure(figsize=(12, 5))
for action_idx in range(number_actions):
    plt.plot(action_distribution_history[:, action_idx], label=f'Accion {action_idx}')
plt.title('Distribución de acciones (proporción) por episodio')
plt.xlabel('Episodio')
plt.ylabel('Proporción de uso de la acción')
plt.legend()
plt.grid(True)
plt.savefig(f'./{EXPERIMENT_FOLDER}/action_distribution_plot.png')
plt.close()

# 7) Distribución de Q-values
#    Podemos mostrar un histograma promedio de las muestras recolectadas cada 100 episodios
#    (guardadas en agent.q_values_samples).
all_q_values = np.concatenate(agent.q_values_samples, axis=0) if len(agent.q_values_samples) > 0 else []
if len(all_q_values) > 0:
    plt.figure(figsize=(12, 5))
    plt.hist(all_q_values.flatten(), bins=50)
    plt.title('Distribución de Q-values muestreados')
    plt.xlabel('Valor Q')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.savefig(f'./{EXPERIMENT_FOLDER}/q_values_distribution.png')
    plt.close()

# 8) Heatmap global de visitas
if track_visits:
    plt.figure(figsize=(6, 5))
    plt.imshow(accumulated_heatmap, cmap='hot', interpolation='nearest')
    plt.title('Heatmap de visitas (acumulado)')
    plt.colorbar()
    plt.savefig(f'./{EXPERIMENT_FOLDER}/heatmap_visits.png')
    plt.close()

print("\nEntrenamiento finalizado. ¡Las gráficas se han guardado en archivos .png!")

# ----------------- Guardar hiper-parámetros ---------------- #
hyperparameters = {
    "env_name": env_name,
    "state_shape": state_shape,
    "number_actions": number_actions,
    "number_episodes": number_episodes,
    "maximum_number_timesteps_per_episode": maximum_number_timesteps_per_episode,
    "learning_rate": learning_rate,
    "minibatch_size": minibatch_size,
    "discount_factor": discount_factor,
    "max_memory": MAX_MEMORY,
    "patience": PATIENCE,
    "earlystop_threshold": EARLYSTOP_THRESHOLD,
    "epsilon_starting_value": epsilon_starting_value,
    "epsilon_ending_value": epsilon_ending_value,
    "epsilon_decay_value": epsilon_decay_value
    }

# Nombre del archivo donde se guardarán los hiperparámetros
file_name = "hyperparameters.txt"

# Escribir los hiperparámetros en un archivo de texto
with open(f'./{EXPERIMENT_FOLDER}/{file_name}', "w") as file:
    file.write("# Hiperparámetros\n")
    for key, value in hyperparameters.items():
        file.write(f"{key} = {value}\n")

print(f"Hiperparámetros guardados en {file_name}")