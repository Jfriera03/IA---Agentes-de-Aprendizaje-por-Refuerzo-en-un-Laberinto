# 🔍 Agentes de Aprendizaje por Refuerzo en un Laberinto

## 🤝 Colaboradores

- [Jfriera03](https://github.com/Jfriera03)
- [MasoalM](https://github.com/MasoalM)

## 📌 Descripción

Este proyecto implementa agentes inteligentes basados en **aprendizaje por refuerzo** para resolver un laberinto. Se desarrollan estrategias mediante **Q-Learning** y **SARSA**, optimizando el comportamiento de los agentes en un entorno dinámico.

## 🛠️ Tecnologías y Herramientas

- **Lenguaje:** Python 🐍
- **Librerías utilizadas:** `numpy`, `random`, `logging`, `pygame`.
- **Estructuras de datos empleadas:**
  - Diccionarios para almacenar valores Q.
  - Tablas de valores para las estrategias de refuerzo.
  - Algoritmos de actualización de política (ε-greedy, aprendizaje temporal).

## 📌 Arquitectura del Código

El código está estructurado en los siguientes archivos:

- **`__main__.py`**: Punto de entrada del programa, inicializa el entorno y entrena a los agentes.
- **`agent.py`**: Implementación base de un agente con aprendizaje por refuerzo.
- **`agentSARSA.py`**: Implementación del agente usando el algoritmo **SARSA**.
- **`abstractmodel.py`**: Clase base abstracta para modelos de predicción y entrenamiento.
- **`joc.py`**: Define la estructura del laberinto y las reglas del juego.

## 🚀 Instalación y Uso

1. Clona el repositorio:
   ```bash
   git clone https://github.com/Jfriera03/agentes-refuerzo.git
   ```
2. Accede al directorio del proyecto:
   ```bash
   cd agentes-refuerzo
   ```
3. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```
4. Ejecuta el código principal:
   ```bash
   python __main__.py
   ```

## 🎮 Funcionamiento del Algoritmo

El entorno es un tablero de **8x8** donde un agente debe encontrar el camino óptimo hasta la meta evitando obstáculos. Las acciones disponibles son:

- **MOVE UP** → Moverse hacia arriba.
- **MOVE DOWN** → Moverse hacia abajo.
- **MOVE LEFT** → Moverse hacia la izquierda.
- **MOVE RIGHT** → Moverse hacia la derecha.

El agente aprende a través de ensayo y error, actualizando su estrategia con cada episodio de entrenamiento.

### 🔹 Q-Learning
- Aprende mediante la ecuación de Bellman.
- Utiliza **exploración-explotación** para mejorar decisiones.
- Almacena valores en una **tabla Q** para cada estado y acción.

### 🔹 SARSA
- Similar a Q-Learning, pero actualiza valores considerando la acción siguiente.
- Permite estrategias más conservadoras en entornos dinámicos.
- Utiliza **ε-greedy** para ajustar la exploración.

## 🏗️ Implementación Técnica

### 📌 Estados y Percepción
Cada agente recibe información del entorno a través de un diccionario:
- **`POS`**: Posición actual del agente en el laberinto.
- **`MAZE`**: Matriz de celdas con obstáculos y caminos libres.
- **`EXIT`**: Coordenadas de la casilla objetivo.

### 📌 Clases Principales

- **`AbstractModel`**: Clase base para modelos de aprendizaje.
- **`AgentQ`**: Implementación de **Q-Learning**.
- **`AgentSARSA`**: Implementación de **SARSA**.
- **`Laberint`**: Controla el entorno y aplica reglas del juego.

### 📌 Recompensas y Penalizaciones
El entrenamiento se basa en un sistema de recompensas:
- **Recompensa positiva** (+10) al alcanzar el objetivo.
- **Penalización** (-0.05) por cada movimiento sin éxito.
- **Penalización alta** (-0.75) por intentar moverse a una pared.

Los algoritmos ajustan sus decisiones para maximizar la recompensa acumulada.
