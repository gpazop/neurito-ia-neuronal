# Neurito 3.0: aprende a encontrar comida en un laberinto

Neurito busca comida dentro de tres laberintos. Su cerebro debe elegir entre arriba, abajo, izquierda y derecha. La red se construye con NumPy, y Pygame dibuja el escenario, la mascota y una gráfica de la pérdida.

## Cómo aprende

1. BFS calcula rutas cortas y genera ejemplos etiquetados: mapa, posición, meta y movimiento recomendado.
2. Los ejemplos se dividen en entrenamiento (80 %) y prueba (20 %). Los pares posición-meta de prueba no aparecen en entrenamiento.
3. Cada entrada incluye un vector one-hot de tres mapas y dos vectores one-hot de 81 casillas: 165 características.
4. La arquitectura es 165 → 96 → 4. Usa ReLU en la capa oculta y Softmax en las cuatro salidas.
5. El entrenamiento minimiza entropía cruzada multiclase mediante retropropagación y descenso de gradiente con mini-lotes.
6. Durante el juego, el modelo decide sin consultar BFS. BFS es el experto que proporcionó las etiquetas.

Esto es aprendizaje supervisado por imitación, NO aprendizaje por refuerzo.

## Dos modos

- Red pura: toma la salida de mayor probabilidad. Puede chocar con paredes o quedarse en ciclos.
- Asistido: descarta movimientos contra paredes y penaliza casillas visitadas. Son reglas adicionales, no conocimientos aprendidos por la red.

La precisión de clasificación no equivale al porcentaje de laberintos completados.

## Estructura

- laberinto_mundo.py: mapas, movimiento, BFS y navegación asistida.
- laberinto_red.py: red neuronal y entrenamiento con NumPy.
- laberinto.py: juego Pygame, sprites y pérdida.
- test_laberinto.py: pruebas automáticas.
- GUIA_LABERINTO.md: esta guía.

Reutiliza assets/neutral_1.png, neutral_2.png y neutral_3.png. No modifica las versiones anteriores.

## Ejecutar

    uv sync
    uv run laberinto.py

## Controles

| Tecla | Acción |
|---|---|
| 1 | Entrenar una época |
| 2 | Entrenar cien épocas |
| 3 | Entrenamiento continuo |
| A | Alternar red pura y navegación asistida |
| P | Alternar navegación automática y paso a paso |
| Espacio | Avanzar un paso en modo manual |
| N | Cambiar mapa |
| G | Nuevas posiciones y meta |
| C | Reiniciar contadores |
| R | Reiniciar pesos y entrenamiento |
| Esc | Salir |

## Probar automáticamente

    uv run python -m unittest -v test_laberinto.py

## Experimentos

1. Observa Neurito sin entrenar. ¿Por qué se atasca?
2. Entrena 100, 200 y 400 épocas. Compara pérdida, precisión y movimientos.
3. Compara red pura frente a navegación asistida. ¿Qué parte corresponde al modelo y cuál a las reglas?
4. Cambia las 96 neuronas ocultas por 16, 32 o 128.
5. Cambia la tasa de aprendizaje 0.45 y observa si converge.
6. Cambia de mapa. Los mapas forman parte del entrenamiento, pero las combinaciones de inicio y meta usadas para probar no se entrenaron.
7. Haz vibecoding para verificar como mejora o empeora Neurito al cambiar BFS, por DFS y A*

## Alcance y limitaciones

El modelo no interpreta capturas de pantalla: trabaja con características estructuradas. Conoce el identificador de cada mapa. Los tres mapas pertenecen al mismo universo de entrenamiento. La evaluación demuestra generalización a pares de posición-meta no entrenados, no a laberintos totalmente desconocidos. Para eso habría que representar también las paredes de cualquier mapa nuevo y entrenar y evaluar en mapas separados.
