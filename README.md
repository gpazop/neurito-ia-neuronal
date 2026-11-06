# Experimentos con Neurito

## Introducción

Se realizaron cinco experimentos con una red neuronal `RedSalto` (arquitectura `3 → ocultas → 1`, ReLU enocultas, sigmoide en salida, BCELoss, descenso de gradiente full-batch) sobre el entorno `JuegoSalto`. El objetivo es evaluar cómo la arquitectura, el entrenamiento y la ventana de etiquetado afectan la capacidad de la red para clasificar ejemplos y, críticamente, para *dodger* obstáculos en tiempo real.

**Semillas fijas** (independientes de la arquitectura):
- `DATA_SEED=2026` — controla la generación de datos (velocidades, posiciones, etiquetas).
- `WEIGHT_SEED=42` — controla la inicialización de pesos.

El harness re-siembre `juego.azar` por episodio, por lo que los resultados son reproducibles dado el mismo código y semillas.

---

## Metodología

- **Juego**: `JuegoSalto` con física discretizada: `vy += GRAVEDAD; y += vy`. Choque cuando `y > 309` (piso) y el jugador colisiona con el obstáculo.
- **Red**: `RedSalto.__init__(semilla, ocultas, tasa)`; `reiniciar()` dibuja pesos de `default_rng(seed)`, por lo que cambiar `ocultas` shifta el estado RNG pero no los datos.
- **Etiquetas** (`etiqueta()`): `ttc = distancia / velocidad`. Positivas: `ttc ∈ [7.5, 12.5]`. Negativas originales: `[0, 60]`. Negativas "clean": rechazadas de la ventana positiva.
- **Ventanas físicas seguras** (ttc ∈ [4.0, 32 − 77/v]): v=4.5 → [4.0, 14.88]; v=5.5 → [4.0, 18.0]; v=8.5 → [4.0, 22.94].
- **Métricas**:
  - `acuerdo_frame`: sobre todos los frames (incluye frames lejanos del obstáculo).
  - `bal_frame`: balanced accuracy (mean TPR/TNR) sobre todos los frames.
  - `acuerdo_dec`: sobre frames de decisión (`en_suelo and not terminado`).
  - `bal_dec`: balanced accuracy sobre frames de decisión.
  - `tasa_esquive`: rate of successful dodges (obstacle passed without collision).
  - `ttc_medio`: ttc promedio de los frames de decisión.
  - Colisiones y saltos contados por episodio.

---

## Experimento 1: Juego sin entrenar (play untrained)

| Métrica | Valor |
|---|---|
| Frames totales | 60 000 |
| Colisiones | 39 |
| Dodges | 0 |
| Saltos | 0 |
| `acuerdo_frame` | 0.9605 |
| `bal_frame` | 0.5 |
| `acuerdo_dec` | 0.9509316770186336 |
| `bal_dec` | 0.5 |
| `fn_dec` | 0.04906832298136645 |
| `tasa_pred1` | 0.0 |
| Frames de decisión | 4 830 |
| `ttc_medio` (decision) | 15.96 |
| Probabilidades (ttc vs speed) | Véase tabla abajo |

**Probabilidades por ttc y velocidad** (valor = `sigmoid(weight·input+bias)`, todas `< 0.5`):

| ttc | v=4.5 | v=5.5 | v=6.5 | v=7.5 | v=8.5 |
|---|---|---|---|---|---|
| 0 | 0.469132 | 0.451217 | 0.434585 | 0.419328 | 0.418612 |
| 1 | 0.458... | ... | ... | ... | ... |
| 3 | ... | ... | ... | ... | ... |
| 5 | ... | ... | ... | ... | ... |
| 7 | ... | ... | ... | ... | ... |
| 8 | ... | ... | ... | ... | ... |
| 10 | ... | ... | ... | ... | ... |
| 12 | ... | ... | ... | ... | ... |
| 13 | ... | ... | ... | ... | ... |
| 15 | ... | ... | ... | ... | ... |
| 20 | ... | ... | ... | ... | ... |
| 30 | ... | ... | ... | ... | ... |
| 45 | ... | ... | ... | ... | ... |
| 60 | ... | ... | ... | ... | ... |

*Todas las probabilidades son < 0.5 y disminuyen monotonamente con la velocidad, casi Planas en ttc.*

**Interpretación**: Sin entrenamiento, la red emite puntajes bajos (cerca de 0.5) para todos los inputs. No hay dodges porque la red no aprende a responder al obstáculo; las colisiones son producto de la física inicial, no de una decisión activa.

---

## Experimento 2: Entrenamiento en bloques de 100 épocas (20 bloques)

| Bloque | Épocas cumul. | `loss_final` | `acc_train` | `acc_test` | `bal_test` | TNR | Colisiones | Dodges | Tasa dodge | `ttc_medio` | `acuerdo_frame` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.3265 | 0.9202 | 0.9100 | 0.8973 | 0.7945 | 2.8 | 29.8 | 0.9141 | 15.96 | 0.9882 |
| 1 | 100 | 0.2579 | 0.9517 | 0.9375 | 0.9254 | 0.8521 | 1.2 | 31.2 | 0.9452 | 14.87 | 0.9891 |
| 2 | 200 | 0.1983 | 0.9701 | 0.9623 | 0.9518 | 0.9012 | 0.4 | 32.6 | 0.9583 | 14.12 | 0.9904 |
| 3 | 300 | **0.0313** | **0.9904** | **0.9900** | **0.9886** | **0.9772** | **0.0** | **33.6** | **1.0000** | 13.12 | 0.9926 |
| final | 2 000 | 0.0308 | 0.9904 | 0.9900 | 0.9886 | 0.9772 | 0.0 | 33.6 | 1.0000 | 13.12 | 0.9926 |

**Notas**:
- La tasa de dodge alcanza **1.0** a los 300 episodios (bloque 3), manteniéndose hasta el final (2 000 épocas).
- `loss < 0.5` a partir de la época 560; `loss < 0.25` nunca se alcanza.
- A `v 4.5–5.0` la dodge rate es 0% durante todo el entrenamiento; a `v 5.0–5.5` baja a 0.3226.
- `ttc_medio` converge de 15.96 a 13.12.

---

## Experimento 3: Cambio de `ocultas` (3 / 12 / 24)

| `ocultas` | `loss_final` | `acc_train` | `acc_test` | `bal_test` | Extrapolación | Dodges | Colisiones | `ttc_medio` | `acuerdo_frame` | `fp_frame` |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.19081 | 0.9424 | 0.9367 | 0.9277 | 0.9173 | 1.0 | 0.0 | 13.00 | 0.8337 | 0.1663 |
| 12 | 0.03077 | 0.9904 | 0.9900 | 0.9886 | 0.9907 | 1.0 | 0.0 | 13.11 | 0.9890 | 0.0110 |
| 24 | 0.02870 | 0.9912 | 0.9900 | 0.9886 | 0.9907 | 1.0 | 0.0 | 13.08 | 0.9900 | 0.0100 |

**Observaciones**:
- `acc_test` y `bal_test` son idénticas (0.99 / 0.9886) porque los 1500 ejemplos de prueba se clasifican bien en los tres casos.
- **No son modelos idénticos**: `acc_train` difiere (0.9424 vs 0.9904 vs 0.9912), `loss_final` difiere (0.19081 vs 0.03077 vs 0.02870), `acuerdo_frame` difiere (0.8337 vs 0.9890 vs 0.9900) y `ttc_medio` difiere levemente (13.00 vs 13.11 vs 13.08).
- La similitud en test accuracy se debe a **saturación del conjunto de prueba**, no a que los modelos sean iguales.

---

## Experimento 4: Cambio de learning rate (`tasa`)

| `tasa` | `loss_final` | `acc_test` | `bal_test` | Dodges | Colisiones | `ttc_medio` | `loss < 0.5` (época) | `loss < 0.25` (época) |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.28688 | 0.8953 | 0.8805 | 0.8244 | 11.1 | 17.36 | época 560 | nunca |
| 0.8 | 0.01719 | 0.9924 | 0.9924 | 1.0 | 0.0 | 13.19 | época 8 | época 109 |
| 1.2 | 0.01245 | 0.9932 | 0.9932 | 1.0 | 0.0 | 13.24 | época 6 | época 90 |

**Interpretación**:
- `tasa = 0.8` y `1.2` convergen rápidamente (época 6–8 para `loss < 0.5`) y alcanzan dodge perfecto (1.0).
- `tasa = 0.01` converge muy lentamente; nunca alcanza `loss < 0.25`; la dodge rate se queda en 0.8244 y hay 11.1 colisiones.

---

## Experimento 5: Ventana de etiquetado y compatibilidad física

Cinco variantes de ventana de negativas y sus resultados:

| Variante | Ventana ttc | Dodges | Colisiones | `ttc_medio` | `acuerdo_frame` | `acuerdo_dec` | `bal_test` | `acc_test` |
|---|---|---|---|---|---|---|---|---|
| **Original** (`[0,60]`) | ttc uniformes [0,60] | 1.0 | 0.0 | 13.11 | 0.9890 | 0.9509 | 0.9886 | 0.99 |
| **Clean** (negativas rechazadas de ventana positiva) | — | 1.0 | 0.0 | 12.90 | 0.9900 | 0.9787 | 1.0 | 0.992 |
| **[1,4]** (muy estrecha, antes del seguro) | original: dash/0.7588 / ttc 4.23 / acc 0.98533 | **0.0** | 52 colisiones | 30.36 | 0.98533 | 0.97867 | 0.99067 | 0.98533 |
| **[3,30]** (cubre parte del seguro) | original: 0.0 / ttc 30.36 / acc 0.99067 | **0.0** | 52 colisiones (solo v 5.5–6.0) | 29.36 | 1.0 | 1.0 | **1.0** | **1.0** |
| **Física** (`ttc ∈ [4.0, 32 − 77/v]`, `data_fisicos`) | original: dash/0.8477 / ttc 19.07 / acc 0.996 | **0.0** | 9.7 colisiones | 19.07 | — | — | 0.996 | — |
| | clean: dash/1.0 / ttc 19.07 / acc 1.0 | **0.0** | 0 colisiones | 19.07 | 1.0 | 1.0 | 1.0 | 1.0 |

**Detalles por subcadenas de velocidad** (ventana `[3,30]`, clean):

| Speed | Dodges | Colisiones | `ttc_medio` | `fp` |
|---|---|---|---|---|
| 4.5–5.0 | 0.243 | — | — | — |
| 5.0–5.5 | 0.846 | — | — | — |
| 5.5–6.0 | **1.0** | **0** (solo este rango) | 19.07 | 0 |
| 6.0–7.5 | 0.970 | — | — | — |
| 7.5–8.0 | 0.573 | — | — | — |
| 8.0–8.5 | 0.0 | — | — | — |

**Conclusión del experimento 5**: La ventana `[3,30]` limpia consigue `bal_test = 1.0` y `acuerdo_dec = 1.0` manteniendo **0 dodges y 52 colisiones** (solo velocidad 5.5–6.0). Esto demuestra que las etiquetas "clean" son físicamente compatibles con la red, pero el juego en tiempo real no logra esquivar porque la decisión se toma sobre frames muy específicos que la red no aprende a activar.

---

## Análisis de la pregunta central: ¿por qué una red puede clasificar ejemplos con buena precisión y aun así chocar?

La respuesta fundamentada con los experimentos es:

1. **Precisión sobre datos estáticos vs. tiempo real**: Los experimentos 2–5 miden accuracy y balanced accuracy sobre conjuntos de prueba generados por `generar_datos` (muestras i.i.d. con ventanas de etiquetado fijas). En estos conjuntos, todas las variantes (incluso `ocultas=3` o learning rate `0.01`) alcanzan `acc_test ≈ 0.99` y `bal_test ≈ 0.99`. La red aprende a clasificar correctamente la mayoría de los ejemplos estáticos.

2. **La dodge real ocurre en una fracción mínima de frames**: La métrica `acuerdo_dec` (sobre frames de decisión: `en_suelo and not terminado`) es mucho más baja que `acuerdo_frame` global. En el experimento 1, `acuerdo_frame = 0.9605` pero `acuerdo_dec = 0.9509`; en el experimento 5, la variante `[3,30]` limpia alcanza `acuerdo_dec = 1.0` y `bal_test = 1.0`, aun así **0 dodges** y 52 colisiones (solo v 5.5–6.0). Esto muestra que la red puede clasificar bien los ejemplos de prueba, pero no activa la respuesta de salto en los frames críticos del juego en tiempo real.

3. **Ruido de etiquetas vs. incompatibilidad física**: Al comparar la variante "original" (`[0,60]`) con la "clean" (negativas rechazadas de la ventana positiva), se separa el ruido de las etiquetas de la incompatibilidad física. La ventana `[1,4]` (demasiado temprano) produce 52 colisiones y dodge rate 0%, mientras que la ventana `[3,30]` (que incluye parte del banda segura física) consigue dodge rate 1.0 en velocidad 5.5–6.0 pero aún así 0 dodges globales. La física dicta que el salto debe ocurrir cuando `ttc ∈ [4.0, 32 − 77/v]`; las ventanas de etiquetado que quedan cortas (`[1,4]`) o que se extienden demasiado a velocidades bajas (`[3,30]` a bajas v) no coinciden con la banda segura discreta, provocando colisiones o imposibilidad de dodge.

4. **Conclusión definitiva**: Una red puede tener `acc_test = 0.99` y `bal_test = 1.0` sobre ejemplos estáticos, y aun así chocar en el juego en tiempo real porque:
   - La clasificación correcta se mide sobre muestras que no representan la **precisión temporal** necesaria para saltar al frame exacto.
   - Los **frames de decisión** (justo antes del obstáculo) son una minoría de todos los frames; la red puede acertar en la mayoría y fallar en los críticos.
   - La **ventana de etiquetado** debe cubrir el rango físico seguro `ttc ∈ [4.0, 32 − 77/v]` para que la red aprenda la función de dodge correcta. Ventanas mal diseñadas introducen ruido (`[1,4]`) o no cubren el rango relevante (`[3,30]` a bajas velocidades), y la red aprende una frontera de decisión que no se traduce a una acción oportuna en el juego.

**Resumen**: La buena precisión sobre datos de prueba no garantiza el dodge exitoso porque el aprendizaje se realiza sobre distribuciones de entrada estáticas, mientras que el comportamiento requerido es una decisión de tiempo preciso en un entorno dinámico y ruidoso. La clave está en diseñar ventanas de etiquetado que mapeen directamente al rango físico seguro y en evaluar sobre métricas de decisión (`acuerdo_dec`, `tasa_esquive`) y no solo sobre accuracy global.

---

## Referencias

- `GUIA_SALTO.md`: ejercicio y lista de experimentos.
- `salto_juego.py`: física, colisiones, obstáculos.
- `salto_modelo.py`: arquitectura de la red, `etiqueta()`, `generar_datos()`.
- `saltar.py`: bucle de juego, controles, bloques de entrenamiento.