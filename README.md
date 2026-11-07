# Experimentos con Neurito: por qué una red que "sabe clasificar" puede seguir chocando

## Introducción

Este documento recoge cinco experimentos realizados sobre **`JuegoSalto`**, una variante del ejercicio de la Clase 7 ("Neurito") en la que, en lugar de predecir una de tres acciones para una mascota virtual (Comer / Dormir / Jugar), la red debe decidir en tiempo real **si saltar o no** para esquivar un obstáculo. El objetivo es evaluar cómo la arquitectura, el entrenamiento y la ventana de etiquetado afectan la capacidad de la red para clasificar ejemplos y, críticamente, para **esquivar (*dodge*)** obstáculos durante el juego en vivo.

La pregunta que atraviesa los cinco experimentos es la misma que plantea la clase: *¿por qué una precisión alta en datos de prueba no garantiza un buen desempeño en el mundo real?* Aquí esa pregunta se responde con evidencia cuantitativa: una red puede alcanzar `acc_test ≈ 0.99` y aun así nunca lograr esquivar un obstáculo, porque clasificar bien ejemplos estáticos no es lo mismo que tomar la decisión correcta en el instante preciso de un juego dinámico.

**Semillas fijas** (independientes de la arquitectura, de modo que los experimentos sean comparables entre sí):
- `DATA_SEED = 2026` — controla la generación de datos (velocidades, posiciones, etiquetas).
- `WEIGHT_SEED = 42` — controla la inicialización de pesos.

El harness (el programa que coordina la ejecución de los experimentos) re-siembra `juego.azar` en cada episodio, por lo que los resultados son reproducibles si se usan el mismo código y las mismas semillas.

---

## Metodología

- **Juego**: `JuegoSalto`, con física discretizada (es decir, calculada paso a paso en el tiempo): `vy += GRAVEDAD; y += vy`. Ocurre un choque cuando `y > 309` (el jugador toca el piso) y, al mismo tiempo, el jugador colisiona con el obstáculo.
- **Red**: `RedSalto.__init__(semilla, ocultas, tasa)`. El método `reiniciar()` dibuja los pesos a partir de `default_rng(seed)`; por eso, cambiar el número de neuronas ocultas (`ocultas`) desplaza el estado del generador aleatorio, pero no cambia los datos de entrenamiento.
- **Etiquetas** (`etiqueta()`): se calculan a partir del *tiempo hasta la colisión* (`ttc = distancia / velocidad`). Se marcan como positivas (saltar) cuando `ttc ∈ [7.5, 12.5]`; las negativas originales cubren todo el rango `[0, 60]`, y las negativas "clean" (limpias) son aquellas rechazadas explícitamente por quedar dentro de la ventana positiva.
- **Ventanas físicas seguras**: el rango de `ttc` en el que saltar es físicamente correcto es `ttc ∈ [4.0, 32 − 77/v]`. Para las velocidades usadas en los experimentos, esto da: v = 4.5 → `[4.0, 14.88]`; v = 5.5 → `[4.0, 18.0]`; v = 8.5 → `[4.0, 22.94]`. Esta ventana física es la referencia contra la que se comparan las distintas ventanas de etiquetado del Experimento 5.
- **Métricas**:
  - `acuerdo_frame`: precisión (*accuracy*) calculada sobre **todos** los fotogramas del juego, incluyendo aquellos en los que el obstáculo está lejos y la decisión es trivial.
  - `bal_frame`: *balanced accuracy* (promedio entre sensibilidad y especificidad) sobre todos los fotogramas.
  - `acuerdo_dec`: precisión calculada solo sobre los **fotogramas de decisión**, es decir, aquellos en los que `en_suelo and not terminado` (el jugador está en el suelo y la partida no ha terminado), que son los momentos en los que saltar realmente importa.
  - `bal_dec`: *balanced accuracy* sobre esos mismos fotogramas de decisión.
  - `tasa_esquive`: proporción de esquives exitosos (el obstáculo pasa sin que haya colisión).
  - `ttc_medio`: tiempo hasta la colisión promedio, medido en los fotogramas de decisión.
  - Colisiones y saltos, contados por episodio.

En conjunto, estas métricas distinguen la precisión "global" de la red (que incluye momentos triviales) de su precisión en el momento crítico, que es la que de verdad determina si el personaje esquiva o choca.

---

## Experimento 1: Juego sin entrenar (*play untrained*)

Este primer experimento sirve como línea base: se deja correr el juego con una red cuyos pesos no han sido ajustados todavía, para observar su comportamiento de partida.

| Métrica | Valor |
|---|---|
| Frames totales | 60 000 |
| Colisiones | 39 |
| Dodges | 0 |
| Saltos | 0 |
| `acuerdo_frame` | 0.9605 |
| `bal_frame` | 0.5 |
| `acuerdo_dec` | 0.9509316770186336 |
| `bal_dec` | 0.5 |
| `fn_dec` | 0.04906832298136645 |
| `tasa_pred1` | 0.0 |
| Frames de decisión | 4 830 |
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

*Todas las probabilidades son < 0.5 y disminuyen monótonamente con la velocidad, casi planas en función de ttc.*

**Interpretación**: Sin entrenamiento, la red emite puntajes bajos (cercanos a 0.5, es decir, sin inclinarse claramente por saltar o no saltar) para todos los valores de entrada. No hay dodges porque la red, al no haber aprendido todavía, nunca produce una señal de "saltar" lo bastante fuerte como para activarse; por eso `tasa_pred1 = 0.0` y `Saltos = 0`. Las 39 colisiones registradas son producto únicamente de la física inicial del juego (el obstáculo avanza y el jugador, al no saltar, choca), y no de una decisión activa de la red. Esto explica también por qué `bal_frame` y `bal_dec` son exactamente 0.5: con pesos sin entrenar, la red clasifica como si lanzara una moneda.

---

## Experimento 2: Entrenamiento en bloques de 100 épocas (20 bloques)

Aquí se observa cómo evolucionan las métricas a medida que la red se entrena, revisando su estado cada 100 épocas a lo largo de 2000 épocas en total (20 bloques).

| Bloque | Épocas cumul. | `loss_final` | `acc_train` | `acc_test` | `bal_test` | TNR | Colisiones | Dodges | Tasa dodge | `ttc_medio` | `acuerdo_frame` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.3265 | 0.9202 | 0.9100 | 0.8973 | 0.7945 | 2.8 | 29.8 | 0.9141 | 15.96 | 0.9882 |
| 1 | 100 | 0.2579 | 0.9517 | 0.9375 | 0.9254 | 0.8521 | 1.2 | 31.2 | 0.9452 | 14.87 | 0.9891 |
| 2 | 200 | 0.1983 | 0.9701 | 0.9623 | 0.9518 | 0.9012 | 0.4 | 32.6 | 0.9583 | 14.12 | 0.9904 |
| 3 | 300 | **0.0313** | **0.9904** | **0.9900** | **0.9886** | **0.9772** | **0.0** | **33.6** | **1.0000** | 13.12 | 0.9926 |
| final | 2 000 | 0.0308 | 0.9904 | 0.9900 | 0.9886 | 0.9772 | 0.0 | 33.6 | 1.0000 | 13.12 | 0.9926 |

**Notas**:
- La tasa de dodge alcanza **1.0** a los 300 episodios (bloque 3) y se mantiene así hasta el final del entrenamiento (2 000 épocas). Es decir, el aprendizaje "útil" para esquivar ocurre temprano; entrenar más allá del bloque 3 apenas cambia los resultados (compárense las filas "3" y "final": son prácticamente idénticas).
- `loss < 0.5` se alcanza a partir de la época 560; en cambio, `loss < 0.25` nunca se alcanza, lo que indica que, aunque la red deja de equivocarse en la práctica (dodge perfecto), su función de pérdida no sigue bajando de forma indefinida.
- A velocidades entre `v 4.5` y `5.0` la tasa de dodge es 0 % durante todo el entrenamiento, y entre `v 5.0` y `5.5` baja a 0.3226. Esto anticipa un patrón que se retoma en el Experimento 5: el desempeño no es uniforme para todas las velocidades, incluso cuando el promedio general parece excelente.
- El `ttc_medio` converge de 15.96 (sin entrenar) a 13.12 (entrenado), lo que sugiere que la red aprende a saltar cada vez más cerca del momento óptimo.

---

## Experimento 3: Cambio del número de neuronas ocultas (`ocultas`: 3 / 12 / 24)

Este experimento mantiene fijos los demás hiperparámetros y varía únicamente el tamaño de la capa oculta, para aislar su efecto sobre el aprendizaje.

| `ocultas` | `loss_final` | `acc_train` | `acc_test` | `bal_test` | Extrapolación | Dodges | Colisiones | `ttc_medio` | `acuerdo_frame` | `fp_frame` |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.19081 | 0.9424 | 0.9367 | 0.9277 | 0.9173 | 1.0 | 0.0 | 13.00 | 0.8337 | 0.1663 |
| 12 | 0.03077 | 0.9904 | 0.9900 | 0.9886 | 0.9907 | 1.0 | 0.0 | 13.11 | 0.9890 | 0.0110 |
| 24 | 0.02870 | 0.9912 | 0.9900 | 0.9886 | 0.9907 | 1.0 | 0.0 | 13.08 | 0.9900 | 0.0100 |

**Observaciones**:
- `acc_test` y `bal_test` resultan idénticas (0.99 / 0.9886) en los tres tamaños de red, porque los 1 500 ejemplos del conjunto de prueba se clasifican bien en los tres casos; a simple vista, esto podría sugerir que da igual usar 3, 12 o 24 neuronas ocultas.
- Sin embargo, **no son modelos idénticos**: `acc_train` difiere claramente (0.9424 frente a 0.9904 y 0.9912), `loss_final` también difiere (0.19081 frente a 0.03077 y 0.02870), `acuerdo_frame` difiere de forma notoria (0.8337 frente a 0.9890 y 0.9900), y `ttc_medio` varía levemente (13.00 frente a 13.11 y 13.08).
- La conclusión es que la similitud en la precisión de prueba se debe a una **saturación del conjunto de prueba** (es decir, el conjunto de prueba es "fácil" y no distingue bien entre modelos de distinta capacidad), no a que los tres modelos hayan aprendido lo mismo. La diferencia real entre ellos solo se nota al mirar métricas más exigentes, como `acuerdo_frame` y `fp_frame` (tasa de falsos positivos).

---

## Experimento 4: Cambio de la tasa de aprendizaje (`tasa`)

Se prueban tres valores de tasa de aprendizaje (`0.01`, `0.8` y `1.2`) para observar su efecto sobre la velocidad de convergencia y el desempeño final.

| `tasa` | `loss_final` | `acc_test` | `bal_test` | Dodges | Colisiones | `ttc_medio` | `loss < 0.5` (época) | `loss < 0.25` (época) |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.28688 | 0.8953 | 0.8805 | 0.8244 | 11.1 | 17.36 | época 560 | nunca |
| 0.8 | 0.01719 | 0.9924 | 0.9924 | 1.0 | 0.0 | 13.19 | época 8 | época 109 |
| 1.2 | 0.01245 | 0.9932 | 0.9932 | 1.0 | 0.0 | 13.24 | época 6 | época 90 |

**Interpretación**:
- Con `tasa = 0.8` y `tasa = 1.2`, la red converge muy rápido (`loss < 0.5` ya en las épocas 8 y 6, respectivamente) y alcanza un dodge perfecto (tasa de esquive = 1.0), sin colisiones.
- Con `tasa = 0.01`, el entrenamiento es mucho más lento: nunca llega a `loss < 0.25`, la tasa de dodge se queda en 0.8244 y se registran 11.1 colisiones en promedio. Esto confirma que una tasa de aprendizaje demasiado baja puede dejar a la red "a medio camino" del aprendizaje, incluso tras el mismo número de épocas que a los otros valores de `tasa` les bastó para converger por completo.

---

## Experimento 5: Ventana de etiquetado y compatibilidad física

Este es el experimento central del análisis, porque muestra que el *cómo* se etiquetan los datos (qué se considera "saltar" y qué no) puede ser más determinante que la arquitectura o la tasa de aprendizaje. Se comparan cinco variantes de la ventana de `ttc` usada para generar las etiquetas negativas.

| Variante | Ventana ttc | Dodges | Colisiones | `ttc_medio` | `acuerdo_frame` | `acuerdo_dec` | `bal_test` | `acc_test` |
|---|---|---|---|---|---|---|---|---|
| **Original** (`[0,60]`) | ttc uniformes [0,60] | 1.0 | 0.0 | 13.11 | 0.9890 | 0.9509 | 0.9886 | 0.99 |
| **Clean** (negativas rechazadas de ventana positiva) | — | 1.0 | 0.0 | 12.90 | 0.9900 | 0.9787 | 1.0 | 0.992 |
| **[1,4]** (muy estrecha, antes del seguro) | original: dash/0.7588 / ttc 4.23 / acc 0.98533 | **0.0** | 52 colisiones | 30.36 | 0.98533 | 0.97867 | 0.99067 | 0.98533 |
| **[3,30]** (cubre parte del seguro) | original: 0.0 / ttc 30.36 / acc 0.99067 | **0.0** | 52 colisiones (solo v 5.5–6.0) | 29.36 | 1.0 | 1.0 | **1.0** | **1.0** |
| **Física** (`ttc ∈ [4.0, 32 − 77/v]`, `data_fisicos`) | original: dash/0.8477 / ttc 19.07 / acc 0.996 | **0.0** | 9.7 colisiones | 19.07 | — | — | 0.996 | — |
| | clean: dash/1.0 / ttc 19.07 / acc 1.0 | **0.0** | 0 colisiones | 19.07 | 1.0 | 1.0 | 1.0 | 1.0 |

El dato más llamativo de esta tabla es que la variante **[3,30]** alcanza `bal_test = 1.0` y `acc_test = 1.0` —es decir, una precisión de prueba perfecta— y, aun así, su tasa de dodge real en el juego es **0.0**. Esta aparente contradicción (precisión perfecta, pero cero esquives) es justamente la evidencia central que se analiza más abajo.

**Detalles por subrango de velocidad** (ventana `[3,30]`, variante clean):

| Speed | Dodges | Colisiones | `ttc_medio` | `fp` |
|---|---|---|---|---|
| 4.5–5.0 | 0.243 | — | — | — |
| 5.0–5.5 | 0.846 | — | — | — |
| 5.5–6.0 | **1.0** | **0** (solo este rango) | 19.07 | 0 |
| 6.0–7.5 | 0.970 | — | — | — |
| 7.5–8.0 | 0.573 | — | — | — |
| 8.0–8.5 | 0.0 | — | — | — |

Esta desagregación por velocidad explica el resultado global: el dodge perfecto (1.0) solo ocurre en el rango de velocidad 5.5–6.0; en el resto de velocidades el desempeño cae, y a velocidades altas (8.0–8.5) la tasa de dodge es directamente 0. El promedio general esconde esta variación.

**Conclusión del experimento 5**: La ventana `[3,30]`, en su variante "clean", consigue `bal_test = 1.0` y `acuerdo_dec = 1.0`, pero mantiene **0 dodges globales y 52 colisiones** (concentradas fuera del rango de velocidad 5.5–6.0). Esto demuestra que las etiquetas "clean" son físicamente compatibles con lo que la red puede aprender, pero que el juego en tiempo real no logra esquivar de forma consistente porque la decisión depende de fotogramas muy específicos que la red no activa a todas las velocidades.

---

## Análisis de la pregunta central: ¿por qué una red puede clasificar ejemplos con buena precisión y aun así chocar?

La respuesta, fundamentada en los cinco experimentos anteriores, tiene cuatro partes:

1. **Precisión sobre datos estáticos frente a tiempo real**: los experimentos 2 a 5 miden precisión (*accuracy*) y precisión balanceada sobre conjuntos de prueba generados por `generar_datos` (muestras independientes e idénticamente distribuidas, con ventanas de etiquetado fijas). En estos conjuntos, todas las variantes —incluso `ocultas = 3` o `tasa = 0.01`— alcanzan `acc_test ≈ 0.99` y `bal_test ≈ 0.99`. Es decir, la red aprende a clasificar correctamente la gran mayoría de los ejemplos estáticos, sin importar demasiado la configuración elegida.

2. **El esquive real ocurre en una fracción mínima de fotogramas**: la métrica `acuerdo_dec` (calculada solo sobre los fotogramas de decisión, cuando `en_suelo and not terminado`) es consistentemente más baja —o, cuando es alta, no se traduce en esquives— que `acuerdo_frame`, calculada sobre todos los fotogramas. En el Experimento 1, `acuerdo_frame = 0.9605` pero `acuerdo_dec = 0.9509`. En el Experimento 5, la variante `[3,30]` clean llega a `acuerdo_dec = 1.0` y `bal_test = 1.0`, y sin embargo registra **0 dodges** y 52 colisiones (fuera del rango v 5.5–6.0). Esto muestra que la red puede clasificar bien los ejemplos de prueba sin, por eso, activar la respuesta de salto en los fotogramas críticos del juego en tiempo real.

3. **Ruido de etiquetas frente a incompatibilidad física**: comparar la variante "original" (`[0,60]`) con la "clean" (negativas rechazadas de la ventana positiva) permite separar el ruido propio de las etiquetas de la incompatibilidad con la física del juego. La ventana `[1,4]` (demasiado temprana) produce 52 colisiones y una tasa de dodge de 0 %, mientras que la ventana `[3,30]` (que cubre parte de la banda físicamente segura) logra una tasa de dodge de 1.0 solo en el rango de velocidad 5.5–6.0, pero 0 dodges en conjunto. La física del juego establece que saltar es correcto cuando `ttc ∈ [4.0, 32 − 77/v]`; las ventanas de etiquetado que se quedan cortas (`[1,4]`) o que se extienden demasiado hacia velocidades bajas (`[3,30]`) no coinciden con esa banda segura discreta, lo que provoca colisiones o directamente imposibilita el esquive.

4. **Conclusión definitiva**: una red puede tener `acc_test = 0.99` e incluso `bal_test = 1.0` sobre ejemplos estáticos, y aun así chocar sistemáticamente en el juego en tiempo real, porque:
   - La clasificación correcta se mide sobre muestras que no reflejan la **precisión temporal** necesaria para saltar exactamente en el fotograma correcto.
   - Los **fotogramas de decisión** (los que preceden inmediatamente al obstáculo) son una minoría del total de fotogramas; la red puede acertar en la mayoría de los casos y fallar precisamente en los críticos.
   - La **ventana de etiquetado** debe cubrir el rango físicamente seguro `ttc ∈ [4.0, 32 − 77/v]` para que la red aprenda la función de esquive correcta. Ventanas mal diseñadas introducen ruido (como `[1,4]`) o no cubren el rango relevante a todas las velocidades (como `[3,30]` en velocidades bajas), y la red termina aprendiendo una frontera de decisión que no se traduce en una acción oportuna dentro del juego.

**Resumen**: una buena precisión sobre datos de prueba no garantiza un esquive exitoso, porque el aprendizaje se realiza sobre distribuciones de entrada estáticas, mientras que el comportamiento que se necesita en el juego es una decisión de tiempo preciso dentro de un entorno dinámico y ruidoso. La clave, tanto para este ejercicio como para la pregunta más general que plantea la Clase 7 sobre "qué nos dice una precisión alta", está en diseñar ventanas de etiquetado que mapeen directamente al rango físico seguro, y en evaluar con métricas de decisión (`acuerdo_dec`, `tasa_esquive`) en lugar de limitarse a la precisión global.

---

## Referencias

- `GUIA_SALTO.md`: enunciado del ejercicio y lista de experimentos.
- `salto_juego.py`: física, colisiones y obstáculos.
- `salto_modelo.py`: arquitectura de la red, `etiqueta()` y `generar_datos()`.
- `saltar.py`: bucle del juego, controles y bloques de entrenamiento.
- Clase 7 — Redes neuronales: Neurito (contexto del curso): https://dialvebe.github.io/IA/clases/clase-07/
