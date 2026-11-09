# Neurito 3.0: siete experimentos en el laberinto

La red base, tras 400 épocas, acertó el **91,48 %** de los movimientos etiquetados por BFS, pero llegó sola en **592 de 716 retos (82,68 %)**. Con los mismos pesos y asistencia llegó en **713 de 716 (99,58 %)**. Como en los experimentos de salto del README, acertar decisiones aisladas y completar una tarea son dos medidas distintas.

Los resultados usan la copia local del repositorio en el commit `ce156a2` («7 archivos nuevos de laberinto»). Se usaron semillas 42 para la separación y la inicialización, 2 860 ejemplos de entrenamiento, 716 de prueba y lotes de 256. La pérdida es la entropía cruzada **en prueba**; la precisión mide coincidencia con la acción concreta del profesor, incluso cuando otra acción también lleva por una ruta mínima.

Cada reto empieza en un triple (mapa, inicio, meta) reservado para prueba. Después de moverse, la red puede entrar en estados que sí aparecieron en entrenamiento. Se permite hasta `min(65, 2 × distancia BFS + 8)` intentos, como en las pruebas del proyecto. En red pura se detiene la simulación al primer choque o al volver a una casilla: con los pesos fijos se repetiría la misma decisión. La interfaz, en cambio, reinicia tras 45 intentos y puede escoger pares de entrenamiento; no se usó para medir.

## 1. Sin entrenar: ¿por qué se atasca?

En red pura solo llegó en **22 de 716** retos. Hubo **589** retos que terminaron contra una pared y **105** en un ciclo. Por ejemplo, en el mapa 2, desde (7, 7) hacia (5, 1), la red eligió derecha (27,27 %), hacia la pared en (7, 8). Las coordenadas son (fila, columna) desde cero. El choque deja intacta la entrada, por lo que el siguiente intento vuelve a elegir derecha. Si vuelve a una casilla anterior, también repite su decisión: la red no recibe el historial de visitas. Sus pesos aleatorios todavía no reflejan ni las paredes ni la dirección hacia la meta.

## 2. Entrenamiento: 100, 200 y 400 épocas

| Épocas | Pérdida | Precisión | Llegadas puras / 716 | Fallos por pared | Fallos por ciclo | Llegadas asistidas / 716 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,3940 | 27,65 % | 22 | 589 | 105 | 412 |
| 100 | 0,3675 | 84,92 % | 367 | 15 | 334 | 701 |
| 200 | 0,2421 | 90,08 % | 582 | 5 | 129 | 708 |
| 400 | 0,2228 | 91,48 % | 592 | 3 | 121 | 713 |

A las 100 épocas casi desaparecen los choques, pero el error más común pasa a ser repetir un recorrido. El mayor salto en llegadas puras ocurre entre 100 y 200 épocas (+215); entre 200 y 400 se ganan 10. A las 400 épocas la precisión de entrenamiento es 100 %, frente a 91,48 % en prueba. En las rutas puras que sí terminaron se recorrieron en promedio **6,04 pasos**, con **0,04 pasos extra** frente a su propio óptimo. El promedio excluye los fracasos y debe leerse junto con la tasa de llegada.

## 3. ¿Qué hace el modelo y qué hacen las reglas?

La **red** transforma el ID del mapa, la posición y la meta en cuatro probabilidades. El modo puro ejecuta la mayor. En el **modo asistido**, las reglas primero descartan movimientos hacia paredes y luego eligen el mejor puntaje entre los restantes: `probabilidad − 0,35 × visitas a la casilla de destino`. Una casilla visitada sigue siendo una opción. En ambos modos, el mundo impide atravesar paredes.

Con asistencia y sin entrenamiento ya se completan 412 retos: las reglas pueden explorar aun con probabilidades aleatorias. Con entrenamiento y asistencia se completan 713; solo 3 agotan el límite y los éxitos usan en promedio **0,51 pasos extra** frente a su óptimo. La asistencia evita choques y favorece salir de ciclos, mientras que el modelo aporta una preferencia aprendida hacia la meta. BFS aporta las etiquetas durante el entrenamiento; la asistencia no lo consulta para decidir.

## 4. Tamaño de la capa oculta

Se mantuvieron la tasa 0,45 y 400 épocas. El número de parámetros incluye pesos y sesgos.

| Neuronas | Parámetros | Pérdida | Precisión | Llegadas puras / 716 | Llegadas asistidas / 716 |
|---:|---:|---:|---:|---:|---:|
| 16 | 2 724 | 0,3619 | 87,15 % | 532 | 709 |
| 32 | 5 444 | 0,2940 | 89,25 % | 561 | 704 |
| 96 | 16 324 | 0,2228 | 91,48 % | 592 | 713 |
| 128 | 21 764 | 0,2515 | 90,78 % | 611 | 707 |

Con esta semilla, **128** completó más retos en red pura y **96** logró mejor precisión, menor pérdida y más llegadas asistidas. Con 16 se obtiene una red más pequeña que todavía completa 709 retos con asistencia; sola completa 532. Más neuronas no mejoraron todas las medidas.

## 5. Tasa de aprendizaje

| Tasa | Pérdida a 100 | A 200 | A 400 | Precisión a 400 | Llegadas puras / 716 | Llegadas asistidas / 716 |
|---:|---:|---:|---:|---:|---:|---:|
| 0,01 | 1,2635 | 1,0766 | 0,7592 | 73,88 % | 146 | 627 |
| 0,10 | 0,5555 | 0,4159 | 0,3095 | 87,15 % | 431 | 708 |
| 0,45 | 0,3675 | 0,2421 | 0,2228 | 91,48 % | 592 | 713 |
| 1,50 | 0,2085 | 0,2005 | 0,2062 | 93,44 % | 622 | 713 |
| 5,00 | 1,3099 | 1,4081 | 1,4015 | 30,31 % | 32 | 460 |

Con **0,01** y **0,10** la pérdida aún baja entre 200 y 400 épocas; el presupuesto favorece tasas más rápidas. **0,45** aprende de forma sostenida en los puntos medidos. Con **1,50**, a las 200 épocas ya llega sola en 622 retos, el mismo resultado de las 400 épocas, y con asistencia obtiene 714. La pérdida de prueba vuelve a subir ligeramente a las 400 épocas pese a que mejora el acierto: predecir con más confianza algunos errores puede elevar la pérdida. Con **5,00** el entrenamiento no muestra convergencia útil en estas 400 épocas: el acierto de entrenamiento cae de 38,81 % a 32,41 %. Los resultados corresponden a una semilla; los puntos medidos no prueban convergencia matemática ni que 1,50 sea la tasa óptima.

## 6. Cambiar de mapa

Misma red base a 400 épocas; cambian el mapa y los pares inicio–meta de prueba.

| Mapa | Retos reservados | Llegadas puras | Llegadas asistidas |
|---|---:|---:|---:|
| 1 | 210 | 180 (85,71 %) | 210 (100 %) |
| 2 | 251 | 197 (78,49 %) | 248 (98,80 %) |
| 3 | 255 | 215 (84,31 %) | 255 (100 %) |

El mapa 2 fue el más difícil para la red pura en este conjunto: falló en 54 de 251 retos. Con asistencia solo quedaron 3 fallos por límite. Los tres mapas estuvieron presentes en entrenamiento y la red recibe su ID; lo nuevo es la combinación de inicio y meta dentro de un mapa conocido. Para medir generalización a mapas desconocidos habría que codificar las paredes y reservar mapas completos para prueba.

## 7. Cambiar el profesor: BFS, DFS y A*

Se evaluaron dos variantes del profesor, conservando las entradas, la separación, las semillas, la arquitectura 165 → 96 → 4, la tasa 0,45 y las 400 épocas. **BFS** explora por distancia creciente; **DFS** prioriza profundizar y devuelve la primera ruta encontrada con su orden fijo; **A*** prioriza distancia recorrida más distancia Manhattan restante. Cada variante supervisa a la red con el primer paso de su ruta. También se midió qué ocurre al seguir directamente la ruta completa del planificador.

| Planificador directo | Pasos medios de la ruta | Celdas expandidas medias (incluye meta) | Rutas mínimas / 716 | Llegadas dentro del límite / 716 |
|---|---:|---:|---:|---:|
| BFS | 6,06 | 18,49 | 716 | 716 |
| DFS | 10,67 | 18,65 | 337 | 633 |
| A* | 6,06 | 10,17 | 716 | 716 |

Los tres encontraron una ruta válida para los 716 retos sin límite. A* conservó la longitud óptima y expandió aproximadamente **45 % menos celdas** que BFS en búsquedas individuales desde el inicio. Este dato no compara tiempos de entrenamiento: el profesor BFS original calcula las distancias desde una meta una sola vez y las reutiliza para todos sus inicios.

| Profesor de la red | Pérdida frente a sus etiquetas | Precisión frente a sus etiquetas | Acuerdo con BFS | Llegadas puras / 716 | Llegadas asistidas / 716 |
|---|---:|---:|---:|---:|---:|
| BFS | 0,2228 | 91,48 % | 91,48 % | 592 | 713 |
| DFS | 0,3754 | 93,44 % | 46,93 % | 99 | 304 |
| A* | 0,1977 | 92,60 % | 90,92 % | 619 | 711 |

**DFS empeoró la navegación**, pese a que la red acertó el 93,44 % de sus etiquetas. La prueba que explica el resultado: si el propio DFS calcula una ruta y la sigue completa, llega siempre sin límite; si vuelve a calcularla en cada casilla y toma solo el primer paso, llega en **139** retos y entra en ciclo en **577**. El primer paso de una búsqueda nueva puede mandar de vuelta a la casilla anterior. La red aprende esas recomendaciones locales y no conserva el plan. BFS y A* sí llegan en los 716 retos al recalcular: cada paso de una ruta mínima reduce la distancia real a la meta en uno.

**A*** mejoró las llegadas puras en **27 retos** frente a BFS, pero en modo asistido perdió dos. En rutas mínimas empatadas, ambos profesores pueden elegir primeros pasos distintos; una discrepancia con BFS no implica automáticamente un movimiento malo. En esta ejecución, A* es una alternativa útil para red pura y BFS sigue siendo una referencia sólida con asistencia.

Pasaron las cinco pruebas originales de `test_laberinto.py`.
