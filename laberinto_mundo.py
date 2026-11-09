"""Mapas, búsqueda BFS experta y simulación independiente de Pygame."""
from collections import deque
from dataclasses import dataclass
import random

# (fila, columna), orden fijo de desempate
DIRECCIONES = ((-1, 0), (1, 0), (0, -1), (0, 1))
NOMBRES = ('ARRIBA', 'ABAJO', 'IZQUIERDA', 'DERECHA')
MAPAS = (
    (
        '#########',
        '#...#...#',
        '#.#.#.#.#',
        '#.#...#.#',
        '#.###.#.#',
        '#...#...#',
        '###.#.###',
        '#.......#',
        '#########',
    ),
    (
        '#########',
        '#...#...#',
        '#.#...#.#',
        '#...#...#',
        '###.#.###',
        '#...#...#',
        '#.#...#.#',
        '#...#...#',
        '#########',
    ),
    (
        '#########',
        '#.......#',
        '#.###.#.#',
        '#...#.#.#',
        '###.#...#',
        '#...###.#',
        '#.#.....#',
        '#.......#',
        '#########',
    ),
)


def libres(mapa):
    return [(f, c) for f, fila in enumerate(mapa) for c, v in enumerate(fila) if v == '.']


def vecino(pos, accion):
    df, dc = DIRECCIONES[accion]
    return pos[0] + df, pos[1] + dc


def valido(mapa, pos):
    f, c = pos
    return 0 <= f < len(mapa) and 0 <= c < len(mapa[0]) and mapa[f][c] != '#'


def distancias_bfs(mapa, meta):
    """Distancia real por celdas hasta la comida, calculada SOLO por el profesor BFS."""
    dist = {meta: 0}
    cola = deque([meta])
    while cola:
        actual = cola.popleft()
        for a in range(4):
            siguiente = vecino(actual, a)
            if valido(mapa, siguiente) and siguiente not in dist:
                dist[siguiente] = dist[actual] + 1
                cola.append(siguiente)
    return dist


def accion_experta(mapa, origen, dist):
    """BFS indica el siguiente movimiento que reduce la distancia real a la meta."""
    if origen not in dist or dist[origen] == 0:
        return None
    opciones = [(dist.get(vecino(origen, a), 10**9), a) for a in range(4)]
    minimo, accion = min(opciones)
    assert minimo < dist[origen]
    return accion


@dataclass
class Laberinto:
    indice: int = 0
    semilla: int = 99

    def __post_init__(self):
        self.azar = random.Random(self.semilla)
        self.exitos = 0
        self.pasos = 0
        self.choques = 0
        self.reiniciar(self.indice)

    def reiniciar(self, indice=None):
        if indice is not None:
            self.indice = indice % len(MAPAS)
        self.mapa = MAPAS[self.indice]
        celdas = libres(self.mapa)
        self.meta = self.azar.choice(celdas)
        opciones = [p for p in celdas if p != self.meta and p in distancias_bfs(self.mapa, self.meta)]
        self.posicion = self.azar.choice(opciones)
        self.visitadas = {self.posicion: 1}
        self.pasos = 0
        self.intentos = 0
        self.terminado = False
        self.ultimo = None

    def mover(self, accion):
        self.ultimo = accion
        self.intentos += 1
        nuevo = vecino(self.posicion, accion)
        if not valido(self.mapa, nuevo):
            self.choques += 1
            return 'pared'
        self.posicion = nuevo
        self.pasos += 1
        self.visitadas[nuevo] = self.visitadas.get(nuevo, 0) + 1
        if nuevo == self.meta:
            self.exitos += 1
            self.terminado = True
            return 'comida'
        return 'avanza'


def elegir_accion(red, mundo, visitas, asistido):
    """Argmax puro; asistencia opcional para paredes y visitas repetidas."""
    probs = red.probabilidades(mundo.indice, mundo.posicion, mundo.meta)
    if not asistido:
        return int(probs.argmax()), probs
    opciones = [
        (float(probs[a]) - 0.35 * visitas.get(vecino(mundo.posicion, a), 0), -a, a)
        for a in range(4) if valido(mundo.mapa, vecino(mundo.posicion, a))
    ]
    return max(opciones)[2], probs
