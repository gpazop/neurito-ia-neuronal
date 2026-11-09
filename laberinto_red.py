"""Red neuronal 165 -> 96 -> 4, entrenada con ejemplos de BFS."""
import numpy as np
from laberinto_mundo import MAPAS, libres, distancias_bfs, accion_experta

LADO = 9
ENTRADAS = len(MAPAS) + 2 * LADO * LADO


def codificar(indice, posicion, meta):
    """Codifica mapa (ID), posición y alimento como vectores one-hot."""
    x = np.zeros(ENTRADAS, dtype=np.float64)
    x[indice] = 1
    n = len(MAPAS)
    x[n + posicion[0] * LADO + posicion[1]] = 1
    x[n + LADO * LADO + meta[0] * LADO + meta[1]] = 1
    return x


def generar_datos(semilla=42, fraccion_prueba=0.2):
    """Agrupa por pares posición/meta: evita ejemplos repetidos en entrenamiento y prueba."""
    muestras = []
    for indice, mapa in enumerate(MAPAS):
        celdas = libres(mapa)
        for meta in celdas:
            dist = distancias_bfs(mapa, meta)
            for posicion in celdas:
                if posicion == meta or posicion not in dist:
                    continue
                a = accion_experta(mapa, posicion, dist)
                muestras.append((indice, posicion, meta, a))
    rng = np.random.default_rng(semilla)
    rng.shuffle(muestras)
    corte = int(len(muestras) * (1 - fraccion_prueba))
    def arrays(filas):
        X = np.stack([codificar(i, p, m) for i,p,m,a in filas])
        y = np.asarray([a for i,p,m,a in filas], dtype=np.int64)
        return X,y
    return arrays(muestras[:corte]),arrays(muestras[corte:]),muestras[corte:]


class RedLaberinto:
    def __init__(self, semilla=42, ocultas=96, tasa=0.45):
        self.rng = np.random.default_rng(semilla)
        self.ocultas = ocultas
        self.tasa = tasa
        self.reiniciar()

    def reiniciar(self):
        self.W1 = self.rng.normal(0, np.sqrt(2 / ENTRADAS), (ENTRADAS, self.ocultas))
        self.b1 = np.zeros((1, self.ocultas))
        self.W2 = self.rng.normal(0, np.sqrt(2 / self.ocultas), (self.ocultas, 4))
        self.b2 = np.zeros((1, 4))
        self.epocas = 0
        self.historial = []

    def adelante(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        e = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        p = e / np.sum(e, axis=1, keepdims=True)
        return z1, a1, p

    def entrenar_epoca(self, X, y, lote=256):
        indices = self.rng.permutation(len(X))
        total = 0.0
        for inicio in range(0, len(indices), lote):
            ids = indices[inicio:inicio+lote]
            x = X[ids]
            etiquetas = y[ids]
            z1, a1, p = self.adelante(x)
            n = len(x)
            total -= np.log(np.clip(p[np.arange(n), etiquetas], 1e-10, 1)).sum()
            dz2 = p.copy()
            dz2[np.arange(n), etiquetas] -= 1
            dz2 /= n
            dw2 = a1.T @ dz2
            db2 = dz2.sum(axis=0, keepdims=True)
            dz1 = (dz2 @ self.W2.T) * (z1 > 0)
            dw1 = x.T @ dz1
            db1 = dz1.sum(axis=0, keepdims=True)
            self.W1 -= self.tasa * dw1
            self.b1 -= self.tasa * db1
            self.W2 -= self.tasa * dw2
            self.b2 -= self.tasa * db2
        self.epocas += 1
        perdida = total / len(X)
        self.historial.append(float(perdida))
        return perdida

    def probabilidades(self, indice, posicion, meta):
        return self.adelante(codificar(indice, posicion, meta)[None, :])[2][0]

    def exactitud(self, X, y):
        return float(np.mean(np.argmax(self.adelante(X)[2], axis=1) == y))
