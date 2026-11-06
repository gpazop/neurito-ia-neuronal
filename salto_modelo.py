"""Red neuronal 3 -> 12 -> 1 para aprender a saltar, sin frameworks de IA."""
import numpy as np


class RedSalto:
    def __init__(self, semilla=42, ocultas=12, tasa=0.8):
        self.rng = np.random.default_rng(semilla)
        self.ocultas = ocultas
        self.tasa = tasa
        self.reiniciar()

    def reiniciar(self):
        self.W1 = self.rng.normal(0, 0.65, (3, self.ocultas))
        self.b1 = np.zeros((1, self.ocultas))
        self.W2 = self.rng.normal(0, 0.65, (self.ocultas, 1))
        self.b2 = np.zeros((1, 1))
        self.epocas = 0
        self.historial = []

    @staticmethod
    def preparar_entradas(distancia, velocidad):
        """Distancia, velocidad y tiempo estimado hasta el obstáculo."""
        d = np.asarray(distancia, dtype=float)
        v = np.asarray(velocidad, dtype=float)
        tiempo = np.maximum(d, 0) / np.maximum(v, 0.001)
        return np.column_stack((
            np.clip(d / 550, 0, 1),
            np.clip(v / 9, 0, 1),
            np.clip(tiempo / 60, 0, 1),
        ))

    @staticmethod
    def sigmoide(z):
        return 1 / (1 + np.exp(-np.clip(z, -40, 40)))

    def adelante(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoide(z2)
        return z1, a1, a2

    def probabilidad(self, distancia, velocidad):
        X = self.preparar_entradas([distancia], [velocidad])
        return float(self.adelante(X)[2][0, 0])

    @staticmethod
    def etiqueta(distancia, velocidad):
        """Regla que define las etiquetas supervisadas, no una recompensa."""
        ttc = np.asarray(distancia) / np.maximum(np.asarray(velocidad), 0.001)
        return ((ttc >= 7) & (ttc <= 13)).astype(float).reshape(-1, 1)

    def generar_datos(self, n=5000):
        v = self.rng.uniform(4.5, 8.5, n)
        # Mezcla de ejemplos de saltar y esperar para evitar el desbalance.
        n_pos = n // 2
        ttc = np.concatenate((
            self.rng.uniform(7.5, 12.5, n_pos),
            self.rng.uniform(0, 60, n - n_pos),
        ))
        self.rng.shuffle(ttc)
        d = v * ttc
        return self.preparar_entradas(d, v), self.etiqueta(d, v)

    def entrenar_epoca(self, X, y):
        z1, a1, a2 = self.adelante(X)
        p = np.clip(a2, 1e-8, 1 - 1e-8)
        perdida = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        # Retropropagación de la capa de salida.
        dz2 = (a2 - y) / len(X)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Retropropagación de la capa oculta (derivada de ReLU).
        dz1 = (dz2 @ self.W2.T) * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Descenso de gradiente.
        self.W2 -= self.tasa * dW2
        self.b2 -= self.tasa * db2
        self.W1 -= self.tasa * dW1
        self.b1 -= self.tasa * db1
        self.epocas += 1
        self.historial.append(float(perdida))
        return float(perdida)

    def precision(self, X, y):
        return float(np.mean((self.adelante(X)[2] >= 0.5) == y))
