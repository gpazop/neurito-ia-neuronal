"""Pruebas reproducibles sin interfaz; solo stdlib y NumPy."""
import unittest
import numpy as np
from laberinto_mundo import MAPAS, libres, distancias_bfs, accion_experta, vecino, valido, Laberinto, elegir_accion
from laberinto_red import RedLaberinto, generar_datos


class PruebaLaberinto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ent, cls.prueba, cls.casos = generar_datos(semilla=42)
        cls.red = RedLaberinto()
        for _ in range(400):
            cls.red.entrenar_epoca(*cls.ent)

    def test_mapas_conectados_y_bfs_correcto(self):
        for mapa in MAPAS:
            celdas = libres(mapa)
            self.assertGreaterEqual(len(celdas), 25)
            for meta in celdas:
                dist = distancias_bfs(mapa, meta)
                self.assertEqual(len(dist), len(celdas))
                for p in celdas:
                    if p == meta:
                        continue
                    accion = accion_experta(mapa, p, dist)
                    self.assertTrue(valido(mapa, vecino(p, accion)))
                    self.assertEqual(dist[vecino(p, accion)], dist[p] - 1)

    def test_datos_separados_y_red_aprende(self):
        X, y = self.ent
        Xt, yt = self.prueba
        self.assertEqual(X.shape[1], 165)
        self.assertEqual(len(set(map(tuple, X)) & set(map(tuple, Xt))), 0)
        self.assertEqual(len(self.red.historial), 400)
        self.assertLess(self.red.historial[-1], self.red.historial[0])
        self.assertGreaterEqual(self.red.exactitud(Xt, yt), 0.90)

    def test_probabilidades_validas(self):
        X, _ = self.prueba
        p = self.red.adelante(X[:20])[2]
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertTrue(np.allclose(p.sum(axis=1), 1))
        self.assertTrue(np.all((p >= 0) & (p <= 1)))

    def test_asistencia_no_cruza_paredes(self):
        for indice, inicio, meta, _ in self.casos[:80]:
            mundo = Laberinto(indice)
            mundo.posicion, mundo.meta = inicio, meta
            accion, _ = elegir_accion(self.red, mundo, {inicio: 1}, True)
            self.assertTrue(valido(mundo.mapa, vecino(inicio, accion)))

    def test_exito_en_retos_no_entrenados(self):
        exitos = 0
        sin_asistir = 0
        for indice, inicio, meta, _ in self.casos:
            mapa = MAPAS[indice]
            distancia_optima = distancias_bfs(mapa, meta)[inicio]
            limite = min(65, 2 * distancia_optima + 8)
            for asistido in (False, True):
                mundo = Laberinto(indice)
                mundo.posicion, mundo.meta = inicio, meta
                visitas = {inicio: 1}
                for _ in range(limite):
                    accion, _ = elegir_accion(self.red, mundo, visitas, asistido)
                    siguiente = vecino(mundo.posicion, accion)
                    if not valido(mapa, siguiente):
                        break
                    mundo.posicion = siguiente
                    visitas[siguiente] = visitas.get(siguiente, 0) + 1
                    if siguiente == meta:
                        if asistido:
                            exitos += 1
                        else:
                            sin_asistir += 1
                        break
        self.assertGreaterEqual(sin_asistir / len(self.casos), 0.75)
        self.assertGreaterEqual(exitos / len(self.casos), 0.98)


if __name__ == '__main__':
    unittest.main()
