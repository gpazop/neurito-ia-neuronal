"""Física independiente de la red neuronal: salto y colisiones."""

import random

SUELO = 420
JUGADOR_X = 118
JUGADOR_ANCHO = 50
JUGADOR_ALTO = 64
OBSTACULO_ANCHO = 27
OBSTACULO_ALTO = 47
GRAVEDAD = 0.65
IMPULSO = -12.0


class JuegoSalto:
    def __init__(self):
        self.azar = random.Random(19)
        self.esquivados = 0
        self.choques = 0
        self.reiniciar_pista()

    def reiniciar_pista(self):
        self.y = float(SUELO - JUGADOR_ALTO)
        self.vy = 0.0
        self.en_suelo = True
        self.obstaculo_x = 830.0
        self.velocidad = 5.5
        self.terminado = False
        self.pausa_choque = 0

    def reiniciar_todo(self):
        self.esquivados = 0
        self.choques = 0
        self.reiniciar_pista()

    @property
    def distancia(self):
        return max(0.0, self.obstaculo_x - (JUGADOR_X + JUGADOR_ANCHO))

    def saltar(self):
        if self.en_suelo and not self.terminado:
            self.vy = IMPULSO
            self.en_suelo = False
            return True
        return False

    def actualizar(self):
        if self.terminado:
            self.pausa_choque += 1
            if self.pausa_choque >= 30:
                self.reiniciar_pista()
            return

        if not self.en_suelo:
            self.vy += GRAVEDAD
            self.y += self.vy
            if self.y >= SUELO - JUGADOR_ALTO:
                self.y = float(SUELO - JUGADOR_ALTO)
                self.vy = 0.0
                self.en_suelo = True

        self.obstaculo_x -= self.velocidad
        horizontal = (JUGADOR_X < self.obstaculo_x + OBSTACULO_ANCHO
                      and JUGADOR_X + JUGADOR_ANCHO > self.obstaculo_x)
        vertical = (self.y + JUGADOR_ALTO > SUELO - OBSTACULO_ALTO)
        if horizontal and vertical:
            self.choques += 1
            self.terminado = True
            return

        if self.obstaculo_x + OBSTACULO_ANCHO < JUGADOR_X:
            self.esquivados += 1
            self.obstaculo_x = 830.0
            self.velocidad = self.azar.uniform(4.5, 8.5)
