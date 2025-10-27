# scripts/pso_selector.py
# PSO binario para selección de características:
# - Posición: máscara binaria (0/1) de features
# - Velocidad: real; actualiza con inercia + atracción (pbest, gbest)
# - Binarización: σ(v) = 1/(1+e^-v); bit=1 si U(0,1) < σ(v)

from typing import List, Tuple, Callable
import math, random

class PSOSelector:
    def __init__(self,
                 swarm_size: int = 20,
                 iters: int = 50,
                 w: float = 0.7,          # inercia
                 c1: float = 1.5,         # componente cognitiva (pbest)
                 c2: float = 1.5,         # componente social (gbest)
                 vmax: float = 4.0,       # límite de velocidad (magnitud)
                 seed: int = 13):
        self.swarm_size = swarm_size
        self.iters = iters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.vmax = abs(vmax)
        self.rng = random.Random(seed)

        # Estado
        self.pos: List[List[int]] = []      # posiciones (máscaras)
        self.vel: List[List[float]] = []    # velocidades
        self.fit: List[float] = []          # fitness actual

        self.pbest_pos: List[List[int]] = []
        self.pbest_fit: List[float] = []
        self.gbest_pos: List[int] = []
        self.gbest_fit: float = float("-inf")

    # ---------- utils ----------
    def _rand_mask(self, n: int, p: float = 0.3) -> List[int]:
        m = [1 if self.rng.random() < p else 0 for _ in range(n)]
        if sum(m) == 0:
            m[self.rng.randrange(n)] = 1
        return m

    def _rand_vel(self, n: int, scale: float = 1.0) -> List[float]:
        # velocidad inicial pequeña para no saturar σ
        return [self.rng.uniform(-scale, scale) for _ in range(n)]

    @staticmethod
    def _sigmoid(x: float) -> float:
        # estable numéricamente
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)

    def _clip(self, v: float) -> float:
        return max(-self.vmax, min(self.vmax, v))

    # ---------- core ----------
    def fit_select(self,
                   eval_fn: Callable,
                   train_df,
                   test_df,
                   feature_names: List[str],
                   lam: float = 0.01,
                   max_depth: int = 12) -> Tuple[List[int], dict]:
        n = len(feature_names)
        # init
        self.pos = [self._rand_mask(n, p=0.3) for _ in range(self.swarm_size)]
        self.vel = [self._rand_vel(n, scale=1.0) for _ in range(self.swarm_size)]
        self.fit = []
        for i in range(self.swarm_size):
            f, _, _ = eval_fn(train_df, test_df, feature_names, self.pos[i],
                              max_depth=max_depth, lam=lam)
            self.fit.append(f)

        self.pbest_pos = [p[:] for p in self.pos]
        self.pbest_fit = self.fit[:]
        best_idx = max(range(self.swarm_size), key=lambda i: self.fit[i])
        self.gbest_pos = self.pos[best_idx][:]
        self.gbest_fit = self.fit[best_idx]

        history = [{"iter": 0, "best_obj": self.gbest_fit, "best_k": sum(self.gbest_pos)}]

        # loop
        for it in range(1, self.iters + 1):
            for i in range(self.swarm_size):
                # actualizar velocidad y posición (binaria vía σ)
                new_pos = self.pos[i][:]
                for d in range(n):
                    r1 = self.rng.random()
                    r2 = self.rng.random()
                    cognitive = self.c1 * r1 * (self.pbest_pos[i][d] - self.pos[i][d])
                    social    = self.c2 * r2 * (self.gbest_pos[d]      - self.pos[i][d])
                    self.vel[i][d] = self._clip(self.w * self.vel[i][d] + cognitive + social)

                    prob_one = self._sigmoid(self.vel[i][d])
                    new_pos[d] = 1 if self.rng.random() < prob_one else 0

                # evitar máscara vacía
                if sum(new_pos) == 0:
                    new_pos[self.rng.randrange(n)] = 1

                # evaluar
                f_new, _, _ = eval_fn(train_df, test_df, feature_names, new_pos,
                                      max_depth=max_depth, lam=lam)

                # greedy: aceptar si mejora o iguala con menos bits
                f_old = self.fit[i]
                if (f_new > f_old) or (abs(f_new - f_old) < 1e-12 and sum(new_pos) < sum(self.pos[i])):
                    self.pos[i] = new_pos
                    self.fit[i] = f_new

                    # pbest
                    if (f_new > self.pbest_fit[i]) or (abs(f_new - self.pbest_fit[i]) < 1e-12 and sum(new_pos) < sum(self.pbest_pos[i])):
                        self.pbest_pos[i] = new_pos[:]
                        self.pbest_fit[i] = f_new

                    # gbest
                    if (f_new > self.gbest_fit) or (abs(f_new - self.gbest_fit) < 1e-12 and sum(new_pos) < sum(self.gbest_pos)):
                        self.gbest_pos = new_pos[:]
                        self.gbest_fit = f_new

            history.append({"iter": it, "best_obj": self.gbest_fit, "best_k": sum(self.gbest_pos)})

        return self.gbest_pos, {"best_obj": self.gbest_fit, "history": history}