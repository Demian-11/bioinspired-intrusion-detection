# scripts/fpa_selector.py
# Flower Pollination Algorithm (FPA) binario con early-stop y tope de evaluaciones.

from __future__ import annotations
import math, random
from typing import List, Tuple, Callable, Dict, Optional

class FPASelector:
    def __init__(
        self,
        pop_size: int = 12,
        iters: int = 12,
        p_global: float = 0.8,
        levy_beta: float = 1.5,
        step_scale: float = 0.25,
        seed: int = 13,
        elitism: int = 1,
        patience: int = 5,          # early-stop si no mejora en X iteraciones
        max_evals: Optional[int] = None,  # tope duro de evaluaciones
    ):
        self.pop_size = pop_size
        self.iters = iters
        self.p_global = p_global
        self.beta = levy_beta
        self.step_scale = step_scale
        self.elitism = elitism
        self.patience = patience
        self.max_evals = max_evals
        self.rng = random.Random(seed)

    # ---------- utils ----------
    def _levy_step_scalar(self) -> float:
        u = self.rng.gauss(0.0, 1.0)
        v = abs(self.rng.gauss(0.0, 1.0)) ** (1.0 / self.beta)
        return (u / (v + 1e-12)) * self.step_scale

    def _sigmoid(self, x: float) -> float:
        if x >= 35: return 1.0
        if x <= -35: return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def _global_pollination(self, x: List[int], gbest: List[int]) -> List[int]:
        y = x[:]
        step = self._levy_step_scalar()
        for i in range(len(x)):
            delta = (gbest[i] - x[i]) * step
            p_flip = abs(self._sigmoid(delta) - 0.5) * 2.0
            if self.rng.random() < p_flip:
                y[i] = 1 - y[i]
        return y

    def _local_pollination(self, x: List[int], a: List[int], b: List[int]) -> List[int]:
        y = x[:]
        for i in range(len(x)):
            y[i] = a[i] if self.rng.random() < 0.5 else b[i]
        return y

    def _init_mask(self, D: int, target_k: int) -> List[int]:
        idxs = list(range(D))
        self.rng.shuffle(idxs)
        m = [0]*D
        for i in idxs[:max(1, min(target_k, D))]:
            m[i] = 1
        return m

    # ---------- main ----------
    def fit_select(
        self,
        eval_fn: Callable,               # def f(mask) -> (obj, metrics, _)
        feature_names: List[str],
        init_mask: List[int] | None,
        lam: float,
        max_depth: int,
        repair_fn: Callable[[List[int]], List[int]],
    ) -> Tuple[List[int], Dict]:
        D = len(feature_names)
        K_hint = 9
        pop: List[List[int]] = []

        if init_mask is not None:
            pop.append(repair_fn(init_mask[:]))
        while len(pop) < self.pop_size:
            pop.append(repair_fn(self._init_mask(D, K_hint)))

        objs = []
        for m in pop:
            obj, _, _ = eval_fn(m)
            objs.append(obj)

        g_idx = max(range(len(pop)), key=lambda i: objs[i])
        gbest = pop[g_idx][:]
        gbest_obj = objs[g_idx]
        history = [float(gbest_obj)]
        n_eval = len(pop)

        no_improve = 0
        for _ in range(self.iters):
            # Early stop: tope de evaluaciones
            if self.max_evals is not None and n_eval >= self.max_evals:
                break

            # Elitismo
            elite_idx = sorted(range(len(pop)), key=lambda i: objs[i], reverse=True)[:self.elitism]
            elites = [pop[i][:] for i in elite_idx]
            elite_objs = [objs[i] for i in elite_idx]

            new_pop = elites[:]
            new_objs = elite_objs[:]

            while len(new_pop) < self.pop_size:
                if self.max_evals is not None and n_eval >= self.max_evals:
                    # rellena con copias para cerrar generación
                    i = len(new_pop) % len(pop)
                    new_pop.append(pop[i][:])
                    new_objs.append(objs[i])
                    continue

                i = self.rng.randrange(len(pop))
                x = pop[i]

                if self.rng.random() < self.p_global:
                    cand = self._global_pollination(x, gbest)
                else:
                    a_idx = self.rng.randrange(len(pop))
                    b_idx = self.rng.randrange(len(pop))
                    while b_idx == a_idx:
                        b_idx = self.rng.randrange(len(pop))
                    cand = self._local_pollination(x, pop[a_idx], pop[b_idx])

                cand = repair_fn(cand)
                cand_obj, _, _ = eval_fn(cand)
                n_eval += 1

                if cand_obj > objs[i]:
                    new_pop.append(cand)
                    new_objs.append(cand_obj)
                else:
                    new_pop.append(x[:])
                    new_objs.append(objs[i])

            pop, objs = new_pop, new_objs
            g_idx = max(range(len(pop)), key=lambda j: objs[j])
            improved = False
            if objs[g_idx] > gbest_obj:
                gbest_obj = objs[g_idx]
                gbest = pop[g_idx][:]
                improved = True
                no_improve = 0
            else:
                no_improve += 1

            history.append(float(gbest_obj))
            if self.patience is not None and no_improve >= self.patience:
                break

        return gbest, {
            "best_obj": float(gbest_obj),
            "history": history,
            "iters": self.iters,
            "evaluations": int(n_eval),
            "stopped_early": (self.patience is not None and no_improve >= self.patience)
                             or (self.max_evals is not None and n_eval >= self.max_evals),
        }