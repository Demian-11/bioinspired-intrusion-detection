# scripts/abc_selector.py
# ABC binario para selección de características:
# - Employed: vecino y aceptación greedy.
# - Onlookers: ruleta con prob ∝ fitness (shift para evitar negativos).
# - Scouts: reinicio tras 'limit' sin mejora.

from typing import List, Tuple, Callable
import random

class ABCSelector:
    def __init__(self,
                 num_bees: int = 20,
                 max_iters: int = 50,
                 limit: int = 10,
                 lam: float = 0.01,
                 seed: int = 13):
        self.num_bees = num_bees
        self.max_iters = max_iters
        self.limit = limit
        self.lam = lam
        self.rng = random.Random(seed)

        self.pop: List[List[int]] = []
        self.fit: List[float] = []
        self.trials: List[int] = []

    # -------- helpers --------
    def _rand_mask(self, n: int, p: float = 0.3) -> List[int]:
        m = [1 if self.rng.random() < p else 0 for _ in range(n)]
        if sum(m) == 0:
            m[self.rng.randrange(n)] = 1
        return m

    def _neighbor(self, x: List[int], y: List[int]) -> List[int]:
        n = len(x)
        j = self.rng.randrange(n)
        v = x[:]
        if self.rng.random() < 0.5:
            v[j] = y[j]              # mover hacia y
        else:
            v[j] = 1 - v[j]          # flip
        if self.rng.random() < 0.1:   # micro-perturbación
            k = self.rng.randrange(n)
            v[k] = 1 - v[k]
        if sum(v) == 0:
            v[self.rng.randrange(n)] = 1
        return v

    def _prob_from_fitness(self, fits: List[float]) -> List[float]:
        m = min(fits)
        shifted = [f - m + 1e-9 for f in fits]  # evita negativos
        s = sum(shifted)
        if s == 0:
            return [1.0 / len(fits)] * len(fits)
        return [v / s for v in shifted]

    # -------- core --------
    def fit_select(self,
                   eval_fn: Callable,
                   train_df,
                   test_df,
                   feature_names: List[str],
                   max_depth: int = 12) -> Tuple[List[int], dict]:
        n = len(feature_names)

        # init población
        self.pop = [self._rand_mask(n, p=0.3) for _ in range(self.num_bees)]
        self.fit = []
        for x in self.pop:
            obj, _, _ = eval_fn(train_df, test_df, feature_names, x,
                                max_depth=max_depth, lam=self.lam)
            self.fit.append(obj)
        self.trials = [0] * self.num_bees

        best_idx = max(range(self.num_bees), key=lambda i: self.fit[i])
        best = self.pop[best_idx][:]
        best_fit = self.fit[best_idx]
        history = [{"iter": 0, "best_obj": best_fit, "best_k": sum(best)}]

        for it in range(1, self.max_iters + 1):
            # Employed
            for i in range(self.num_bees):
                k = self.rng.randrange(self.num_bees)
                while k == i:
                    k = self.rng.randrange(self.num_bees)
                v = self._neighbor(self.pop[i], self.pop[k])
                obj_v, _, _ = eval_fn(train_df, test_df, feature_names, v,
                                      max_depth=max_depth, lam=self.lam)
                if (obj_v > self.fit[i]) or (abs(obj_v - self.fit[i]) < 1e-12 and sum(v) < sum(self.pop[i])):
                    self.pop[i] = v
                    self.fit[i] = obj_v
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1

            # Onlookers (ruleta)
            probs = self._prob_from_fitness(self.fit)
            for _ in range(self.num_bees):
                r = self.rng.random()
                acc, idx = 0.0, 0
                for j, p in enumerate(probs):
                    acc += p
                    if r <= acc:
                        idx = j
                        break
                k = self.rng.randrange(self.num_bees)
                while k == idx:
                    k = self.rng.randrange(self.num_bees)
                v = self._neighbor(self.pop[idx], self.pop[k])
                obj_v, _, _ = eval_fn(train_df, test_df, feature_names, v,
                                      max_depth=max_depth, lam=self.lam)
                if (obj_v > self.fit[idx]) or (abs(obj_v - self.fit[idx]) < 1e-12 and sum(v) < sum(self.pop[idx])):
                    self.pop[idx] = v
                    self.fit[idx] = obj_v
                    self.trials[idx] = 0
                else:
                    self.trials[idx] += 1

            # Scouts
            for i in range(self.num_bees):
                if self.trials[i] >= self.limit:
                    self.pop[i] = self._rand_mask(n, p=0.2)
                    obj_i, _, _ = eval_fn(train_df, test_df, feature_names, self.pop[i],
                                          max_depth=max_depth, lam=self.lam)
                    self.fit[i] = obj_i
                    self.trials[i] = 0

            # Mejor global
            idx_best = max(range(self.num_bees), key=lambda i: self.fit[i])
            if self.fit[idx_best] > best_fit:
                best_fit = self.fit[idx_best]
                best = self.pop[idx_best][:]

            history.append({"iter": it, "best_obj": best_fit, "best_k": sum(best)})

        return best, {"best_obj": best_fit, "history": history}