# scripts/aco_selector.py
# Ant Colony Optimization (ACO) para selección de características (binario).
# Compatible con run_aco.py y fitness.py (eval_fn(mask) -> (obj, f1, k)).
# Optimizaciones:
#   - Early stopping por paciencia.
#   - Límite de evaluaciones (max_evals).
#   - Búsqueda por probabilidad Bernoulli basada en feromonas.
#   - Evaporación + refuerzo proporcional al objetivo.
#   - Respeta hard-cap de k_min..k_max mediante repair_fn externa.

from __future__ import annotations
from typing import Callable, List, Dict, Tuple
import random
import numpy as np


class ACOSelector:
    """
    ACO binario para selección de características.

    Parámetros principales
    ----------------------
    num_ants : int
        Hormigas por iteración.
    iters : int
        Iteraciones máximas.
    alpha : float
        Exponente para feromonas (exploit).
    beta : float
        Exponente para heurística (explore). En esta implementación la heurística
        es uniforme por defecto (se puede setear a 0.0 para no usarla).
    rho : float
        Tasa de evaporación (0..1).
    q : float
        Escala del refuerzo de feromonas.
    seed : int
        Semilla RNG.
    max_evals : int
        Máximo de evaluaciones de la función objetivo (corte duro).
    patience : int
        Nº de evaluaciones sin mejora tras las cuales se detiene (early stop).

    Interfaz de búsqueda
    --------------------
    fit_select(eval_fn, feature_names, init_mask, lam, max_depth, repair_fn, k_min, k_max)
        -> (best_mask, info)

    Donde:
      - eval_fn: Callable(mask) -> (obj, metrics_f1, k)  [en tu repo es evaluate_subset wrapper]
      - feature_names: lista de nombres (solo para forma/longitud)
      - init_mask: máscara inicial (warm-start), o None
      - lam, max_depth: parámetros pasados solo para mantener firma; no se usan aquí
      - repair_fn: Callable que ajusta la máscara a hard-cap y evita máscara vacía
      - k_min, k_max: límites deseados del tamaño de subset (enforced por repair_fn)
    """

    def __init__(
        self,
        num_ants: int = 10,
        iters: int = 10,
        alpha: float = 1.0,
        beta: float = 0.0,
        rho: float = 0.2,
        q: float = 1.0,
        seed: int = 13,
        max_evals: int = 300,
        patience: int = 6,
    ):
        self.num_ants = int(num_ants)
        self.iters = int(iters)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.q = float(q)
        self.seed = int(seed)
        self.max_evals = int(max_evals)
        self.patience = int(patience)

        # RNG determinista
        self._py_rng = random.Random(self.seed)
        np.random.seed(self.seed)

    # ------------------------------------------------------------------ #
    # Utilidades internas
    # ------------------------------------------------------------------ #
    def _build_mask(self, pheromone: np.ndarray, heuristic: np.ndarray) -> List[int]:
        """
        Construye una solución binaria por muestreo Bernoulli independiente
        con probabilidad proporcional a tau^alpha * eta^beta.
        """
        # Evitar división por cero o NaN en normalización
        score = (pheromone ** self.alpha) * (heuristic ** self.beta)
        score = np.clip(score, 1e-12, None)
        probs = score / score.max()  # escala a [0,1] aprox; no hace falta suma=1
        probs = np.clip(probs, 1e-6, 1 - 1e-6)  # evitar 0/1 exactos que congelan
        mask = (np.random.rand(len(probs)) < probs).astype(int).tolist()
        return mask

    # ------------------------------------------------------------------ #
    # Bucle principal ACO
    # ------------------------------------------------------------------ #
    def fit_select(
        self,
        eval_fn: Callable[[List[int]], Tuple[float, float, int]],
        feature_names: List[str],
        init_mask: List[int] | None = None,
        lam: float = 0.01,
        max_depth: int = 12,
        repair_fn: Callable[[List[int]], List[int]] | None = None,
        k_min: int = 1,
        k_max: int | None = None,
    ) -> Tuple[List[int], Dict]:
        """
        Devuelve: (best_mask, info)
          - best_mask : lista de 0/1
          - info : dict con {"best_obj", "history", "iters", "evaluations"}
        """
        n_features = len(feature_names)
        if n_features == 0:
            raise ValueError("feature_names vacío")

        # Feromonas iniciales (uniformes)
        pheromone = np.ones(n_features, dtype=float)
        # Heurística: uniforme (puedes cambiarla por ranking si lo calculas fuera)
        heuristic = np.ones(n_features, dtype=float)

        # Mejor global
        if init_mask is not None:
            m0 = init_mask[:]
            if repair_fn:
                m0 = repair_fn(m0)
            best_mask = m0
        else:
            # Warm start sencillo: activa unos pocos (k_min) al azar
            idx = list(range(n_features))
            self._py_rng.shuffle(idx)
            m0 = [0] * n_features
            for j in idx[: max(1, k_min)]:
                m0[j] = 1
            if repair_fn:
                m0 = repair_fn(m0)
            best_mask = m0

        best_obj, _, _ = eval_fn(best_mask)
        evals = 1
        no_improve = 0
        history = [float(best_obj)]

        # Bucle de iteraciones
        for _ in range(self.iters):
            ants_masks: List[List[int]] = []
            ants_objs: List[float] = []

            # Construcción de soluciones
            for _a in range(self.num_ants):
                mask = self._build_mask(pheromone, heuristic)
                if repair_fn:
                    mask = repair_fn(mask)

                obj, _, _ = eval_fn(mask)
                evals += 1

                ants_masks.append(mask)
                ants_objs.append(obj)

                # Actualizar mejor global
                if obj > best_obj:
                    best_obj = obj
                    best_mask = mask
                    no_improve = 0
                else:
                    no_improve += 1

                # Cortes por recursos
                if evals >= self.max_evals:
                    break

            # Evaporación
            pheromone *= (1.0 - self.rho)

            # Refuerzo (todas las hormigas de esta iteración)
            if ants_masks:
                for mask, obj in zip(ants_masks, ants_objs):
                    if obj > 0.0:  # solo refuerzo positivo
                        pheromone += self.q * obj * np.array(mask, dtype=float)

            # Refuerzo extra al best global (elitismo de feromona)
            if best_obj > 0.0:
                pheromone += (self.q * 0.5 * best_obj) * np.array(best_mask, dtype=float)

            # Evitar colapso numérico (clamp)
            pheromone = np.clip(pheromone, 1e-6, 1e6)

            # Historial y condiciones de parada
            history.append(float(best_obj))
            if evals >= self.max_evals or no_improve >= self.patience:
                break

        info = {
            "best_obj": float(best_obj),
            "history": history,
            "iters": self.iters,
            "evaluations": int(evals),
        }
        return best_mask, info