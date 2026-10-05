"""SA, ILS ve MOCO-CEM - makaledeki denklemlere sadik uygulamalar.

Onceki surumde bu algoritmalar peptide_generation_comparison.py icinde
model tanimlariyla ic ice yazilmisti. Burada tek bir amac fonksiyonu
arayuzu (score_fn) uzerinden calisirlar; boylece ayni optimize edici hem
egitilmis model hem de bagimsiz bir dogrulayici model ile kullanilabilir
(surrogate-driven optimisation).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from ..encoding import N_AA, PEPTIDE_LENGTH, decode_one


@dataclass
class OptimizationResult:
    best_sequence: str
    best_score: float
    n_evaluations: int
    trajectory: list[float] = field(default_factory=list)
    all_candidates: list[tuple[str, float]] = field(default_factory=list)
    method: str = ""


class _Optimizer:
    """Ortak taban. score_fn: (n, 12) uint8 -> (n,) skor dizisi (dusuk = iyi)."""

    name = "base"

    def __init__(self, score_fn, seed: int = 0, max_iterations: int = 3000,
                 batch_eval: int = 1):
        self.score_fn = score_fn
        self.rng = np.random.default_rng(seed)
        self.max_iterations = max_iterations
        self.batch_eval = batch_eval
        self._n_eval = 0

    def _score(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.uint8))
        self._n_eval += len(X)
        return np.asarray(self.score_fn(X), dtype=np.float64)

    def _score_one(self, x: np.ndarray) -> float:
        return float(self._score(x[None, :])[0])

    def _random_sequence(self) -> np.ndarray:
        return self.rng.integers(0, N_AA, PEPTIDE_LENGTH, dtype=np.uint8)

    def _substitute(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        i = int(self.rng.integers(PEPTIDE_LENGTH))
        choices = [a for a in range(N_AA) if a != y[i]]
        y[i] = self.rng.choice(choices)
        return y

    def _swap(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        i, j = self.rng.choice(PEPTIDE_LENGTH, 2, replace=False)
        y[i], y[j] = y[j], y[i]
        return y

    def _neighbour(self, x: np.ndarray) -> np.ndarray:
        """Denklem 8: %75 ikame, %25 yer degistirme."""
        return self._substitute(x) if self.rng.random() < 0.75 else self._swap(x)


class SimulatedAnnealing(_Optimizer):
    """Denklem 8-10. Sicaklik her iterasyonda degil, tau periyodunda guncellenir."""
    name = "SA"

    def __init__(self, score_fn, seed=0, max_iterations=3000,
                 t0=0.5, gamma=0.9, t_min=0.1, tau=50):
        super().__init__(score_fn, seed, max_iterations)
        self.t0, self.gamma, self.t_min, self.tau = t0, gamma, t_min, tau

    def optimize(self, initial=None) -> OptimizationResult:
        cur = self._random_sequence() if initial is None else np.asarray(initial, np.uint8)
        cur_s = self._score_one(cur)
        best, best_s = cur.copy(), cur_s
        traj, cands = [best_s], [(decode_one(best), best_s)]
        T = self.t0
        for it in range(self.max_iterations):
            cand = self._neighbour(cur)
            cand_s = self._score_one(cand)
            dE = cand_s - cur_s
            if dE < 0 or self.rng.random() < np.exp(-dE / max(T, 1e-9)):
                cur, cur_s = cand, cand_s
            if cur_s < best_s:
                best, best_s = cur.copy(), cur_s
                cands.append((decode_one(best), best_s))
            if (it + 1) % self.tau == 0:
                T = max(self.gamma * T, 0.0)
                if T < self.t_min:
                    break
            traj.append(best_s)
        return OptimizationResult(decode_one(best), best_s, self._n_eval,
                                  traj, cands, self.name)


class IteratedLocalSearch(_Optimizer):
    """Denklem 11-21. Uyarlanabilir bozma siddeti + yeniden baslatma."""
    name = "ILS"

    def __init__(self, score_fn, seed=0, max_iterations=3000,
                 local_steps=30, k_max=4, stagnation_limit=40):
        super().__init__(score_fn, seed, max_iterations)
        self.local_steps, self.k_max = local_steps, k_max
        self.stagnation_limit = stagnation_limit

    def _local_search(self, x, s):
        """Denklem 11-13: acgozlu tepe tirmanisi."""
        for _ in range(self.local_steps):
            cand = self._neighbour(x)
            cand_s = self._score_one(cand)
            if cand_s < s:
                x, s = cand, cand_s
        return x, s

    def _segment_kick(self, x):
        """Denklem 15: 4 kesim noktasiyla 5 segmentin yeniden siralanmasi."""
        cuts = np.sort(self.rng.choice(np.arange(1, PEPTIDE_LENGTH), 4, replace=False))
        segs = np.split(x, cuts)
        self.rng.shuffle(segs)
        return np.concatenate(segs).astype(np.uint8)

    def _mutation_kick(self, x, k):
        y = x.copy()
        for _ in range(k):
            y = self._substitute(y)
        return y

    def _perturb(self, x, k):
        """Denklem 16: %60 segment, %40 nokta mutasyonu."""
        return self._segment_kick(x) if self.rng.random() < 0.60 else self._mutation_kick(x, k)

    def optimize(self, initial=None) -> OptimizationResult:
        cur = self._random_sequence() if initial is None else np.asarray(initial, np.uint8)
        cur_s = self._score_one(cur)
        cur, cur_s = self._local_search(cur, cur_s)
        best, best_s = cur.copy(), cur_s
        traj, cands = [best_s], [(decode_one(best), best_s)]
        k, stagnant = 1, 0
        budget = self.max_iterations // max(self.local_steps, 1)

        for _ in range(budget):
            cand = self._perturb(cur, k)
            cand_s = self._score_one(cand)
            cand, cand_s = self._local_search(cand, cand_s)
            if cand_s < cur_s:                      # Denklem 19: acgozlu kabul
                cur, cur_s = cand, cand_s
            if cand_s < best_s:
                best, best_s = cand.copy(), cand_s
                cands.append((decode_one(best), best_s))
                k = max(1, k - 1)                   # Denklem 17
                stagnant = 0
            else:
                stagnant += 1
                k = min(self.k_max, k + 1)          # Denklem 18
            if stagnant >= self.stagnation_limit:   # Denklem 20-21
                cur = self._mutation_kick(best, self.k_max)
                cur_s = self._score_one(cur)
                cur, cur_s = self._local_search(cur, cur_s)
                stagnant = 0
            traj.append(best_s)
        return OptimizationResult(decode_one(best), best_s, self._n_eval,
                                  traj, cands, self.name)


class MocoCEM(_Optimizer):
    """Denklem 22-25. Pozisyon bazli dagilim + elit siralama agirligi + EMA."""
    name = "MOCO-CEM"

    def __init__(self, score_fn, seed=0, max_iterations=3000,
                 population=64, elite_frac=0.15, alpha=0.3,
                 tau0=1.0, tau_min=0.15):
        super().__init__(score_fn, seed, max_iterations)
        self.population = population
        self.n_elite = max(2, int(elite_frac * population))
        self.alpha, self.tau0, self.tau_min = alpha, tau0, tau_min

    def _tau(self, it, total):
        return max(self.tau_min, self.tau0 * (1 - it / max(total, 1)))

    def optimize(self, initial=None) -> OptimizationResult:
        theta = np.zeros((PEPTIDE_LENGTH, N_AA))          # Denklem 22
        if initial is not None:
            x0 = np.asarray(initial, np.uint8)
            theta[np.arange(PEPTIDE_LENGTH), x0] = 1.0
        P = np.full((PEPTIDE_LENGTH, N_AA), 1.0 / N_AA)
        best, best_s = None, np.inf
        traj, cands = [], []
        n_gen = max(1, self.max_iterations // self.population)

        # Denklem 24: sira tabanli agirliklar
        ranks = np.arange(1, self.n_elite + 1)
        w = (1.0 / ranks) / np.sum(1.0 / ranks)

        for gen in range(n_gen):
            tau = self._tau(gen, n_gen)
            e = np.exp((theta - theta.max(axis=1, keepdims=True)) / tau)   # Denklem 23
            P = e / e.sum(axis=1, keepdims=True)
            pop = np.stack([
                np.array([self.rng.choice(N_AA, p=P[p]) for p in range(PEPTIDE_LENGTH)],
                         dtype=np.uint8)
                for _ in range(self.population)])
            scores = self._score(pop)
            order = np.argsort(scores)
            if scores[order[0]] < best_s:
                best_s = float(scores[order[0]]); best = pop[order[0]].copy()
                cands.append((decode_one(best), best_s))

            elite = pop[order[:self.n_elite]]
            P_elite = np.zeros_like(P)
            for r in range(self.n_elite):
                P_elite[np.arange(PEPTIDE_LENGTH), elite[r]] += w[r]
            P = (1 - self.alpha) * P + self.alpha * P_elite               # Denklem 25
            theta = np.log(P + 1e-12) * tau
            traj.append(best_s)

        return OptimizationResult(decode_one(best), best_s, self._n_eval,
                                  traj, cands, self.name)


ALGORITHMS = {"SA": SimulatedAnnealing, "ILS": IteratedLocalSearch,
              "MOCO-CEM": MocoCEM}


def run_all(score_fn, n_peptides=10, seed=0, max_iterations=3000) -> dict:
    """Her algoritma ile n_peptides aday uretir."""
    out = {}
    for name, cls in ALGORITHMS.items():
        runs = []
        for r in range(n_peptides):
            opt = cls(score_fn, seed=seed * 1000 + r, max_iterations=max_iterations)
            res = opt.optimize()
            runs.append({"sequence": res.best_sequence,
                         "score": round(res.best_score, 4),
                         "n_evaluations": res.n_evaluations})
        seqs = [r["sequence"] for r in runs]
        out[name] = {
            "runs": runs,
            "best": min(runs, key=lambda r: r["score"]),
            "mean_score": round(float(np.mean([r["score"] for r in runs])), 4),
            "diversity_ratio": round(len(set(seqs)) / len(seqs), 3),
        }
    return out
