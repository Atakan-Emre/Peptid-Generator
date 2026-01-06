"""
ils_peptide_optimizer.py

Iterated Local Search (ILS) optimizer for peptide design (replacement for SA/MOCO).

- Objective: MINIMIZE the model-predicted score (smaller is better).
- Peptides are length 12 (same as your code) over AMINO_ACIDS.
- Uses local search (greedy hill-climb with substitution/swap),
  plus strong perturbations ("kicks") and adaptive kick strength.

Requirements from your project:
    - AMINO_ACIDS : iterable of allowed amino acid symbols
    - device      : torch device (e.g., torch.device("cuda") or "cpu")

Usage:
    from ils_peptide_optimizer import ILSPeptideOptimizer

    optimizer = ILSPeptideOptimizer(
        model=model,
        model_type=model_type,
        score_mean=score_mean,
        score_std=score_std,
        max_iterations=800,
        local_steps=60,
        kick_strength=1,
        kick_max=4,
        stagnation_iters=50,
        restart_iters=250,
        accept_mode="greedy",   # or "metropolis"
        T0=0.02,
        Tend=0.001,
        seed=1
    )

    results = optimizer.optimize(initial_peptide, n_samples=5)

Returned `results` matches your previous SA format keys:
    best_peptide, best_score, initial_peptide, initial_score, trajectory, improvement
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch


@dataclass
class ILSOptions:
    # Budget
    max_iterations: int = 800            # ILS outer iterations
    local_steps: int = 60               # hill-climb steps inside local search

    # Perturbation control
    kick_strength: int = 1              # starting number of perturb moves per iter
    kick_max: int = 4                   # maximum kick under stagnation
    p_segment_kick: float = 0.60        # prob of segment-rearrangement kick vs point mutation kick
    segment_min_len: int = 2
    segment_max_len: int = 5

    # Stagnation and restart
    stagnation_iters: int = 50          # after this many non-improving iters, increase kick
    restart_iters: int = 250            # after this many non-improving iters, restart around best

    # Acceptance (optional)
    accept_mode: str = "greedy"         # "greedy" or "metropolis"
    T0: float = 0.02                    # initial temperature (relative)
    Tend: float = 0.001                 # final temperature
    temp_schedule: str = "exp"          # "exp" or "linear"

    # Reproducibility
    seed: int = 1


class ILSPeptideOptimizer:
    """
    Iterated Local Search (ILS) for peptide optimization.

    Minimization: smaller score is better.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_type: str,
        score_mean: float,
        score_std: float,
        *,
        options: Optional[ILSOptions] = None,
        # allow convenient overrides
        max_iterations: Optional[int] = None,
        local_steps: Optional[int] = None,
        kick_strength: Optional[int] = None,
        kick_max: Optional[int] = None,
        stagnation_iters: Optional[int] = None,
        restart_iters: Optional[int] = None,
        accept_mode: Optional[str] = None,
        T0: Optional[float] = None,
        Tend: Optional[float] = None,
        temp_schedule: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        if options is None:
            options = ILSOptions()

        if max_iterations is not None: options.max_iterations = int(max_iterations)
        if local_steps is not None: options.local_steps = int(local_steps)
        if kick_strength is not None: options.kick_strength = int(kick_strength)
        if kick_max is not None: options.kick_max = int(kick_max)
        if stagnation_iters is not None: options.stagnation_iters = int(stagnation_iters)
        if restart_iters is not None: options.restart_iters = int(restart_iters)
        if accept_mode is not None: options.accept_mode = str(accept_mode)
        if T0 is not None: options.T0 = float(T0)
        if Tend is not None: options.Tend = float(Tend)
        if temp_schedule is not None: options.temp_schedule = str(temp_schedule)
        if seed is not None: options.seed = int(seed)

        self.opts = options

        self.model = model
        self.model_type = model_type
        self.score_mean = float(score_mean)
        self.score_std = float(score_std)

        # get AMINO_ACIDS and device from the running script context
        try:
            from __main__ import AMINO_ACIDS  # common when you run a single script/notebook
        except Exception:
            AMINO_ACIDS = None
        if AMINO_ACIDS is None:
            raise RuntimeError(
                "AMINO_ACIDS is not defined. Define AMINO_ACIDS in your main script "
                "before creating ILSPeptideOptimizer."
            )

        try:
            from __main__ import device
        except Exception:
            device = None
        if device is None:
            raise RuntimeError(
                "device is not defined. Define `device = torch.device(...)` in your main script "
                "before creating ILSPeptideOptimizer."
            )

        self.device = device
        self.amino_acids = list(AMINO_ACIDS)
        self.A = len(self.amino_acids)
        self.L = 12

        self.aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}

        self.model.eval()

        random.seed(self.opts.seed)
        np.random.seed(self.opts.seed)

    # ----------------------
    # Scoring (same as your SA)
    # ----------------------
    def peptide_to_one_hot(self, peptide: str) -> np.ndarray:
        one_hot = np.zeros((self.L, self.A), dtype=np.float32)
        for i, aa in enumerate(peptide):
            idx = self.aa_to_idx.get(aa, None)
            if idx is not None:
                one_hot[i, idx] = 1.0
        return one_hot[np.newaxis, :, :]

    def predict_score(self, peptide: str) -> float:
        one_hot = self.peptide_to_one_hot(peptide)
        X = torch.tensor(one_hot, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            if self.model_type == "lstm_vae":
                _, _, _, pred_norm = self.model(X)
            elif self.model_type == "encdec":
                _, pred_norm = self.model(X)
            else:
                pred_norm = self.model(X)

            pred = pred_norm.cpu().numpy()[0] * self.score_std + self.score_mean
            return float(pred)

    # ----------------------
    # Neighborhood operators
    # ----------------------
    def move_operator(self, peptide: str) -> str:
        """75% substitution, 25% swap."""
        if random.random() < 0.75:
            pos = random.randint(0, self.L - 1)
            current = peptide[pos]
            choices = [aa for aa in self.amino_acids if aa != current]
            new_aa = random.choice(choices)
            return peptide[:pos] + new_aa + peptide[pos + 1:]
        else:
            pos1, pos2 = random.sample(range(self.L), 2)
            lst = list(peptide)
            lst[pos1], lst[pos2] = lst[pos2], lst[pos1]
            return "".join(lst)

    def segment_rearrangement_kick(self, peptide: str) -> str:
        """
        Double-bridge-like perturbation adapted to sequences:
        cut into 5 parts and reorder middle parts to create a stronger shuffle
        while keeping the same multiset of amino acids.
        """
        n = self.L
        if n < 8:
            # fallback: reverse a segment
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            lst = list(peptide)
            lst[i:j+1] = reversed(lst[i:j+1])
            return "".join(lst)

        # choose 4 cut points in 1..n-1
        pts = sorted(random.sample(range(1, n), 4))
        a, b, c, d = pts
        s1 = peptide[0:a]
        s2 = peptide[a:b]
        s3 = peptide[b:c]
        s4 = peptide[c:d]
        s5 = peptide[d:n]
        # reorder: 1 | 3 | 2 | 4 | 5
        return "".join([s1, s3, s2, s4, s5])

    def point_mutation_kick(self, peptide: str) -> str:
        """
        Stronger kick: apply 2-3 random move_operator steps.
        """
        k = random.choice([2, 3])
        p = peptide
        for _ in range(k):
            p = self.move_operator(p)
        return p

    # ----------------------
    # Local search
    # ----------------------
    def local_search(self, peptide: str, score: float) -> Tuple[str, float]:
        """
        Greedy hill-climb using move_operator.
        """
        best_p = peptide
        best_s = score
        for _ in range(self.opts.local_steps):
            cand = self.move_operator(best_p)
            cs = self.predict_score(cand)
            if cs < best_s:
                best_p, best_s = cand, cs
        return best_p, best_s

    # ----------------------
    # Acceptance temperature
    # ----------------------
    def _temp(self, it: int) -> float:
        if self.opts.accept_mode.lower() != "metropolis":
            return 0.0

        K = max(1, self.opts.max_iterations)
        T0 = max(self.opts.T0, 1e-12)
        Tend = max(self.opts.Tend, 1e-12)

        if K == 1:
            return Tend

        if self.opts.temp_schedule.lower() == "linear":
            T = T0 + (Tend - T0) * (it - 1) / (K - 1)
        else:
            c = 5.0 / (K - 1)
            T = Tend + (T0 - Tend) * np.exp(-c * (it - 1))

        return float(max(T, 1e-12))

    # ----------------------
    # Main optimize
    # ----------------------
    def optimize(self, initial_peptide: str, n_samples: int = 1) -> List[Dict[str, Any]]:
        """
        Multi-start ILS.
        Returns list of dicts compatible with your SA output.
        """
        results: List[Dict[str, Any]] = []

        for sample_idx in range(n_samples):
            if sample_idx == 0:
                start_peptide = initial_peptide
            else:
                start_peptide = "".join(np.random.choice(self.amino_acids, size=self.L))

            start_score = self.predict_score(start_peptide)

            # local optimum from start
            cur_p, cur_s = self.local_search(start_peptide, start_score)

            best_p, best_s = cur_p, cur_s

            trajectory: List[Tuple[int, float]] = [(0, best_s)]

            kick = max(1, int(self.opts.kick_strength))
            no_imp = 0

            for it in range(1, self.opts.max_iterations + 1):
                T = self._temp(it)

                # ----- perturb ("kick") -----
                cand = cur_p
                for _ in range(kick):
                    if random.random() < self.opts.p_segment_kick:
                        cand = self.segment_rearrangement_kick(cand)
                    else:
                        cand = self.point_mutation_kick(cand)

                # ----- local search -----
                cand_s0 = self.predict_score(cand)
                cand, cand_s = self.local_search(cand, cand_s0)

                # ----- acceptance -----
                accept = False
                if cand_s < cur_s:
                    accept = True
                elif self.opts.accept_mode.lower() == "metropolis":
                    d = cand_s - cur_s
                    denom = max(abs(cur_s), 1e-12)
                    pacc = float(np.exp(-(d / denom) / max(T, 1e-12)))
                    if random.random() < pacc:
                        accept = True

                if accept:
                    cur_p, cur_s = cand, cand_s

                # ----- best update + adaptive kick -----
                if cur_s < best_s:
                    best_p, best_s = cur_p, cur_s
                    trajectory.append((it, best_s))
                    no_imp = 0
                    kick = max(1, kick - 1)  # reward improvement: smaller kick
                else:
                    no_imp += 1
                    # stagnation => increase kick (up to kick_max)
                    if no_imp > 0 and (no_imp % max(1, self.opts.stagnation_iters) == 0):
                        kick = min(int(self.opts.kick_max), kick + 1)

                    # long stagnation => restart around best
                    if no_imp >= self.opts.restart_iters:
                        cur_p = best_p
                        # apply a few strong kicks then local search
                        for _ in range(int(self.opts.kick_max)):
                            cur_p = self.segment_rearrangement_kick(cur_p)
                        cur_s0 = self.predict_score(cur_p)
                        cur_p, cur_s = self.local_search(cur_p, cur_s0)
                        no_imp = 0
                        kick = max(1, int(self.opts.kick_strength))

            results.append({
                "best_peptide": best_p,
                "best_score": best_s,
                "initial_peptide": start_peptide,
                "initial_score": start_score,
                "trajectory": trajectory,
                "improvement": start_score - best_s,
                "ils_opts": self.opts.__dict__.copy(),
            })

        return results
