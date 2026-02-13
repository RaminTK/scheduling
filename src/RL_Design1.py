# ===========================
# rl_design1.py
# ===========================
"""
RL algorithm-switching for JSSP using State Design 1:
  State = (recent improvement trend bin, stagnation bin)

- Reads your existing files: sa_only.py and tabu_only.py
- Uses the SAME representation and decoder from your baselines.
- Action: choose between {SA, TABU} at each decision step.
- Reward: 1 if global best makespan improves during the step, else 0.

Run examples:
  python rl_design1.py ft06
  python rl_design1.py abz7 --steps 300 --budget 300 --tabu-samples 150 --tenure 10
  python rl_design1.py la01 --seed 7 --eps 0.2

Requirements:
  - Put this file in the same folder as sa_only.py and tabu_only.py
  - Ensure both baseline files run/import without errors.
"""

import argparse
import random
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple, Optional
from collections import deque

import sa_only as sa
import tabu_only as ts


# =========================================================
# Budgeted operators (share decoder + moves from your files)
# =========================================================

@dataclass
class SAState:
    T: float

@dataclass
class TabuState:
    tabu: Deque[Tuple[int, int, str]]  # move signatures FIFO


def _apply_move(seq: List[int], i: int, j: int, neighborhood: str) -> List[int]:
    if neighborhood == "swap":
        return sa.move_swap(seq, i, j)
    else:
        return sa.move_insert(seq, i, j)


def sa_step_budgeted(
    jobs_ops,
    curr_seq: List[int],
    curr_val: int,
    rng: random.Random,
    budget_decodes: int,
    sa_state: SAState,
    alpha: float,
    neighborhood: str,
) -> Tuple[List[int], int, int, SAState]:
    """
    Runs SA for approximately 'budget_decodes' neighbor evaluations.
    Returns: new_seq, new_val, decodes_used, updated_sa_state
    """
    n_jobs = len(jobs_ops)
    n_machines = len(jobs_ops[0])
    n = n_jobs * n_machines

    T = sa_state.T
    dec = 0

    while dec < budget_decodes:
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            j = (j + 1) % n

        cand = _apply_move(curr_seq, i, j, neighborhood)
        cand_val = sa.decode_priority_list(jobs_ops, cand)
        dec += 1

        delta = cand_val - curr_val  # minimization

        if delta <= 0:
            curr_seq, curr_val = cand, cand_val
        else:
            if T > 1e-12 and rng.random() < (2.718281828459045 ** (-delta / T)):
                curr_seq, curr_val = cand, cand_val

        T *= alpha

    return curr_seq, curr_val, dec, SAState(T=T)


def tabu_step_budgeted(
    jobs_ops,
    curr_seq: List[int],
    curr_val: int,
    rng: random.Random,
    budget_decodes: int,
    tabu_state: TabuState,
    tenure: int,
    neighborhood: str,
    samples: int,
) -> Tuple[List[int], int, int, TabuState]:
    """
    Runs Tabu Search for as many iterations as fit in 'budget_decodes' evaluations.
    Each tabu iteration evaluates up to 'samples' neighbors.
    Returns: new_seq, new_val, decodes_used, updated_tabu_state
    """
    n_jobs = len(jobs_ops)
    n_machines = len(jobs_ops[0])
    n = n_jobs * n_machines

    dec = 0
    tabu = tabu_state.tabu

    # If samples <= 0, fall back to 1
    samples = max(1, samples)

    while dec < budget_decodes:
        cand_best = None
        cand_best_val = None
        cand_move = None

        # sample neighbors, each costs 1 decode
        for _ in range(samples):
            if dec >= budget_decodes:
                break

            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue

            move_sig = (min(i, j), max(i, j), neighborhood)
            if move_sig in tabu:
                continue

            cand = _apply_move(curr_seq, i, j, neighborhood)
            val = sa.decode_priority_list(jobs_ops, cand)
            dec += 1

            if cand_best_val is None or val < cand_best_val:
                cand_best = cand
                cand_best_val = val
                cand_move = move_sig

        # fallback if no candidate found (all tabu or zero evaluated)
        if cand_best is None:
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                j = (j + 1) % n
            cand_best = _apply_move(curr_seq, i, j, "swap")  # safe fallback
            cand_best_val = sa.decode_priority_list(jobs_ops, cand_best)
            dec += 1
            cand_move = (min(i, j), max(i, j), "swap")

        curr_seq, curr_val = cand_best, cand_best_val

        tabu.append(cand_move)
        while len(tabu) > tenure:
            tabu.popleft()

    return curr_seq, curr_val, dec, TabuState(tabu=tabu)


# =========================================================
# State Design 1: (trend bin, stagnation bin)
# =========================================================

def _bin_trend(trend: float) -> int:
    """
    trend = (ms_{t-W} - ms_t) / ms_{t-W}
    bins:
      0 = worsening/negative (<= 0)
      1 = small improvement (0..1%)
      2 = strong improvement (>1%)
    """
    if trend <= 0.0:
        return 0
    elif trend <= 0.01:
        return 1
    else:
        return 2


def _bin_stagnation(stag_steps: int) -> int:
    """
    bins:
      0 = short (0..2)
      1 = medium (3..7)
      2 = long (>=8)
    """
    if stag_steps <= 2:
        return 0
    elif stag_steps <= 7:
        return 1
    else:
        return 2


def state_design_1(
    best_history: List[int],
    stagnation_steps: int,
    W: int,
) -> int:
    """
    Returns a discrete state id in [0..8] for 3x3 bins.
    """
    if len(best_history) <= W:
        trend = 0.0
    else:
        prev = best_history[-1 - W]
        now = best_history[-1]
        trend = (prev - now) / float(prev) if prev > 0 else 0.0

    tbin = _bin_trend(trend)
    sbin = _bin_stagnation(stagnation_steps)
    return 3 * tbin + sbin  # 0..8


# =========================================================
# Q-learning loop
# =========================================================

def choose_action_eps_greedy(Q: List[List[float]], s: int, rng: random.Random, eps: float) -> int:
    # 0 = SA, 1 = TABU
    if rng.random() < eps:
        return rng.randrange(2)
    return 0 if Q[s][0] >= Q[s][1] else 1


def qlearn_run(
    jobs_ops,
    seed: int,
    steps: int,
    budget: int,
    # SA params
    sa_T0: float,
    sa_alpha: float,
    sa_neigh: str,
    # Tabu params
    tenure: int,
    tabu_samples: int,
    tabu_neigh: str,
    # RL params
    eps: float,
    lr: float,
    gamma: float,
    window_W: int,
    verbose_every: int,
) -> Dict[str, float]:
    rng = random.Random(seed)

    n_jobs = len(jobs_ops)
    n_machines = len(jobs_ops[0])

    # initial solution
    curr_seq = sa.random_priority(n_jobs, n_machines, rng)
    curr_val = sa.decode_priority_list(jobs_ops, curr_seq)
    best_val = curr_val

    # algorithm internal states
    sa_state = SAState(T=sa_T0)
    tabu_state = TabuState(tabu=deque())

    # RL bookkeeping
    Q = [[0.0, 0.0] for _ in range(9)]  # 9 states x 2 actions
    best_history = [best_val]
    stagnation = 0

    # stats
    total_reward = 0
    sa_chosen = 0
    tabu_chosen = 0

    for t in range(steps):
        s = state_design_1(best_history=best_history, stagnation_steps=stagnation, W=window_W)
        a = choose_action_eps_greedy(Q, s, rng, eps)

        prev_best = best_val

        # action -> apply chosen method for equal decode budget
        if a == 0:
            sa_chosen += 1
            curr_seq, curr_val, _, sa_state = sa_step_budgeted(
                jobs_ops,
                curr_seq,
                curr_val,
                rng,
                budget_decodes=budget,
                sa_state=sa_state,
                alpha=sa_alpha,
                neighborhood=sa_neigh,
            )
        else:
            tabu_chosen += 1
            curr_seq, curr_val, _, tabu_state = tabu_step_budgeted(
                jobs_ops,
                curr_seq,
                curr_val,
                rng,
                budget_decodes=budget,
                tabu_state=tabu_state,
                tenure=tenure,
                neighborhood=tabu_neigh,
                samples=tabu_samples,
            )

        if curr_val < best_val:
            best_val = curr_val

        # reward: 1 if global best improved
        r = 1 if best_val < prev_best else 0
        total_reward += r

        # update stagnation + history
        if r == 1:
            stagnation = 0
        else:
            stagnation += 1
        best_history.append(best_val)

        # next state
        s2 = state_design_1(best_history=best_history, stagnation_steps=stagnation, W=window_W)

        # Q update
        Q[s][a] = Q[s][a] + lr * (r + gamma * max(Q[s2][0], Q[s2][1]) - Q[s][a])

        if verbose_every and ((t + 1) % verbose_every == 0 or t == 0):
            print(
                f"[D1] step {t+1:5d}/{steps} | a={'SA' if a==0 else 'TABU'} | "
                f"curr={curr_val} | best={best_val} | r={r} | stag={stagnation} | eps={eps:.3f}"
            )

    return {
        "best_makespan": float(best_val),
        "total_reward": float(total_reward),
        "sa_chosen": float(sa_chosen),
        "tabu_chosen": float(tabu_chosen),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=str, help="e.g., ft06, abz7, la01")
    parser.add_argument("--file", type=str, default=None, help="path to OR-Library jobshop1.txt")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--budget", type=int, default=200, help="decode budget per RL step")

    # SA params
    parser.add_argument("--sa-T0", type=float, default=10.0)
    parser.add_argument("--sa-alpha", type=float, default=0.9995)
    parser.add_argument("--sa-neigh", type=str, default="swap", choices=["swap", "insert"])

    # Tabu params
    parser.add_argument("--tenure", type=int, default=10)
    parser.add_argument("--tabu-samples", type=int, default=150)
    parser.add_argument("--tabu-neigh", type=str, default="swap", choices=["swap", "insert"])

    # RL params
    parser.add_argument("--eps", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)

    # State design params
    parser.add_argument("--W", type=int, default=5, help="trend window length in RL steps")

    parser.add_argument("--verbose-every", type=int, default=20)
    args = parser.parse_args()

    # use same default file path as your baselines, unless overridden
    file_path = args.file if args.file is not None else sa.__dict__.get("DEFAULT_FILE", None)
    if file_path is None:
        # fall back to the default from sa_only.py argument default if not exposed
        file_path = "jobshop1.txt"

    jobs_ops, n_jobs, n_machines = sa.load_jssp_instance(file_path, args.instance)
    print(f"Loaded {args.instance}: jobs={n_jobs}, machines={n_machines}")

    out = qlearn_run(
        jobs_ops,
        seed=args.seed,
        steps=args.steps,
        budget=args.budget,
        sa_T0=args.sa_T0,
        sa_alpha=args.sa_alpha,
        sa_neigh=args.sa_neigh,
        tenure=args.tenure,
        tabu_samples=args.tabu_samples,
        tabu_neigh=args.tabu_neigh,
        eps=args.eps,
        lr=args.lr,
        gamma=args.gamma,
        window_W=args.W,
        verbose_every=args.verbose_every,
    )

    print("\n=== RL (Design 1) summary ===")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
