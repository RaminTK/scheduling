# ===========================
# rl_design2.py
# ===========================
"""
RL algorithm-switching for JSSP using State Design 2:
  State = (local ruggedness bin, tabu-pressure bin)

- local ruggedness ~ fraction of random neighbors that are worse than current
- tabu pressure ~ fraction of sampled move signatures currently tabu

- Reads your existing files: sa_only.py and tabu_only.py
- Action: choose between {SA, TABU} at each decision step.
- Reward: 1 if global best makespan improves during the step, else 0.

Run examples:
  python rl_design2.py ft06
  python rl_design2.py abz7 --steps 300 --budget 300 --probe 40
  python rl_design2.py la01 --seed 7 --eps 0.2
"""

import argparse
import random
from dataclasses import dataclass
from typing import Deque, List, Tuple, Dict
from collections import deque

import sa_only as sa
import tabu_only as ts
import time


# =========================================================
# Budgeted operators (same as rl_design1.py)
# =========================================================

@dataclass
class SAState:
    T: float

@dataclass
class TabuState:
    tabu: Deque[Tuple[int, int, str]]


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

        delta = cand_val - curr_val

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
    n_jobs = len(jobs_ops)
    n_machines = len(jobs_ops[0])
    n = n_jobs * n_machines

    dec = 0
    tabu = tabu_state.tabu
    samples = max(1, samples)

    while dec < budget_decodes:
        cand_best = None
        cand_best_val = None
        cand_move = None

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

        if cand_best is None:
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                j = (j + 1) % n
            cand_best = _apply_move(curr_seq, i, j, "swap")
            cand_best_val = sa.decode_priority_list(jobs_ops, cand_best)
            dec += 1
            cand_move = (min(i, j), max(i, j), "swap")

        curr_seq, curr_val = cand_best, cand_best_val

        tabu.append(cand_move)
        while len(tabu) > tenure:
            tabu.popleft()

    return curr_seq, curr_val, dec, TabuState(tabu=tabu)


# =========================================================
# State Design 2: (local ruggedness, tabu pressure)
# =========================================================

def _bin_3(p: float) -> int:
    """
    Map a probability-like value to 3 bins: low/mid/high
      0: p < 1/3
      1: 1/3 <= p < 2/3
      2: p >= 2/3
    """
    if p < (1.0 / 3.0):
        return 0
    elif p < (2.0 / 3.0):
        return 1
    else:
        return 2


def estimate_local_ruggedness(
    jobs_ops,
    curr_seq: List[int],
    curr_val: int,
    rng: random.Random,
    probe: int,
    neighborhood: str,
) -> float:
    """
    p_worse = fraction of random neighbors that have makespan > curr_val
    Cost: 'probe' decodes.
    """
    n_jobs = len(jobs_ops)
    n_machines = len(jobs_ops[0])
    n = n_jobs * n_machines

    worse = 0
    tested = 0

    for _ in range(max(1, probe)):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            j = (j + 1) % n
        cand = _apply_move(curr_seq, i, j, neighborhood)
        v = sa.decode_priority_list(jobs_ops, cand)
        tested += 1
        if v > curr_val:
            worse += 1

    return worse / float(tested) if tested > 0 else 0.0


def estimate_tabu_pressure(
    rng: random.Random,
    tabu: Deque[Tuple[int, int, str]],
    n_positions: int,
    probe: int,
    neighborhood: str,
) -> float:
    """
    p_tabu = fraction of sampled random move signatures currently tabu
    (no decoding needed; just membership checks)
    """
    hits = 0
    tested = 0
    tabu_set = set(tabu)  # fast membership

    for _ in range(max(1, probe)):
        i = rng.randrange(n_positions)
        j = rng.randrange(n_positions)
        if i == j:
            continue
        sig = (min(i, j), max(i, j), neighborhood)
        tested += 1
        if sig in tabu_set:
            hits += 1

    return hits / float(tested) if tested > 0 else 0.0


def state_design_2(
    jobs_ops,
    curr_seq: List[int],
    curr_val: int,
    rng: random.Random,
    probe: int,
    neigh_for_probe: str,
    tabu: Deque[Tuple[int, int, str]],
    n_positions: int,
    tabu_neigh: str,
) -> Tuple[int, float, float]:
    """
    Returns:
      state_id in [0..8], p_worse, p_tabu
    """
    p_worse = estimate_local_ruggedness(jobs_ops, curr_seq, curr_val, rng, probe, neigh_for_probe)
    p_tabu = estimate_tabu_pressure(rng, tabu, n_positions, probe, tabu_neigh)

    rb = _bin_3(p_worse)
    tb = _bin_3(p_tabu)
    s = 3 * rb + tb
    return s, p_worse, p_tabu


# =========================================================
# Q-learning loop
# =========================================================

def choose_action_eps_greedy(Q: List[List[float]], s: int, rng: random.Random, eps: float) -> int:
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
    # state probe
    probe: int,
    probe_neigh: str,
    # RL params
    eps: float,
    lr: float,
    gamma: float,
    verbose_every: int,
) -> Dict[str, float]:
    rng = random.Random(seed)

    n_jobs = len(jobs_ops)
    n_machines = len(jobs_ops[0])
    n_positions = n_jobs * n_machines

    curr_seq = sa.random_priority(n_jobs, n_machines, rng)
    curr_val = sa.decode_priority_list(jobs_ops, curr_seq)
    best_val = curr_val

    sa_state = SAState(T=sa_T0)
    tabu_state = TabuState(tabu=deque())

    Q = [[0.0, 0.0] for _ in range(9)]
    total_reward = 0
    sa_chosen = 0
    tabu_chosen = 0

    for t in range(steps):
        # compute state before action
        s, p_worse, p_tabu = state_design_2(
            jobs_ops=jobs_ops,
            curr_seq=curr_seq,
            curr_val=curr_val,
            rng=rng,
            probe=probe,
            neigh_for_probe=probe_neigh,
            tabu=tabu_state.tabu,
            n_positions=n_positions,
            tabu_neigh=tabu_neigh,
        )

        a = choose_action_eps_greedy(Q, s, rng, eps)
        prev_best = best_val

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

        r = 1 if best_val < prev_best else 0
        total_reward += r

        # next state (post-action) for Q update
        s2, _, _ = state_design_2(
            jobs_ops=jobs_ops,
            curr_seq=curr_seq,
            curr_val=curr_val,
            rng=rng,
            probe=probe,
            neigh_for_probe=probe_neigh,
            tabu=tabu_state.tabu,
            n_positions=n_positions,
            tabu_neigh=tabu_neigh,
        )

        Q[s][a] = Q[s][a] + lr * (r + gamma * max(Q[s2][0], Q[s2][1]) - Q[s][a])

        if verbose_every and ((t + 1) % verbose_every == 0 or t == 0):
            print(
                f"[D2] step {t+1:5d}/{steps} | a={'SA' if a==0 else 'TABU'} | "
                f"curr={curr_val} | best={best_val} | r={r} | "
                f"p_worse={p_worse:.2f} | p_tabu={p_tabu:.2f} | eps={eps:.3f}"
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
    parser.add_argument("--budget", type=int, default=200)

    # SA params
    parser.add_argument("--sa-T0", type=float, default=10.0)
    parser.add_argument("--sa-alpha", type=float, default=0.9995)
    parser.add_argument("--sa-neigh", type=str, default="swap", choices=["swap", "insert"])

    # Tabu params
    parser.add_argument("--tenure", type=int, default=10)
    parser.add_argument("--tabu-samples", type=int, default=150)
    parser.add_argument("--tabu-neigh", type=str, default="swap", choices=["swap", "insert"])

    # State design 2 probe
    parser.add_argument("--probe", type=int, default=30, help="neighbors sampled for p_worse and p_tabu")
    parser.add_argument("--probe-neigh", type=str, default="swap", choices=["swap", "insert"],
                        help="neighborhood used when probing ruggedness (p_worse)")

    # RL params
    parser.add_argument("--eps", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)

    parser.add_argument("--verbose-every", type=int, default=20)
    args = parser.parse_args()

    file_path = args.file if args.file is not None else "jobshop1.txt"

    jobs_ops, n_jobs, n_machines = sa.load_jssp_instance(file_path, args.instance)
    print(f"Loaded {args.instance}: jobs={n_jobs}, machines={n_machines}")

    out = qlearn_run(
        jobs_ops=jobs_ops,
        seed=args.seed,
        steps=args.steps,
        budget=args.budget,
        sa_T0=args.sa_T0,
        sa_alpha=args.sa_alpha,
        sa_neigh=args.sa_neigh,
        tenure=args.tenure,
        tabu_samples=args.tabu_samples,
        tabu_neigh=args.tabu_neigh,
        probe=args.probe,
        probe_neigh=args.probe_neigh,
        eps=args.eps,
        lr=args.lr,
        gamma=args.gamma,
        verbose_every=args.verbose_every,
    )

    print("\n=== RL (Design 2) summary ===")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")