# ===========================
# sa_only.py
# ===========================
"""
Simulated Annealing baseline for OR-Library JSSP instances in jobshop1.txt.

Run:
  python sa_only.py ft06
  python sa_only.py abz7 --iters 30000 --T0 30 --alpha 0.999 --neigh swap
  python sa_only.py la01 --seed 7

If your jobshop1.txt is somewhere else:
  python sa_only.py ft06 --file "/full/path/to/jobshop1.txt"
"""

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple
import time


# =========================================================
# OR-Library parser (jobshop1.txt format)
# =========================================================
def load_jssp_instance(filepath: str, instance_name: str):
    with open(filepath, "r") as f:
        raw_lines = [ln.rstrip("\n") for ln in f]
    lines = [l.strip() for l in raw_lines if l.strip()]

    target = instance_name.lower().strip()

    idx = None
    for i, line in enumerate(lines):
        if line.lower().startswith("instance "):
            parts = line.lower().split()
            if len(parts) >= 2 and parts[1] == target:
                idx = i
                break

    if idx is None:
        available = []
        for line in lines:
            if line.lower().startswith("instance "):
                parts = line.split()
                if len(parts) >= 2:
                    available.append(parts[1])
        raise ValueError(
            f"Instance '{instance_name}' not found in {filepath}. "
            f"Example available: {', '.join(available[:15])} ..."
        )

    dims = lines[idx + 3].split()
    n_jobs = int(dims[0])
    n_machines = int(dims[1])

    jobs_ops: List[List[Tuple[int, int]]] = []
    cursor = idx + 4
    for _ in range(n_jobs):
        row = list(map(int, lines[cursor].split()))
        cursor += 1
        if len(row) != 2 * n_machines:
            raise ValueError(
                f"Row length mismatch for {instance_name}: expected {2*n_machines} ints, got {len(row)}"
            )
        jobs_ops.append([(row[k], row[k + 1]) for k in range(0, len(row), 2)])

    for job in jobs_ops:
        for m, p in job:
            if not (0 <= m < n_machines):
                raise ValueError(f"Machine index out of range: {m} (0..{n_machines-1})")
            if p <= 0:
                raise ValueError(f"Non-positive processing time: {p}")

    return jobs_ops, n_jobs, n_machines

# =========================================================
# Decoder: priority list -> feasible schedule -> makespan
# =========================================================
def decode_priority_list(jobs_ops: List[List[Tuple[int, int]]], priority: List[int]) -> int:
    n = len(jobs_ops)
    m = len(jobs_ops[0])

    next_op = [0] * n
    machine_ready = [0] * m
    job_ready = [0] * n

    for j in priority:
        k = next_op[j]
        if k >= m:
            continue
        mach, p = jobs_ops[j][k]
        start = max(job_ready[j], machine_ready[mach])
        finish = start + p
        job_ready[j] = finish
        machine_ready[mach] = finish
        next_op[j] += 1

    return max(job_ready)

def random_priority(n_jobs: int, n_machines: int, rng: random.Random) -> List[int]:
    seq = []
    for j in range(n_jobs):
        seq += [j] * n_machines
    rng.shuffle(seq)
    return seq

# =========================================================
# Neighborhood operators
# =========================================================
def move_swap(seq: List[int], i: int, j: int) -> List[int]:
    new = seq[:]
    new[i], new[j] = new[j], new[i]
    return new

def move_insert(seq: List[int], i: int, j: int) -> List[int]:
    new = seq[:]
    x = new.pop(i)
    new.insert(j, x)
    return new

# =========================================================
# Simulated Annealing
# =========================================================
@dataclass
class SAConfig:
    T0: float = 10.0
    alpha: float = 0.9999
    neighborhood: str = "swap"   # "swap" or "insert"

def simulated_annealing(jobs_ops,
                        seed: int = 1,
                        iters: int = 30000,
                        T0: float = 10.0,
                        alpha: float = 0.9999,
                        neighborhood: str = "swap",
                        verbose_every: int = 1000):
    rng = random.Random(seed)
    n_jobs = len(jobs_ops)
    n_machines = len(jobs_ops[0])
    n = n_jobs * n_machines

    cfg = SAConfig(T0=T0, alpha=alpha, neighborhood=neighborhood)

    curr_seq = random_priority(n_jobs, n_machines, rng)
    curr_val = decode_priority_list(jobs_ops, curr_seq)
    best_seq = curr_seq
    best_val = curr_val

    T = cfg.T0

    for t in range(iters):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            j = (j + 1) % n

        cand = move_swap(curr_seq, i, j) if cfg.neighborhood == "swap" else move_insert(curr_seq, i, j)
        cand_val = decode_priority_list(jobs_ops, cand)
        delta = cand_val - curr_val  # minimization

        if delta <= 0:
            curr_seq, curr_val = cand, cand_val
        else:
            if T > 1e-12 and rng.random() < math.exp(-delta / T):
                curr_seq, curr_val = cand, cand_val

        if curr_val < best_val:
            best_seq, best_val = curr_seq, curr_val

        T *= cfg.alpha

        if verbose_every and ((t + 1) % verbose_every == 0 or t == 0):
            print(f"SA iter {t+1:6d}/{iters} | current={curr_val} | best={best_val} | T={T:.6f}")

    return best_seq, best_val

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=str, help="e.g., ft06, abz7, la01", default="ft06")
    parser.add_argument(
        "--file",
        type=str,
        default="jobshop1.txt",
        #default="/Users/raminkhameneh/Library/CloudStorage/OneDrive-stevens.edu/Scheduling Project/Data/jobshop1.txt",
        help="path to OR-Library jobshop1.txt"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iters", type=int, default=30000)
    parser.add_argument("--T0", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.9999)
    parser.add_argument("--neigh", type=str, default="swap", choices=["swap", "insert"])
    parser.add_argument("--verbose-every", type=int, default=1000)
    args = parser.parse_args()

    jobs_ops, n_jobs, n_machines = load_jssp_instance(args.file, args.instance)
    print(f"Loaded {args.instance}: jobs={n_jobs}, machines={n_machines}")

    _, best_ms = simulated_annealing(
        jobs_ops,
        seed=args.seed,
        iters=args.iters,
        T0=args.T0,
        alpha=args.alpha,
        neighborhood=args.neigh,
        verbose_every=args.verbose_every,
    )
    print(f"\nSA best makespan found: {best_ms}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()