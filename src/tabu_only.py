# ===========================
# tabu_only.py
# ===========================
"""
Tabu Search baseline for OR-Library JSSP instances in jobshop1.txt.

Run:
  python tabu_only.py ft06
  python tabu_only.py abz7 --iters 5000 --tenure 15 --samples 200 --neigh insert
  python tabu_only.py la01 --seed 3

If your jobshop1.txt is somewhere else:
  python tabu_only.py ft06 --file "/full/path/to/jobshop1.txt"
"""

import argparse
import random
from collections import deque
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

    # idx: "instance name"
    # idx+1: separators
    # idx+2: description
    # idx+3: "n_jobs n_machines"
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

    # sanity checks
    for job in jobs_ops:
        if len(job) != n_machines:
            raise ValueError("Bad instance parse: wrong operations per job.")
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
# Tabu Search
# =========================================================
@dataclass
class TabuConfig:
    tenure: int = 10
    neighborhood: str = "swap"   # "swap" or "insert"
    samples: int = 150           # sampled neighbors per iteration

def tabu_search(jobs_ops,
                seed: int = 1,
                iters: int = 2000,
                tenure: int = 10,
                neighborhood: str = "swap",
                samples: int = 150,
                verbose_every: int = 200):
    rng = random.Random(seed)
    n_jobs = len(jobs_ops)
    n_machines = len(jobs_ops[0])
    n = n_jobs * n_machines

    cfg = TabuConfig(tenure=tenure, neighborhood=neighborhood, samples=samples)

    curr_seq = random_priority(n_jobs, n_machines, rng)
    curr_val = decode_priority_list(jobs_ops, curr_seq)

    best_seq = curr_seq
    best_val = curr_val

    tabu = deque()

    for t in range(iters):
        cand_best = None
        cand_best_val = None
        cand_move = None

        for _ in range(cfg.samples):
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue

            move_sig = (min(i, j), max(i, j), cfg.neighborhood)
            if move_sig in tabu:
                continue

            cand = move_swap(curr_seq, i, j) if cfg.neighborhood == "swap" else move_insert(curr_seq, i, j)
            val = decode_priority_list(jobs_ops, cand)

            if cand_best_val is None or val < cand_best_val:
                cand_best = cand
                cand_best_val = val
                cand_move = move_sig

        # fallback if all sampled moves tabu (rare)
        if cand_best is None:
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                j = (j + 1) % n
            cand_best = move_swap(curr_seq, i, j)
            cand_best_val = decode_priority_list(jobs_ops, cand_best)
            cand_move = (min(i, j), max(i, j), cfg.neighborhood)

        curr_seq, curr_val = cand_best, cand_best_val

        tabu.append(cand_move)
        while len(tabu) > cfg.tenure:
            tabu.popleft()

        if curr_val < best_val:
            best_seq, best_val = curr_seq, curr_val

        if verbose_every and ((t + 1) % verbose_every == 0 or t == 0):
            print(f"TS iter {t+1:6d}/{iters} | current={curr_val} | best={best_val} | tabu={len(tabu)}")

    return best_seq, best_val

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("instance",default="ft06", type=str, help="e.g., ft06, abz7, la01")
    parser.add_argument(
        "--file",
        type=str,
        default="jobshop1.txt",
        #default="/Users/raminkhameneh/Library/CloudStorage/OneDrive-stevens.edu/Scheduling Project/Data/jobshop1.txt",
        help="path to OR-Library jobshop1.txt"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--tenure", type=int, default=10)
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--neigh", type=str, default="swap", choices=["swap", "insert"])
    parser.add_argument("--verbose-every", type=int, default=200)
    args = parser.parse_args()

    jobs_ops, n_jobs, n_machines = load_jssp_instance(args.file, args.instance)
    print(f"Loaded {args.instance}: jobs={n_jobs}, machines={n_machines}")

    _, best_ms = tabu_search(
        jobs_ops,
        seed=args.seed,
        iters=args.iters,
        tenure=args.tenure,
        neighborhood=args.neigh,
        samples=args.samples,
        verbose_every=args.verbose_every,
    )
    print(f"\nTS best makespan found: {best_ms}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()