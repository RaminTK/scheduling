#This code integrates all the metaheuristics and the RL agent to solve the problem.
# It integrates algorithms to iterate over all the scheduling samaples in a dataset with standard format (ex: jobshop1.txt).


# ===========================
# benchmark_all_methods.py
# ===========================
"""
Run SA, TS, and RL on all JSSP instances in an OR-Library file
(e.g., jobshop1.txt), then build a pandas DataFrame with results.

Expected files in the same folder:
- sa_only.py
- tabu_only.py
- rl_design2.py

Example:
  python benchmark_all_methods.py --file jobshop1.txt
  python benchmark_all_methods.py --file jobshop1.txt --out results.csv
"""

import argparse
import time
import pandas as pd

import sa_only as sa
import tabu_only as ts
import RL_Design2 as rl


# =========================================================
# Read all instance names from jobshop1.txt
# =========================================================
def list_instance_names(filepath: str):
    names = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("instance "):
                parts = line.split()
                if len(parts) >= 2:
                    names.append(parts[1])
    return names


# =========================================================
# Run one instance with all 3 methods
# =========================================================
def run_one_instance(instance_name, file_path, args):
    # Load instance once
    jobs_ops, n_jobs, n_machines = sa.load_jssp_instance(file_path, instance_name)

    row = {
        "instance": instance_name,
        "n_jobs": n_jobs,
        "n_machines": n_machines,
    }

    # -------------------------
    # Simulated Annealing
    # -------------------------
    t0 = time.perf_counter()
    _, sa_best = sa.simulated_annealing(
        jobs_ops,
        seed=args.seed,
        iters=args.sa_iters,
        T0=args.sa_T0,
        alpha=args.sa_alpha,
        neighborhood=args.sa_neigh,
        verbose_every=0,
    )
    row["sa_time"] = time.perf_counter() - t0
    row["sa_makespan"] = sa_best

    # -------------------------
    # Tabu Search
    # -------------------------
    t0 = time.perf_counter()
    _, ts_best = ts.tabu_search(
        jobs_ops,
        seed=args.seed,
        iters=args.ts_iters,
        tenure=args.ts_tenure,
        neighborhood=args.ts_neigh,
        samples=args.ts_samples,
        verbose_every=0,
    )
    row["ts_time"] = time.perf_counter() - t0
    row["ts_makespan"] = ts_best

    # -------------------------
    # RL
    # -------------------------
    t0 = time.perf_counter()
    rl_out = rl.qlearn_run(
        jobs_ops=jobs_ops,
        seed=args.seed,
        steps=args.rl_steps,
        budget=args.rl_budget,
        sa_T0=args.rl_sa_T0,
        sa_alpha=args.rl_sa_alpha,
        sa_neigh=args.rl_sa_neigh,
        tenure=args.rl_tenure,
        tabu_samples=args.rl_tabu_samples,
        tabu_neigh=args.rl_tabu_neigh,
        probe=args.rl_probe,
        probe_neigh=args.rl_probe_neigh,
        eps=args.rl_eps,
        lr=args.rl_lr,
        gamma=args.rl_gamma,
        verbose_every=0,
    )
    row["rl_time"] = time.perf_counter() - t0
    row["rl_makespan"] = rl_out["best_makespan"]

    return row


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    # file / output
    parser.add_argument("--file", type=str, default="jobshop1.txt")
    parser.add_argument("--out", type=str, default="benchmark_results.csv")

    # reproducibility
    parser.add_argument("--seed", type=int, default=1)

    # -------------------------
    # SA settings
    # -------------------------
    parser.add_argument("--sa-iters", type=int, default=30000)
    parser.add_argument("--sa-T0", type=float, default=10.0)
    parser.add_argument("--sa-alpha", type=float, default=0.9999)
    parser.add_argument("--sa-neigh", type=str, default="swap", choices=["swap", "insert"])

    # -------------------------
    # TS settings
    # -------------------------
    parser.add_argument("--ts-iters", type=int, default=2000)
    parser.add_argument("--ts-tenure", type=int, default=10)
    parser.add_argument("--ts-samples", type=int, default=150)
    parser.add_argument("--ts-neigh", type=str, default="swap", choices=["swap", "insert"])

    # -------------------------
    # RL settings
    # -------------------------
    parser.add_argument("--rl-steps", type=int, default=200)
    parser.add_argument("--rl-budget", type=int, default=200)

    parser.add_argument("--rl-sa-T0", type=float, default=10.0)
    parser.add_argument("--rl-sa-alpha", type=float, default=0.9995)
    parser.add_argument("--rl-sa-neigh", type=str, default="swap", choices=["swap", "insert"])

    parser.add_argument("--rl-tenure", type=int, default=10)
    parser.add_argument("--rl-tabu-samples", type=int, default=150)
    parser.add_argument("--rl-tabu-neigh", type=str, default="swap", choices=["swap", "insert"])

    parser.add_argument("--rl-probe", type=int, default=30)
    parser.add_argument("--rl-probe-neigh", type=str, default="swap", choices=["swap", "insert"])

    parser.add_argument("--rl-eps", type=float, default=0.15)
    parser.add_argument("--rl-lr", type=float, default=0.1)
    parser.add_argument("--rl-gamma", type=float, default=0.9)

    args = parser.parse_args()

    instance_names = list_instance_names(args.file)

    print(f"Found {len(instance_names)} instances in {args.file}")
    print("Running SA, TS, and RL on all instances...\n")

    rows = []
    total_start = time.perf_counter()

    for k, name in enumerate(instance_names, start=1):
        print(f"[{k:>2}/{len(instance_names)}] Running instance: {name}")
        try:
            row = run_one_instance(name, args.file, args)
            rows.append(row)

            print(
                f"    SA={row['sa_makespan']} ({row['sa_time']:.2f}s) | "
                f"TS={row['ts_makespan']} ({row['ts_time']:.2f}s) | "
                f"RL={row['rl_makespan']} ({row['rl_time']:.2f}s)"
            )
        except Exception as e:
            print(f"    ERROR on {name}: {e}")
            rows.append({
                "instance": name,
                "n_jobs": None,
                "n_machines": None,
                "sa_time": None,
                "sa_makespan": None,
                "ts_time": None,
                "ts_makespan": None,
                "rl_time": None,
                "rl_makespan": None,
            })

    df = pd.DataFrame(rows)

    # Optional: keep instance as the index
    # df = df.set_index("instance")

    print("\n=== Final DataFrame ===")
    print(df)

    df.to_csv(args.out, index=False)
    print(f"\nSaved results to: {args.out}")
    print(f"Total elapsed time: {time.perf_counter() - total_start:.2f} seconds")


if __name__ == "__main__":
    main()
