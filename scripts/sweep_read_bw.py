#!/usr/bin/env python3
"""Read bandwidth vs. SM count on bullet's hardware (HBM3e, 16 channels).

Runs ndp_sim --read-bw-test, where every GPU instance only streams reads, in
two sweeps:

  standalone  one GPU with --sm-count N, nothing else touching DRAM: the
              bandwidth an N-SM group can pull on its own.
  bullet      --bullet with --bullet-prefill-sm P: the prefill and decode
              halves (P and total-P SMs) reading at the same time.

With --b200 CSV (columns sm_count, <mode>_GBps for direct / tma_s2 / tma_s4 /
tma_s8, measured on a full 8-stack B200), it also writes calibration.csv:
the simulator's standalone bandwidth at each measured SM count against the
best measured mode divided by 8, since the bullet hardware models one stack.

Usage:  scripts/sweep_read_bw.py [--build build] [--ns 50000] [--out results/read_bw]
                                 [--sm-bw-scaling-factor auto|N] [--b200 CSV]
"""
import argparse
import csv
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

LINE = re.compile(r"\[Read BW\] (\w+): SMs=(\d+) max_outstanding=(\d+) bytes=\d+ BW=([\d.]+) GB/s")
FACTOR = re.compile(r"\((\d+) SMs \* (\d+) per SM\)")
B200_MODES = ["direct", "tma_s2", "tma_s4", "tma_s8"]
B200_STACKS = 8


def run(sim, build, args):
    out = subprocess.run([sim, *args], cwd=build, stdout=subprocess.PIPE,
                         universal_newlines=True, check=True).stdout
    res = {m[0]: (int(m[1]), int(m[2]), float(m[3])) for m in LINE.findall(out)}
    res["factors"] = [int(m[1]) for m in FACTOR.findall(out)]
    return res


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=os.path.join(repo, "build"))
    ap.add_argument("--ns", type=int, default=50000, help="measurement window per run, in ns")
    ap.add_argument("--total-sm", type=int, default=148)
    ap.add_argument("--sm-bw-scaling-factor", default="auto")
    ap.add_argument("--b200", help="measured B200 bandwidth by SM count, for calibration.csv")
    ap.add_argument("--out", default=os.path.join(repo, "results", "read_bw"))
    ap.add_argument("-j", type=int, default=os.cpu_count())
    a = ap.parse_args()

    sim = os.path.join(a.build, "simulator", "ndp_sim")
    common = ["--read-bw-test", str(a.ns), "--total-sm-count", str(a.total_sm),
              "--sm-bw-scaling-factor", a.sm_bw_scaling_factor,
              "--ndp-config", os.path.join(repo, "simulator", "ndp_config.json")]
    b200 = list(csv.DictReader(open(a.b200))) if a.b200 else []
    sms = sorted(set([1, 2, 4] + list(range(8, a.total_sm, 8)) + [a.total_sm - 1]
                     + [int(r["sm_count"]) for r in b200]))
    os.makedirs(a.out, exist_ok=True)

    with ThreadPoolExecutor(a.j) as pool:
        # 0 = no SM masking, i.e. no in-flight cap: the DRAM model's own ceiling.
        standalone = dict(pool.map(
            lambda n: (n, run(sim, a.build, common + ["--bullet-hw", "--sm-count", str(n)])),
            sms + [a.total_sm, 0]))
        bullet = list(pool.map(
            lambda p: (p, run(sim, a.build, common + ["--bullet", "--bullet-prefill-sm", str(p)])),
            [p for p in sms if 0 < p < a.total_sm]))

    path = os.path.join(a.out, "standalone.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sm_count", "max_outstanding", "read_bw_GBps"])
        for n, r in sorted(standalone.items(), key=lambda kv: kv[0] or 1 << 30):
            _, mo, bw = r["gpu"]
            w.writerow([n if n else "unlimited", mo, f"{bw:.2f}"])
    print(f"wrote {path}")

    path = os.path.join(a.out, "bullet.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prefill_sm", "decode_sm", "prefill_factor", "decode_factor",
                    "prefill_bw_GBps", "decode_bw_GBps", "total_bw_GBps"])
        for p, r in bullet:
            pf, dc = r["prefill"][2], r["decode"][2]
            w.writerow([p, a.total_sm - p, *r["factors"], f"{pf:.2f}", f"{dc:.2f}", f"{pf + dc:.2f}"])
    print(f"wrote {path}")

    if b200:
        path = os.path.join(a.out, "calibration.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sm_count", *[f"b200_{m}_per_stack_GBps" for m in B200_MODES],
                        "b200_best_mode", "b200_best_per_stack_GBps",
                        "sim_factor", "sim_max_outstanding", "sim_read_bw_GBps", "error_pct"])
            for row in b200:
                n = int(row["sm_count"])
                per_stack = {m: float(row[f"{m}_GBps"]) / B200_STACKS for m in B200_MODES}
                best = max(per_stack, key=per_stack.get)
                r = standalone[n]
                _, mo, bw = r["gpu"]
                err = 100 * (bw - per_stack[best]) / per_stack[best]
                w.writerow([n, *[f"{per_stack[m]:.2f}" for m in B200_MODES], best,
                            f"{per_stack[best]:.2f}", r["factors"][0], mo, f"{bw:.2f}", f"{err:+.2f}"])
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
