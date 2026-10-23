#!/usr/bin/env python3
"""Rerun the bullet b1 / 32K experiments with the calibrated SM bandwidth model.

Each run reuses the command of its original runs/<model>/bullet<P>_tp8_pp1_b1_c32k
and adds --sm-bw-scaling-factor auto (config/SmBandwidthModel.h), writing to
runs/<model>/bullet<P>_cal_tp8_pp1_b1_c32k so the old result stays alongside.
Rows are appended to results/<model>_summary.csv in the same columns.

Usage:  scripts/run_bullet_cal.py [-j 9]
"""
import argparse
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {"gpt-175b": ("GPT-175B", 96), "llama2-70b": ("LLAMA2-70B", 80), "qwen-235b": ("Qwen-235B", 94)}
SPLITS = [74, 104, 133]


def last(pattern, text):
    m = re.findall(pattern, text)
    return m[-1] if m else ""


def run(job):
    model_dir, prefill_sm = job
    name, ndec = MODELS[model_dir]
    old = os.path.join(REPO, "runs", model_dir, f"bullet{prefill_sm}_tp8_pp1_b1_c32k")
    tag = f"bullet{prefill_sm}_cal_tp8_pp1_b1_c32k"
    out = os.path.join(REPO, "runs", model_dir, tag)
    os.makedirs(out, exist_ok=True)
    row_path = os.path.join(out, "row.csv")
    if os.path.exists(row_path):
        line = open(row_path).read().strip()
        if line.endswith(",ok"):
            print(f"[skip] {name}/{tag} (already finished)", flush=True)
            return model_dir, line

    cmd = open(os.path.join(old, "cmd.txt")).read().split() + ["--sm-bw-scaling-factor", "auto"]
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write(" ".join(cmd) + "\n")
    # The DRAM ini is opened as ../configs/..., i.e. runs/<model>/configs.
    t0 = time.time()
    with open(os.path.join(out, "run.log"), "w") as log:
        rc = subprocess.call(cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT)
    wall = int(time.time() - t0)

    text = open(os.path.join(out, "run.log")).read()
    prefill = last(r"GPU model done at cycle (\d+)", text)
    decode = last(r"avg cycles/step: ([\d.]+)", text)
    mem = last(r"Total memory requests pushed: (\d+)", text)
    ok = rc == 0 and prefill
    ms = lambda ns: f"{float(ns) * ndec / 1e6:.4f}" if ns else ""
    row = [name, tag, "bullet_cal", 8, 1, 1, 32768, ndec, prefill, ms(prefill), decode, ms(decode),
           "", "", "", 0, mem, wall, "ok" if ok else f"fail({rc})"]
    line = ",".join(map(str, row))
    with open(os.path.join(out, "row.csv"), "w") as f:
        f.write(line + "\n")
    print(f"[done] {name}/{tag} {wall}s {row[-1]}", flush=True)
    return model_dir, line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", type=int, default=len(MODELS) * len(SPLITS))
    a = ap.parse_args()
    jobs = [(m, p) for m in MODELS for p in SPLITS]
    with ThreadPoolExecutor(a.j) as pool:
        rows = list(pool.map(run, jobs))
    # Replace a row with the same tag, so a rerun never duplicates one.
    for model_dir, line in rows:
        path = os.path.join(REPO, "results", f"{model_dir}_summary.csv")
        tag = line.split(",")[1]
        kept = [l for l in open(path).read().splitlines() if l.split(",")[1:2] != [tag]]
        with open(path, "w") as f:
            f.write("\n".join(kept + [line]) + "\n")


if __name__ == "__main__":
    main()
