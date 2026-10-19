#!/usr/bin/env python3
"""Sweep QPS for queueing_estimator.py across one or more calibration JSON files.

Contract
- Inputs: one or more --calib files and an --qps list or range.
- Action: runs scripts/queueing_estimator.py for each (calib, qps) combination.
- Output: prints a compact CSV-like table to stdout and optionally writes a summary CSV.

Design notes
- This script is intentionally lightweight and relies on queueing_estimator.py as the
  source of truth.
- It passes through extra flags to queueing_estimator.py via `--`.

Example (matches README-style commands)

  python3 scripts/sweep_qps_queueing_estimator.py \
    --calib scripts/_tmp_calib.json scripts/_tmp_calib2.json \
    --qps 0.5 1 2 3 4 5 \
    --n 1000 --max-batch 32 --decode-max-batch 32 --decode-window-ms 0.01 \
    --lout-mu 128 --lout-sigma 32 --decode-scales-with-context \
    --input-len-mu 8192 --input-len-sigma 2048 --seed 1 --ttft-includes-prefill \
    --num-devices 4 --enable-duoPD \
    -- --log-prefix sweep_out

If you want a range:

  python3 scripts/sweep_qps_queueing_estimator.py --calib scripts/_tmp_calib.json \
    --qps-range 0.5 10 0.5 -- ...

"""

import argparse
import csv
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


def _parse_qps_list(vals: List[str]) -> List[float]:
    out: List[float] = []
    for v in vals:
        out.append(float(v))
    return out


def _parse_qps_range(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    out: List[float] = []
    x = float(start)
    # inclusive-ish stop to be convenient for sweeps
    while x <= float(stop) + 1e-12:
        out.append(float(x))
        x += float(step)
    return out


def _run_one(argv: List[str]) -> str:
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out_b, _ = proc.communicate()
    out = out_b.decode("utf-8", errors="replace") if isinstance(out_b, (bytes, bytearray)) else str(out_b)
    if proc.returncode != 0:
        raise RuntimeError("command failed ({}): {}\n{}".format(proc.returncode, " ".join(argv), out))
    return out.strip()


def _extract_metrics(line: str) -> Dict[str, float]:
    # Expected format from queueing_estimator.py is:
    # qps=... n=... max_batch=... p50_TTFT_ms=... ... p50_TBT_ms=...
    # We'll parse key=value pairs.
    parts = line.replace("\n", " ").split()
    out: Dict[str, float] = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        # strip commas/spurious
        v = v.strip().strip(",")
        try:
            out[k] = float(v)
        except ValueError:
            # allow n=... as float too; ignore anything else
            continue
    return out


def _default_log_name(prefix: str, calib_path: str, qps: float) -> str:
    base = os.path.splitext(os.path.basename(calib_path))[0]
    return f"{prefix}_{base}_qps{qps:g}.csv"


_DUOPD_CALIB_BASENAMES = {
    "duopd.json",
    "duopd_on_duplex.json",
    "duopd_sm.json",
}

_SM_RATIO_CALIB_BASENAMES = {
    "sm_9:1.json",
    "sm_7:3.json",
    "sm_5:5.json",
}


def _is_duopd_calib(calib_path: str) -> bool:
    """Heuristic: enable DuoPD only for DuoPD calibration JSONs.

    We intentionally keep this conservative (basename match) to avoid accidentally
    enabling DuoPD for other calibration files.
    """
    base = os.path.basename(calib_path).lower()
    return base in _DUOPD_CALIB_BASENAMES


def _default_num_devices_for_calib(calib_path: str) -> int:
    """Heuristic defaults for --num-devices based on known calib filenames.

    This is meant to make mixed sweeps less error-prone.
    - PD_1_1.json -> 4 devices
    - PD_3_1.json -> 2 devices
    - else -> 1 device
    """
    base = os.path.basename(calib_path).lower()
    if base == "pd_1_1.json":
        return 4
    if base == "pd_3_1.json":
        return 2
    return 1


def _is_pd_3_1_calib(calib_path: str) -> bool:
    base = os.path.basename(calib_path)
    return base.lower() == "pd_3_1.json"


def _is_pd_duplex_calib(calib_path: str) -> bool:
    base = os.path.basename(calib_path).lower()
    return base in ("pd_1_1_duplex.json", "pd_3_1_duplex.json")


def _is_chunked_calib(calib_path: str) -> bool:
    """Heuristic: enable chunked prefill only for Chunked calibration JSONs."""
    base = os.path.basename(calib_path)
    return base.lower() == "chunked.json"


def _is_sm_ratio_calib(calib_path: str) -> bool:
    base = os.path.basename(calib_path).lower()
    return base in _SM_RATIO_CALIB_BASENAMES


def _default_gqa_size_for_model(model: str) -> int:
    m = (model or "").strip().lower()
    if m in ("gpt", "gpt3", "gpt-3"):
        return 1
    if m in ("llama", "llama2", "llama-2"):
        return 8
    if m in ("qwen", "qwen2"):
        return 16
    raise ValueError("Unknown model preset: {} (expected GPT, LLAMA, or QWEN)".format(model))


def _default_prefill_hidden_dim_for_model(model: str) -> int:
    m = (model or "").strip().lower()
    if m in ("gpt", "gpt3", "gpt-3"):
        return 12288
    if m in ("llama", "llama2", "llama-2"):
        return 8192
    if m in ("qwen", "qwen2"):
        return 4096
    raise ValueError("Unknown model preset: {} (expected GPT, LLAMA, or QWEN)".format(model))


def _default_prefill_chunk_tokens_for_calib(calib_path: str, model: str = None) -> int:
    m = (model or "").strip().lower()
    if m in ("qwen", "qwen2"):
        return 4096

    return 1024


def _default_prefill_base_n_for_calib(calib_path: str, model: str = None) -> int:
    # Non-chunked default policy only.
    m = (model or "").strip().lower()
    if m in ("qwen", "qwen2"):
        return 32768

    return 8192


def _default_prefill_base_n_for_chunked_calib(calib_path: str, model: str = None) -> int:
    m = (model or "").strip().lower()
    if m in ("qwen", "qwen2"):
        return 4096

    return 1024


def _filter_passthrough_for_calib(passthrough: List[str], calib_path: str, *, model: str = None) -> List[str]:
    """Filter global passthrough args for a specific calib.

    Today this special-cases:
    - If the user passed it globally, only keep it for DuoPD calib files.
    - If the calib is DuoPD and the user did NOT pass it, we may auto-add it.

    And `--num-devices`:
    - If the user did not pass --num-devices, inject a sensible default per calib.
    - If the user *did* pass it, keep it as-is.

    The rest of passthrough args are preserved as-is.
    """
    out: List[str] = []
    is_duopd = _is_duopd_calib(calib_path)
    is_chunked = _is_chunked_calib(calib_path)
    is_sm_ratio = _is_sm_ratio_calib(calib_path)
    is_pd_3_1 = _is_pd_3_1_calib(calib_path)
    is_pd_duplex = _is_pd_duplex_calib(calib_path)
    model_l = (model or "").strip().lower()
    saw_num_devices = False
    saw_chunked_prefill = False
    saw_prefill_chunk_tokens = False
    saw_pd_duplex = False
    saw_gqa_size = False
    saw_prefill_base_n = False
    saw_skip_ttft_kv_transfer_penalty = False
    # Batch policy:
    # - Chunked.json -> --max-batch 32
    # - others -> --max-batch 1
    # And decode max batch is always 32.
    # We treat these as per-sweep policy knobs and override any global passthrough.
    skip_next = False
    for a in passthrough:
        if skip_next:
            skip_next = False
            continue

        if a == "--enable-duoPD" and not is_duopd:
            continue

        # Strip any global batch knobs; we'll re-inject per calib below.
        if a in ("--max-batch", "--decode-max-batch"):
            skip_next = True
            continue

        if a == "--num-devices":
            saw_num_devices = True
        if a == "--chunked-prefill":
            saw_chunked_prefill = True
        if a == "--prefill-chunk-tokens":
            saw_prefill_chunk_tokens = True
        if a == "--pd-duplex":
            saw_pd_duplex = True
        if a == "--gqa-size":
            saw_gqa_size = True
        if a == "--prefill-base-n":
            saw_prefill_base_n = True
        if a == "--skip-ttft-kv-transfer-penalty":
            saw_skip_ttft_kv_transfer_penalty = True
        out.append(a)

    if not saw_num_devices:
        out += ["--num-devices", str(_default_num_devices_for_calib(calib_path))]

    # Enforce per-calib batching policy.
    if is_chunked:
        out += ["--max-batch", "32"]
    else:
        out += ["--max-batch", "1"]
    # Decode max batch policy: PD_3_1.json uses 8, all others use 32.
    out += ["--decode-max-batch", "2" if is_pd_3_1 else "8"]

    # Non-chunked default for prefill-base-n.
    # Chunked mode keeps its original base_n default (1024) below.

    # Chunked-prefill defaults for Chunked.json.
    # - If user already provided these flags globally, we respect them.
    # - Otherwise, we auto-enable chunked prefill and set chunk size based on model preset.
    if is_chunked:
        if not saw_chunked_prefill:
            out += ["--chunked-prefill"]
        if not saw_prefill_chunk_tokens:
            out += ["--prefill-chunk-tokens", str(_default_prefill_chunk_tokens_for_calib(calib_path, model=model))]
        if not saw_prefill_base_n:
            out += ["--prefill-base-n", str(_default_prefill_base_n_for_chunked_calib(calib_path, model=model))]

    # PD_duplex auto-enable policy: *_duplex.json implies --pd-duplex unless user already set it.
    if is_pd_duplex and not saw_pd_duplex:
        out += ["--pd-duplex"]

    # SM ratio policy: do NOT enable DuoPD, and also skip TTFT KV-transfer penalty.
    if is_sm_ratio and not saw_skip_ttft_kv_transfer_penalty:
        out += ["--skip-ttft-kv-transfer-penalty"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)

    ap.add_argument(
        "--model",
        default=None,
        choices=["GPT", "LLAMA", "QWEN", "gpt", "llama", "qwen"],
        help=(
            "Model preset for sweep convenience. Sets a default --gqa-size for all runs unless you explicitly pass "
            "--gqa-size. GPT->1, LLAMA->8, QWEN->16."
        ),
    )
    ap.add_argument(
        "--calib",
        nargs="+",
        required=True,
        help="One or more calibration JSON files.",
    )

    qps_group = ap.add_mutually_exclusive_group(required=True)
    qps_group.add_argument("--qps", nargs="+", help="List of qps values")
    qps_group.add_argument(
        "--qps-range",
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        help="Range sweep: start stop step",
    )

    ap.add_argument(
        "--out-csv",
        default=None,
        help="Optional summary CSV path (one row per run).",
    )

    ap.add_argument(
        "--plot-out-dir",
        default=None,
        help=(
            "If set, automatically generate plots into this directory after the sweep finishes. "
            "Requires --out-csv."
        ),
    )
    ap.add_argument(
        "--plot-ttft-ymax-ms",
        type=float,
        default=2000.0,
        help="Y-axis max (ms) for TTFT plot when --plot-out-dir is used. Default: 2000.",
    )
    ap.add_argument(
        "--plot-e2e-ymax-ms",
        type=float,
        default=2000.0,
        help="Y-axis max (ms) for E2E plot when --plot-out-dir is used. Default: 2000.",
    )
    ap.add_argument(
        "--plot-ttft-norm-ymax",
        type=float,
        default=10.0,
        help=(
            "Optional y-axis max (ratio) for normalized TTFT plots when --normalize-to-duopd-first is used. "
            "Default: 10."
        ),
    )
    ap.add_argument(
        "--plot-e2e-norm-ymax",
        type=float,
        default=10.0,
        help=(
            "Optional y-axis max (ratio) for normalized E2E plots when --normalize-to-duopd-first is used. "
            "Default: 10."
        ),
    )
    ap.add_argument(
        "--plot-tbt-norm-ymax",
        type=float,
        default=10.0,
        help=(
            "Optional y-axis max (ratio) for normalized TBT plots when --normalize-to-duopd-first is used. "
            "Default: 10."
        ),
    )
    ap.add_argument(
        "--plot-tpot-norm-ymax",
        type=float,
        default=None,
        help=(
            "Optional y-axis max (ratio) for normalized TPOT plots when --normalize-to-duopd-first is used. "
            "Default: 10."
        ),
    )
    ap.add_argument(
        "--plot-tpot-ymax-ms",
        type=float,
        default=None,
        help="Optional y-axis max (ms) for TPOT plot when --plot-out-dir is used. Default: uncapped.",
    )
    ap.add_argument(
        "--plot-title",
        default=None,
        help="Optional title prefix used in plots when --plot-out-dir is used.",
    )
    ap.add_argument(
        "--plot-split-by-percentile",
        action="store_true",
        help=(
            "If set (and --plot-out-dir is used), plot p50/p90/p99 into separate files, "
            "overlaying all calibration curves in each file."
        ),
    )
    ap.add_argument(
        "--normalize-to-duopd-first",
        action="store_true",
        help=(
            "If set (and --plot-out-dir is used), also generate normalized latency plots "
            "where each series is divided by the first DuoPD point (smallest QPS), per percentile."
        ),
    )
    ap.add_argument(
        "--log-prefix",
        default=None,
        help=(
            "If set, will pass --log <prefix>_<calib>_qpsX.csv to queueing_estimator.py. "
            "This is useful when you also want per-request/per-token logs."
        ),
    )

    # Back-compat + ergonomics:
    # - Preferred: pass estimator args after an explicit `--`.
    # - Still supported: pass estimator args directly (without `--`), e.g. `--n 1000`.
    #
    # IMPORTANT: argparse's `parse_known_args()` is not sufficient here because some
    # sweep options (like --qps / --calib) take variable-length lists; that can cause
    # unknown estimator flags to be swallowed as additional list values.
    #
    # Instead we:
    #  1) Parse once normally to get sweep args (this will work for well-formed sweeps).
    #  2) Build passthrough by filtering sweep-known tokens out of the original argv,
    #     leaving all remaining tokens intact (including unknown flags + their values).
    argv = sys.argv[1:]
    if "--" in argv:
        idx = argv.index("--")
        sweep_argv = argv[:idx]
        passthrough = argv[idx + 1 :]
        args = ap.parse_args(sweep_argv)
    else:
        args = ap.parse_args(argv)

        # Remove sweep args from argv while preserving order.
        # Known sweep flags and how many values they consume.
        sweep_flags_fixed_arity = {
            "--model": 1,
            "--out-csv": 1,
            "--plot-out-dir": 1,
            "--plot-ttft-ymax-ms": 1,
            "--plot-e2e-ymax-ms": 1,
            "--plot-ttft-norm-ymax": 1,
            "--plot-e2e-norm-ymax": 1,
            "--plot-tbt-norm-ymax": 1,
            "--plot-tpot-norm-ymax": 1,
            "--plot-tpot-ymax-ms": 1,
            "--plot-title": 1,
            "--log-prefix": 1,
        }
        sweep_flags_no_value = {
            "--plot-split-by-percentile",
            "--normalize-to-duopd-first",
        }

        passthrough = []
        i = 0
        while i < len(argv):
            tok = argv[i]

            if tok == "--calib":
                i += 1
                # Consume calib paths until next flag (or end).
                while i < len(argv) and not argv[i].startswith("-"):
                    i += 1
                continue

            if tok == "--qps":
                i += 1
                while i < len(argv) and not argv[i].startswith("-"):
                    i += 1
                continue

            if tok == "--qps-range":
                i += 4  # flag + 3 values
                continue

            if tok in sweep_flags_fixed_arity:
                i += 1 + sweep_flags_fixed_arity[tok]
                continue

            if tok in sweep_flags_no_value:
                i += 1
                continue

            # Anything else is estimator passthrough.
            passthrough.append(tok)
            i += 1

    # Apply model preset -> default gqa-size unless user already provided --gqa-size.
    if args.model:
        if "--gqa-size" not in passthrough:
            passthrough += ["--gqa-size", str(_default_gqa_size_for_model(args.model))]
        if "--prefill-hidden-dim" not in passthrough:
            passthrough += [
                "--prefill-hidden-dim",
                str(_default_prefill_hidden_dim_for_model(args.model)),
            ]

    if args.qps is not None:
        qps_list = _parse_qps_list(args.qps)
    else:
        start, stop, step = (float(args.qps_range[0]), float(args.qps_range[1]), float(args.qps_range[2]))
        qps_list = _parse_qps_range(start, stop, step)

    # Prepare output
    rows: List[Dict[str, object]] = []

    # Table header for stdout
    print(
    "calib,qps,enable_duoPD,pd_duplex,num_devices,"
    "p50_ttft_ms,p90_ttft_ms,p99_ttft_ms,p50_e2e_ms,p90_e2e_ms,p99_e2e_ms,p50_tbt_ms,p90_tbt_ms,p99_tbt_ms,"
    "p50_tpot_ms,p90_tpot_ms,p99_tpot_ms,"
    "throughput_qps,throughput_tps"
    )

    for calib in args.calib:
        for qps in qps_list:
            calib_passthrough = _filter_passthrough_for_calib(passthrough, calib, model=args.model)
            cmd = [sys.executable, os.path.join("scripts", "queueing_estimator.py"), "--qps", str(qps), "--calib", calib]
            if args.log_prefix:
                cmd += ["--log", _default_log_name(args.log_prefix, calib, qps)]

            # Auto-enable DuoPD only for DuoPD calibration JSONs.
            # This lets you sweep across mixed calibs without manually toggling per-calib flags.
            if _is_duopd_calib(calib) and "--enable-duoPD" not in calib_passthrough:
                cmd += ["--enable-duoPD"]
            if calib_passthrough:
                cmd += calib_passthrough

            out_line = _run_one(cmd).splitlines()[-1]
            metrics = _extract_metrics(out_line)

            row = {
                "calib": calib,
                "qps": qps,
                "enable_duoPD": int(metrics.get("enable_duoPD", 0.0)),
                "pd_duplex": int(metrics.get("pd_duplex", 0.0)),
                "num_devices": int(metrics.get("num_devices", 0.0)),
                "p50_ttft_ms": metrics.get("p50_TTFT_ms", float("nan")),
                "p90_ttft_ms": metrics.get("p90_TTFT_ms", float("nan")),
                "p99_ttft_ms": metrics.get("p99_TTFT_ms", float("nan")),
                "p50_e2e_ms": metrics.get("p50_E2E_ms", float("nan")),
                "p90_e2e_ms": metrics.get("p90_E2E_ms", float("nan")),
                "p99_e2e_ms": metrics.get("p99_E2E_ms", float("nan")),
                "p50_tbt_ms": metrics.get("p50_TBT_ms", float("nan")),
                "p90_tbt_ms": metrics.get("p90_TBT_ms", float("nan")),
                "p99_tbt_ms": metrics.get("p99_TBT_ms", float("nan")),
                "p50_tpot_ms": metrics.get("p50_TPOT_ms", float("nan")),
                "p90_tpot_ms": metrics.get("p90_TPOT_ms", float("nan")),
                "p99_tpot_ms": metrics.get("p99_TPOT_ms", float("nan")),
                "throughput_qps": metrics.get("throughput_qps", float("nan")),
                "throughput_tps": metrics.get("throughput_tps", float("nan")),
            }
            rows.append(row)

            print(
                f"{calib},{qps:g},{row['enable_duoPD']},{row['pd_duplex']},{row['num_devices']},"
                f"{row['p50_ttft_ms']:.6f},{row['p90_ttft_ms']:.6f},{row['p99_ttft_ms']:.6f},"
                f"{row['p50_e2e_ms']:.6f},{row['p90_e2e_ms']:.6f},{row['p99_e2e_ms']:.6f},"
                f"{row['p50_tbt_ms']:.6f},{row['p90_tbt_ms']:.6f},{row['p99_tbt_ms']:.6f},"
                f"{row['p50_tpot_ms']:.6f},{row['p90_tpot_ms']:.6f},{row['p99_tpot_ms']:.6f},"
                f"{row['throughput_qps']:.6f},{row['throughput_tps']:.6f}"
            )

    if args.out_csv:
        out_path = os.path.expanduser(args.out_csv)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        if args.plot_out_dir:
            plot_out_dir = os.path.expanduser(args.plot_out_dir)
            os.makedirs(plot_out_dir, exist_ok=True)

            plot_cmd = [
                sys.executable,
                os.path.join("scripts", "plot_sweep_summary.py"),
                "--csv",
                out_path,
                "--out-dir",
                plot_out_dir,
                "--no-show",
                "--ttft-ymax-ms",
                str(args.plot_ttft_ymax_ms),
                "--e2e-ymax-ms",
                str(args.plot_e2e_ymax_ms),
            ]
            if args.plot_tpot_ymax_ms is not None:
                plot_cmd += ["--tpot-ymax-ms", str(args.plot_tpot_ymax_ms)]
            if args.plot_title:
                plot_cmd += ["--title", args.plot_title]
            if args.plot_split_by_percentile:
                plot_cmd += ["--split-by-percentile"]
            if args.normalize_to_duopd_first:
                plot_cmd += ["--normalize-to-duopd-first"]
                if args.plot_ttft_norm_ymax is not None:
                    plot_cmd += ["--ttft-norm-ymax", str(args.plot_ttft_norm_ymax)]
                if args.plot_e2e_norm_ymax is not None:
                    plot_cmd += ["--e2e-norm-ymax", str(args.plot_e2e_norm_ymax)]
                if args.plot_tbt_norm_ymax is not None:
                    plot_cmd += ["--tbt-norm-ymax", str(args.plot_tbt_norm_ymax)]
                if args.plot_tpot_norm_ymax is not None:
                    plot_cmd += ["--tpot-norm-ymax", str(args.plot_tpot_norm_ymax)]

            _run_one(plot_cmd)

    elif args.plot_out_dir:
        raise SystemExit("--plot-out-dir requires --out-csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
