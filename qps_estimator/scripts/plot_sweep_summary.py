#!/usr/bin/env python3
"""Plot percentile curves from sweep_qps_queueing_estimator.py summary CSV.

Contract
- Input: a summary CSV produced by scripts/sweep_qps_queueing_estimator.py
    Columns: calib,qps,p50_ttft_ms,p90_ttft_ms,p99_ttft_ms,p50_e2e_ms,p90_e2e_ms,p99_e2e_ms,p50_tbt_ms,p90_tbt_ms,p99_tbt_ms,p50_tpot_ms,p90_tpot_ms,p99_tpot_ms,throughput_qps,throughput_tps
- Output: one PNG per metric (ttft/e2e/tbt/tpot/throughput) with curves.

Notes
- Uses matplotlib only (assumed available in most Python envs). If not, install it.
- Groups by `calib` and draws one line per (calib, percentile).

Example
  python3 scripts/plot_sweep_summary.py \
    --csv ~/results/sweep_summary.csv \
    --out-dir ~/results/plots

"""

import argparse
import csv
import os
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def _to_float(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return float("nan")


def _basename_no_ext(p: str) -> str:
    b = os.path.basename(p)
    return os.path.splitext(b)[0]


def _is_duopd_calib_name(calib: str) -> bool:
    base = _basename_no_ext(calib).lower()
    return base == "duopd" or base == "duopd_sm"


def _safe_div(numer: float, denom: float) -> float:
    if denom == 0.0:
        return float("nan")
    return numer / denom


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to sweep summary CSV.")
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write plots (PNG). Will be created.",
    )
    ap.add_argument(
        "--title",
        default=None,
        help="Optional title prefix (e.g., an experiment name).",
    )
    ap.add_argument(
        "--ttft-ymax-ms",
        type=float,
        default=2000.0,
        help="Max y-axis (ms) for TTFT plot. Default: 2000.",
    )
    ap.add_argument(
        "--e2e-ymax-ms",
        type=float,
        default=2000.0,
        help="Max y-axis (ms) for E2E plot. Default: 2000.",
    )

    ap.add_argument(
        "--ttft-norm-ymax",
        type=float,
        default=None,
        help="Optional max y-axis (ratio) for normalized TTFT plots. Default: uncapped.",
    )
    ap.add_argument(
        "--e2e-norm-ymax",
        type=float,
        default=None,
        help="Optional max y-axis (ratio) for normalized E2E plots. Default: uncapped.",
    )
    ap.add_argument(
        "--tbt-norm-ymax",
        type=float,
        default=10.0,
        help="Optional max y-axis (ratio) for normalized TBT plots. Default: 10.",
    )
    ap.add_argument(
        "--tpot-norm-ymax",
        type=float,
        default=None,
        help="Optional max y-axis (ratio) for normalized TPOT plots. Default: 10.",
    )
    ap.add_argument(
        "--tbt-ymax-ms",
        type=float,
        default=None,
        help="Optional max y-axis (ms) for TBT plot. Default: uncapped.",
    )
    ap.add_argument(
        "--tpot-ymax-ms",
        type=float,
        default=None,
        help="Optional max y-axis (ms) for TPOT plot. Default: uncapped.",
    )
    ap.add_argument(
        "--throughput-qps-ymax",
        type=float,
        default=None,
        help="Optional max y-axis (req/s) for throughput_qps plot. Default: uncapped.",
    )
    ap.add_argument(
        "--throughput-tps-ymax",
        type=float,
        default=None,
        help="Optional max y-axis (tok/s) for throughput_tps plot. Default: uncapped.",
    )
    ap.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive windows (recommended on servers).",
    )

    ap.add_argument(
        "--split-by-percentile",
        action="store_true",
        help=(
            "Write separate PNGs per percentile (p50/p90/p99). Each file overlays all calibration curves. "
            "Example outputs: ttft_p50.png, ttft_p90.png, ttft_p99.png (and similarly for e2e/tbt/tpot)."
        ),
    )

    ap.add_argument(
        "--normalize-to-duopd-first",
        action="store_true",
        help=(
            "Plot TTFT/TBT/TPOT/E2E as normalized ratios relative to the first DuoPD row (smallest QPS). "
            "Normalization is done per percentile (p50/p90/p99) and per metric. "
            "If DuoPD isn't present in the CSV, normalization is skipped with an error."
        ),
    )

    args = ap.parse_args()

    csv_path = os.path.expanduser(args.csv)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = _read_rows(csv_path)
    if not rows:
        raise SystemExit("No rows found in CSV: {}".format(csv_path))

    # Lazily import matplotlib so the script can still print a clear error if missing.
    try:
        import matplotlib

        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install it (e.g., pip install matplotlib).\n{}".format(e)
        )

    # Style: make figures presentation-friendly.
    # - No legend (범주박스 제거)
    # - No title (상단 글 제거)
    # - Larger fonts
    matplotlib.rcParams.update(
        {
            "font.size": 30,
            "axes.labelsize": 30,
            "xtick.labelsize": 30,
            "ytick.labelsize": 30,
            "lines.linewidth": 4,
        }
    )

    # calib -> metric_key -> percentile_key -> List[(qps, value)]
    data: DefaultDict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    metrics = {
        "ttft_ms": ("p50_ttft_ms", "p90_ttft_ms", "p99_ttft_ms"),
        "e2e_ms": ("p50_e2e_ms", "p90_e2e_ms", "p99_e2e_ms"),
        "tbt_ms": ("p50_tbt_ms", "p90_tbt_ms", "p99_tbt_ms"),
        "tpot_ms": ("p50_tpot_ms", "p90_tpot_ms", "p99_tpot_ms"),
    }

    for row in rows:
        calib = row.get("calib", "")
        qps = _to_float(row.get("qps", "nan"))
        for metric_name, cols in metrics.items():
            for col in cols:
                pct = col.split("_")[0]  # p50/p90/p99
                val = _to_float(row.get(col, "nan"))
                data[calib][metric_name][pct].append((qps, val))

    # Fixed calib -> color map (default color cycle in sorted order),
    # with the Chunked and DuoPD colors swapped.
    _cycle = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    calib_colors: Dict[str, str] = {
        c: _cycle[i % len(_cycle)] for i, c in enumerate(sorted(data.keys()))
    }
    _chunked = [c for c in calib_colors if _basename_no_ext(c).lower() == "chunked"]
    _duopd = [c for c in calib_colors if _basename_no_ext(c).lower() == "duopd"]
    if _chunked and _duopd:
        a, b = _chunked[0], _duopd[0]
        calib_colors[a], calib_colors[b] = calib_colors[b], calib_colors[a]

    # metric_name -> pct -> baseline_value
    norm_baseline: Dict[str, Dict[str, float]] = defaultdict(dict)
    throughput_norm_baseline: Dict[str, float] = {}
    duopd_calib = None
    if args.normalize_to_duopd_first:
        duopd_calibs = [c for c in data.keys() if _is_duopd_calib_name(c)]
        if not duopd_calibs:
            raise SystemExit(
                "--normalize-to-duopd-first requires a DuoPD series in the CSV (calib basename must be DuoPD)."
            )

        # If there are multiple DuoPD-like rows, pick the lexicographically first path (stable).
        duopd_calib = sorted(duopd_calibs)[0]
        for metric_name in metrics.keys():
            for pct in ("p50", "p90", "p99"):
                pts = sorted(data[duopd_calib][metric_name].get(pct, []), key=lambda t: t[0])
                if not pts:
                    raise SystemExit(
                        "Missing baseline data for DuoPD metric={} pct={} in CSV".format(metric_name, pct)
                    )
                # Baseline = the first (smallest qps) row for DuoPD for that percentile.
                norm_baseline[metric_name][pct] = float(pts[0][1])

        # Throughput baseline = first (smallest qps) DuoPD point per throughput metric.
        duo_rows = [r for r in rows if r.get("calib", "") == duopd_calib]
        duo_rows = sorted(duo_rows, key=lambda r: _to_float(r.get("qps", "nan")))
        if not duo_rows:
            raise SystemExit("Missing DuoPD rows for throughput normalization in CSV")
        throughput_norm_baseline["throughput_qps"] = _to_float(duo_rows[0].get("throughput_qps", "nan"))
        throughput_norm_baseline["throughput_tps"] = _to_float(duo_rows[0].get("throughput_tps", "nan"))

    def _decorate_and_save(
        *,
        title: str,
        ylabel: str,
        png_name: str,
        ymax_ms: float = None,
    ) -> None:
        plt.xlabel("QPS (req/s)")
        plt.ylabel(ylabel)
        if ymax_ms is not None:
            plt.ylim(0.0, float(ymax_ms))
        plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        plt.tight_layout()

        out_path = os.path.join(out_dir, png_name)
        plt.savefig(out_path)

    def plot_metric_all_percentiles(metric_name: str, ylabel: str, png_name: str, ymax_ms: float = None) -> None:
        plt.figure(figsize=(9, 5), dpi=200)

        for calib, m in sorted(data.items(), key=lambda kv: kv[0]):
            for pct in ("p50", "p90", "p99"):
                pts = m[metric_name].get(pct, [])
                pts_sorted = sorted(pts, key=lambda t: t[0])
                xs = [x for x, _ in pts_sorted]
                ys = [y for _, y in pts_sorted]
                label = "{} {}".format(_basename_no_ext(calib), pct)
                plt.plot(xs, ys, marker="o", linewidth=4.0, markersize=6, label=label, color=calib_colors.get(calib))

        base_title = "{}".format(metric_name)
        if args.title:
            base_title = "{} - {}".format(args.title, metric_name)
        _decorate_and_save(title=base_title, ylabel=ylabel, png_name=png_name, ymax_ms=ymax_ms)

    def plot_series_one_line_per_calib(
        *,
        col_name: str,
        ylabel: str,
        png_name: str,
        ymax: float = None,
    ) -> None:
        plt.figure(figsize=(9, 5), dpi=200)
        by_calib: DefaultDict[str, List[Tuple[float, float]]] = defaultdict(list)
        for row in rows:
            calib = row.get("calib", "")
            qps = _to_float(row.get("qps", "nan"))
            val = _to_float(row.get(col_name, "nan"))
            by_calib[calib].append((qps, val))

        for calib, pts in sorted(by_calib.items(), key=lambda kv: kv[0]):
            pts_sorted = sorted(pts, key=lambda t: t[0])
            xs = [x for x, _ in pts_sorted]
            ys = [y for _, y in pts_sorted]
            label = "{}".format(_basename_no_ext(calib))
            plt.plot(xs, ys, marker="o", linewidth=4.0, markersize=6, label=label, color=calib_colors.get(calib))

        base_title = "{}".format(col_name)
        if args.title:
            base_title = "{} - {}".format(args.title, col_name)
        _decorate_and_save(title=base_title, ylabel=ylabel, png_name=png_name, ymax_ms=ymax)

    def plot_series_one_line_per_calib_normalized(
        *,
        col_name: str,
        ylabel: str,
        png_name: str,
        baseline: float,
        ymax: float = None,
    ) -> None:
        plt.figure(figsize=(9, 5), dpi=200)
        by_calib: DefaultDict[str, List[Tuple[float, float]]] = defaultdict(list)
        for row in rows:
            calib = row.get("calib", "")
            qps = _to_float(row.get("qps", "nan"))
            val = _to_float(row.get(col_name, "nan"))
            by_calib[calib].append((qps, val))

        for calib, pts in sorted(by_calib.items(), key=lambda kv: kv[0]):
            pts_sorted = sorted(pts, key=lambda t: t[0])
            xs = [x for x, _ in pts_sorted]
            ys = [_safe_div(y, baseline) for _, y in pts_sorted]
            label = "{}".format(_basename_no_ext(calib))
            plt.plot(xs, ys, marker="o", linewidth=4.0, markersize=6, label=label, color=calib_colors.get(calib))

        base_title = "{} (normalized)".format(col_name)
        if args.title:
            base_title = "{} - {} (normalized)".format(args.title, col_name)
        _decorate_and_save(title=base_title, ylabel=ylabel, png_name=png_name, ymax_ms=ymax)

    def plot_metric_split_percentile(
        metric_name: str,
        ylabel: str,
        png_prefix: str,
        ymax_ms: float = None,
    ) -> None:
        for pct in ("p50", "p90", "p99"):
            plt.figure(figsize=(9, 5), dpi=200)
            for calib, m in sorted(data.items(), key=lambda kv: kv[0]):
                pts = m[metric_name].get(pct, [])
                pts_sorted = sorted(pts, key=lambda t: t[0])
                xs = [x for x, _ in pts_sorted]
                ys = [y for _, y in pts_sorted]
                label = "{}".format(_basename_no_ext(calib))
                plt.plot(xs, ys, marker="o", linewidth=4.0, markersize=6, label=label, color=calib_colors.get(calib))

            base_title = "{} {}".format(metric_name, pct)
            if args.title:
                base_title = "{} - {} {}".format(args.title, metric_name, pct)

            _decorate_and_save(
                title=base_title,
                ylabel=ylabel,
                png_name="{}_{}.png".format(png_prefix, pct),
                ymax_ms=ymax_ms,
            )

    def plot_metric_split_percentile_normalized(
        metric_name: str,
        ylabel: str,
        png_prefix: str,
        ymax: float = None,
    ) -> None:
        for pct in ("p50", "p90", "p99"):
            plt.figure(figsize=(9, 5), dpi=200)
            baseline = norm_baseline.get(metric_name, {}).get(pct, float("nan"))
            for calib, m in sorted(data.items(), key=lambda kv: kv[0]):
                pts = m[metric_name].get(pct, [])
                pts_sorted = sorted(pts, key=lambda t: t[0])
                xs = [x for x, _ in pts_sorted]
                ys = [_safe_div(y, baseline) for _, y in pts_sorted]
                label = "{}".format(_basename_no_ext(calib))
                plt.plot(xs, ys, marker="o", linewidth=4.0, markersize=6, label=label, color=calib_colors.get(calib))

            base_title = "{} {} (normalized)".format(metric_name, pct)
            if args.title:
                base_title = "{} - {} {} (normalized)".format(args.title, metric_name, pct)

            _decorate_and_save(
                title=base_title,
                ylabel=ylabel,
                png_name="{}_{}_normalized.png".format(png_prefix, pct),
                ymax_ms=ymax,
            )

    if args.normalize_to_duopd_first:
        # Always split by percentile for normalized plots.
        plot_metric_split_percentile_normalized(
            "ttft_ms",
            "Norm. Latency",
            "ttft_norm",
            ymax=args.ttft_norm_ymax,
        )
        plot_metric_split_percentile_normalized(
            "e2e_ms",
            "Norm. Latency",
            "e2e_norm",
            ymax=args.e2e_norm_ymax,
        )
        plot_metric_split_percentile_normalized("tbt_ms", "Norm. Latency", "tbt_norm", ymax=args.tbt_norm_ymax)
        plot_metric_split_percentile_normalized("tpot_ms", "Norm. Latency", "tpot_norm", ymax=args.tpot_norm_ymax)
        plot_series_one_line_per_calib_normalized(
            col_name="throughput_qps",
            ylabel="Norm. Throughput (req/s)",
            png_name="throughput_qps_normalized.png",
            baseline=throughput_norm_baseline.get("throughput_qps", float("nan")),
            ymax=None,
        )
        plot_series_one_line_per_calib_normalized(
            col_name="throughput_tps",
            ylabel="Norm. Throughput (tok/s)",
            png_name="throughput_tps_normalized.png",
            baseline=throughput_norm_baseline.get("throughput_tps", float("nan")),
            ymax=None,
        )

    if args.split_by_percentile and not args.normalize_to_duopd_first:
        plot_metric_split_percentile("ttft_ms", "Latency (ms)", "ttft", ymax_ms=args.ttft_ymax_ms)
        plot_metric_split_percentile("e2e_ms", "Latency (ms)", "e2e", ymax_ms=args.e2e_ymax_ms)
        plot_metric_split_percentile("tbt_ms", "Latency (ms)", "tbt", ymax_ms=args.tbt_ymax_ms)
        plot_metric_split_percentile("tpot_ms", "Latency (ms)", "tpot", ymax_ms=args.tpot_ymax_ms)
        plot_series_one_line_per_calib(
            col_name="throughput_qps",
            ylabel="Norm. Throughput (req/s)",
            png_name="throughput_qps.png",
            ymax=args.throughput_qps_ymax,
        )
        plot_series_one_line_per_calib(
            col_name="throughput_tps",
            ylabel="Norm. Throughput (tok/s)",
            png_name="throughput_tps.png",
            ymax=args.throughput_tps_ymax,
        )
    elif not args.split_by_percentile and not args.normalize_to_duopd_first:
        plot_metric_all_percentiles("ttft_ms", "Latency (ms)", "ttft_p50_p90_p99.png", ymax_ms=args.ttft_ymax_ms)
        plot_metric_all_percentiles("e2e_ms", "Latency (ms)", "e2e_p50_p90_p99.png", ymax_ms=args.e2e_ymax_ms)
        plot_metric_all_percentiles("tbt_ms", "Latency (ms)", "tbt_p50_p90_p99.png", ymax_ms=args.tbt_ymax_ms)
        plot_metric_all_percentiles("tpot_ms", "Latency (ms)", "tpot_p50_p90_p99.png", ymax_ms=args.tpot_ymax_ms)
        plot_series_one_line_per_calib(
            col_name="throughput_qps",
            ylabel="Throughput (req/s)",
            png_name="throughput_qps.png",
            ymax=args.throughput_qps_ymax,
        )
        plot_series_one_line_per_calib(
            col_name="throughput_tps",
            ylabel="Throughput (tok/s)",
            png_name="throughput_tps.png",
            ymax=args.throughput_tps_ymax,
        )

    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
