# Poisson + dynamic batching estimator

`scripts/queueing_estimator.py` is a **discrete-event-inspired** estimator for LLM serving
latencies under **Poisson request arrivals**, with **dynamic batching** for *prefill* and
*decode*.

It’s meant to be simple, fast, and configurable  not a perfect replica of a production
scheduler.

## What it models

### Arrivals

- Requests arrive as a Poisson process with rate `--qps` (req/s).
- Simulates `--n` requests (default: 1000).

### Prefill stage (batched)

- Prefill is modeled as a single server.
- When prefill becomes idle, it takes up to `--max-batch` requests from the queue and
  runs them as one batch.
- Prefill latency comes from calibration (`prefill_ms`) keyed by *batch size*.

**If the JSON does not contain the batch size**, prefill latency falls back to:

> `prefill_s(batch) = prefill_s(batch=1) * batch`

(requires `batch=1` to exist).

### Decode stage (batched, separate from prefill)

- Decode is also modeled as a single server, but it forms **decode batches** that can be
  different from prefill batches.

Batching policy:

- When decode becomes idle at time `t`, it forms a batch from requests whose prefill has
  completed by `t + decode_window`.
- Batch is capped by `--decode-max-batch` (defaults to `--max-batch` if 0).

Decode latency:

- Uses `decode_ms` from calibration keyed by the **formed decode batch size**.
- **Important:** `decode_ms` is interpreted as **average TBT** (*ms per token*), not total decode time.
- If that batch size is missing in JSON, decode latency falls back to `decode_ms(batch=1)`.

### Output length affects decode batching latency

- Each request samples an output length (tokens):
  - mean `--lout-mu`, stddev `--lout-sigma` (clamped to >= 1)
- For a decode batch, decode time scales with the **maximum** output length among requests
  in that batch, relative to calibration `lout`.

### (Optional) Input/context length affects decode time

If you enable `--decode-scales-with-context`, each request samples an input/context length:

- mean `--input-len-mu`, stddev `--input-len-sigma` (clamped to >= 1)

Decode time then scales linearly with context length relative to calibration `lin`.

## Metrics

Per request:

- **TTFT**: time from arrival to first token
- **E2E**: time from arrival to decode completion
- **TBT**: (decode_remaining_time) / (output_tokens-1)

Reported percentiles:

- p50/p90/p99 for TTFT, E2E, TBT

Notes:

- **TBT percentiles (p50/p90/p99)** are computed over **all per-token TBT samples** across
  all requests (i.e., treat each generated token as one sample).
- The per-request CSV logs **TPOT** (time per output token) as `tpot_ms`, which is the
  per-request **average** of recorded per-token TBTs.

## Per-request logging (full TTFT/TBT list)

If you want the **full per-request list** (not just percentiles), pass `--log` and the
script will write a CSV file with one row per request.

Path behavior:

- If you pass a **relative path** (or just a filename), it will be written under `~/results/`.
- `~/results/` is created automatically if it doesn't exist.
- If you pass an **absolute path**, it will be used as-is.

CSV columns:

- `rid`: request id (0..n-1)
- `arrival_s`
- `prefill_done_s`
- `decode_start_s`
- `decode_done_s`
- `input_len`
- `output_len`
- `ttft_ms`
- `e2e_ms`
- `tpot_ms`

Example:

```bash
python3 scripts/queueing_estimator.py \
  --qps 110 \
  --calib scripts/_tmp_calib.json \
  --n 1000 \
  --max-batch 32 \
  --decode-max-batch 32 \
  --decode-window-ms 0.01 \
  --lout-mu 128 \
  --lout-sigma 32 \
  --decode-scales-with-context \
  --input-len-mu 8192 \
  --input-len-sigma 2048 \
  --seed 1 \
  --ttft-includes-prefill \
  --log ttft_tbt.csv
```

## Calibration JSON

Create a JSON like:

```json
{
  "points": [
    {"batch": 1, "lin": 8192, "lout": 128, "prefill_ms": 12.0, "decode_ms": 0.25},
    {"batch": 4, "lin": 8192, "lout": 128, "prefill_ms": 25.0, "decode_ms": 45.0},
    {"batch": 8, "lin": 8192, "lout": 128, "prefill_ms": 35.0, "decode_ms": 55.0}
  ]
}
```

Notes:

- `decode_ms` is **ms/token** (average TBT). For example, `decode_ms: 0.25` means 0.25 ms/token.
- Total decode time for a request with `output_len` tokens is modeled roughly as:
  `decode_time_ms ≈ decode_ms * output_len` (plus batching/queueing effects).

Guidelines:

- Include `batch=1` at minimum.
- Add more batch points (2,4,8,16,32...) for better accuracy.

## Run examples

## QPS sweep (test helper)

If you want to sweep `--qps` across multiple calibration JSONs (e.g., `_tmp_calib*.json`),
use:

- `scripts/sweep_qps_queueing_estimator.py`

See `scripts/README_sweep_qps.md` for examples.

### Plot p50/p90/p99 vs QPS

After generating a sweep summary CSV (from `scripts/sweep_qps_queueing_estimator.py --out-csv ...`),
you can plot percentile curves into PNG files:

```bash
python3 scripts/plot_sweep_summary.py \
  --csv ./results/sweep_summary.csv \
  --out-dir ./results/plots \
  --no-show
```

Default output files:

- `ttft_p50_p90_p99.png`
- `e2e_p50_p90_p99.png`
- `tbt_p50_p90_p99.png`

### Plot p50/p90/p99 into separate files (overlay all calibs)

If you prefer one file per percentile (where each file overlays all `calib` curves), use:

```bash
python3 scripts/plot_sweep_summary.py \
  --csv ./results/sweep_summary.csv \
  --out-dir ./results/plots \
  --split-by-percentile \
  --no-show
```

This will create (at least):

- `ttft_p50.png`, `ttft_p90.png`, `ttft_p99.png`
- `e2e_p50.png`, `e2e_p90.png`, `e2e_p99.png`
- `tbt_p50.png`, `tbt_p90.png`, `tbt_p99.png`

### Simple (prefill batching only, decode batch defaults)

```bash
python3 scripts/queueing_estimator.py \
  --qps 110 \
  --calib scripts/_tmp_calib.json \
  --n 1000 \
  --max-batch 32 \
  --seed 1 \
  --ttft-includes-prefill
```

### Separate decode batching + output length variability

```bash
python3 scripts/queueing_estimator.py \
  --qps 110 \
  --calib scripts/_tmp_calib.json \
  --n 1000 \
  --max-batch 32 \
  --decode-max-batch 8 \
  --decode-window-ms 2 \
  --lout-mu 128 \
  --lout-sigma 32 \
  --seed 1 \
  --ttft-includes-prefill
```

### Add input/context length scaling

```bash
python3 scripts/queueing_estimator.py \
  --qps 3.0 \
  --calib scripts/_tmp_calib.json \
  --n 1000 \
  --max-batch 32 \
  --decode-max-batch 32 \
  --decode-window-ms 0.01 \
  --lout-mu 128 \
  --lout-sigma 32 \
  --decode-scales-with-context \
  --input-len-mu 8192 \
  --input-len-sigma 2048 \
  --seed 1 \
  --ttft-includes-prefill \
  --enable-duoPD \
  --log DuoPD_qps3.0.csv
```

### Example command for PD-disaggregation

```bash
python3 scripts/queueing_estimator.py \
  --qps 3.0 \
  --calib scripts/_tmp_calib2.json \
  --n 1000 \
  --max-batch 32 \
  --decode-max-batch 32 \
  --decode-window-ms 0.01 \
  --lout-mu 128 \
  --lout-sigma 32 \
  --decode-scales-with-context \
  --input-len-mu 8192 \
  --input-len-sigma 2048 \
  --seed 1 \
  --num-devices 4 \
  --ttft-includes-prefill \
  --enable-duoPD \
  --log Dis_PD_qps3.0.csv
```

## Notes / limitations

- This is a simplified model. It does **not** model KV cache eviction, GPU memory pressure,
  multi-stream execution, or per-layer compute overlap.
- Decode batch time is approximated using **max output length in the batch** (padding-like behavior).
- If you have measurements for decode time vs (batch, lout), we can plug them into calibration
  instead of using simple scaling.
