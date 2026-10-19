# QPS sweep helper

`scripts/sweep_qps_queueing_estimator.py` runs `scripts/queueing_estimator.py` repeatedly to sweep
request rate (QPS) for one or more calibration JSON files.

It prints a CSV-formatted table to stdout and can optionally write a summary CSV.

## Examples

### Running Example

### GPT3-175B
```bash
python3 scripts/sweep_qps_queueing_estimator.py \
  --model GPT \
  --calib cycle_info/GPT3-175B/DuoPD.json cycle_info/GPT3-175B/PD_1_1.json cycle_info/GPT3-175B/PD_3_1.json cycle_info/GPT3-175B/Chunked.json \
  --qps 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 \
  --out-csv ./results/sweep_summary_results_GPT3.csv \
  --plot-out-dir ./results/plots_results_GPT3 \
  --normalize-to-duopd-first \
  --plot-split-by-percentile \
  --plot-e2e-ymax-ms 50000 \
  -- \
  --n 1000 \
  --decode-window-ms 0.01 \
  --lout-mu 128 \
  --lout-sigma 32 \
  --decode-scales-with-context \
  --input-len-mu 32768 \
  --input-len-sigma 4096 \
  --seed 1 \
  --ttft-includes-prefill
```
```bash
python3 scripts/sweep_qps_queueing_estimator.py \
  --model GPT \
  --calib cycle_info/GPT3-175B/DuoPD_SM.json cycle_info/GPT3-175B/SM_9:1.json cycle_info/GPT3-175B/SM_7:3.json cycle_info/GPT3-175B/SM_5:5.json \
  --qps 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 \
  --out-csv ./results/sweep_summary_results_GPT3_SM.csv \
  --plot-out-dir ./results/plots_results_GPT3_SM \
  --normalize-to-duopd-first \
  --plot-split-by-percentile \
  --plot-e2e-ymax-ms 50000 \
  --plot-tpot-norm-ymax 10.0 \
  -- \
  --n 1000 \
  --decode-window-ms 0.01 \
  --lout-mu 128 \
  --lout-sigma 32 \
  --decode-scales-with-context \
  --input-len-mu 32768 \
  --input-len-sigma 4096 \
  --seed 1 \
  --ttft-includes-prefill 
```

### LLAMA2-70B
```bash
python3 scripts/sweep_qps_queueing_estimator.py \
  --model LLAMA \
  --calib cycle_info/Llama2-70B/DuoPD_SM.json cycle_info/Llama2-70B/SM_9:1.json cycle_info/Llama2-70B/SM_7:3.json cycle_info/Llama2-70B/SM_5:5.json \
  --qps 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 \
  --out-csv ./results/sweep_summary_results_Llama_SM.csv \
  --plot-out-dir ./results/plots_results_Llama_SM \
  --normalize-to-duopd-first \
  --plot-split-by-percentile \
  --plot-e2e-ymax-ms 50000 \
  -- \
  --n 1000 \
  --decode-window-ms 0.01 \
  --lout-mu 128 \
  --lout-sigma 32 \
  --decode-scales-with-context \
  --input-len-mu 32768 \
  --input-len-sigma 4096 \
  --seed 1 \
  --ttft-includes-prefill
```

```bash
python3 scripts/sweep_qps_queueing_estimator.py \
  --model LLAMA \
  --calib cycle_info/Llama2-70B/DuoPD.json cycle_info/Llama2-70B/PD_1_1.json cycle_info/Llama2-70B/PD_3_1.json cycle_info/Llama2-70B/Chunked.json \
  --qps 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.25 3.5 \
  --out-csv ./results/sweep_summary_results_Llama.csv \
  --plot-out-dir ./results/plots_results_Llama \
  --normalize-to-duopd-first \
  --plot-split-by-percentile \
  --plot-e2e-ymax-ms 50000 \
  -- \
  --n 1000 \
  --decode-window-ms 0.01 \
  --lout-mu 128 \
  --lout-sigma 32 \
  --decode-scales-with-context \
  --input-len-mu 32768 \
  --input-len-sigma 4096 \
  --seed 1 \
  --ttft-includes-prefill
```

### Qwen3-235B
```bash
python3 scripts/sweep_qps_queueing_estimator.py \
  --model QWEN \
  --calib cycle_info/Qwen-235B/DuoPD_SM.json cycle_info/Qwen-235B/SM_9:1.json cycle_info/Qwen-235B/SM_7:3.json cycle_info/Qwen-235B/SM_5:5.json \
  --qps 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0 4.25 4.5 \
  --out-csv ./results/sweep_summary_results_Qwen_SM.csv \
  --plot-out-dir ./results/plots_results_Qwen_SM \
  --normalize-to-duopd-first \
  --plot-split-by-percentile \
  --plot-e2e-ymax-ms 50000 \
  --plot-tpot-norm-ymax 6.0 \
  -- \
  --n 1000 \
  --decode-window-ms 0.01 \
  --lout-mu 128 \
  --lout-sigma 32 \
  --decode-scales-with-context \
  --input-len-mu 32768 \
  --input-len-sigma 4096 \
  --seed 1 \
  --ttft-includes-prefill
```

```bash
python3 scripts/sweep_qps_queueing_estimator.py \
  --model QWEN \
  --calib cycle_info/Qwen-235B/DuoPD.json cycle_info/Qwen-235B/PD_1_1.json cycle_info/Qwen-235B/PD_3_1.json cycle_info/Qwen-235B/Chunked.json \
  --qps 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0 4.25 4.5 4.75 5.0 \
  --out-csv ./results/sweep_summary_results_Qwen.csv \
  --plot-out-dir ./results/plots_results_Qwen \
  --normalize-to-duopd-first \
  --plot-split-by-percentile \
  --plot-e2e-ymax-ms 50000 \
  --plot-tpot-norm-ymax 10.0 \
  -- \
  --n 1000 \
  --decode-window-ms 0.01 \
  --lout-mu 128 \
  --lout-sigma 32 \
  --decode-scales-with-context \
  --input-len-mu 32768 \
  --input-len-sigma 4096 \
  --seed 1 \
  --ttft-includes-prefill
```

### Sweep a list of QPS values

```bash
python3 scripts/sweep_qps_queueing_estimator.py \
  --calib cycle_info/DuoPD.json \
  --qps 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4 \
  --out-csv ./results/sweep_summary_DuoPD.csv \
  --plot-out-dir ./results/plots_DuoPD \
  --plot-split-by-percentile \
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
  --enable-duoPD
```

### PD-disaggregation example

```bash
# 1:1 PD disaggregation
python3 scripts/sweep_qps_queueing_estimator.py \
  --calib cycle_info/PD_1_1.json \
  --qps 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4 \
  --out-csv ./results/sweep_summary_PD_1_1.csv \
  --plot-out-dir ./results/plots_PD_1_1 \
  --plot-split-by-percentile \
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
  --ttft-includes-prefill

# 2:1 PD disaggregation
python3 scripts/sweep_qps_queueing_estimator.py \
  --calib cycle_info/PD_2_1.json \
  --qps 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4 \
  --out-csv ./results/sweep_summary_PD_2_1.csv \
  --plot-out-dir ./results/plots_PD_2_1 \
  --plot-split-by-percentile \
  --plot-e2e-ymax-ms 10000 \
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
  --num-devices 2 \
  --ttft-includes-prefill
```

### Sweep an QPS range

```bash
python3 scripts/sweep_qps_queueing_estimator.py \
  --calib cycle_info/PD_1_1.json \
  --qps-range 0.5 10 0.5 \
  --out-csv ./results/sweep_summary_calib2.csv \
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
  --ttft-includes-prefill
```

### Also generate per-request/per-token logs for every run

Add `--log-prefix` and the script will pass a unique `--log` filename to each run:

```bash
python3 scripts/sweep_qps_queueing_estimator.py \
  --calib cycle_info/PD_1_1.json \
  --qps 1 2 3 \
  --log-prefix DisPD \
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
  --ttft-includes-prefill
```

This produces log files like:

- `DisPD__tmp_calib2_qps1.csv`
- `DisPD__tmp_calib2_qps1.csv.tokens.csv`

(under `~/results/` because `queueing_estimator.py` normalizes relative `--log` paths).

### Plot p50/p90/p99 vs QPS

After generating a sweep summary CSV, you can plot percentile curves (p50/p90/p99) into PNG files:

```bash
python3 scripts/plot_sweep_summary.py \
  --csv ./results/sweep_summary.csv \
  --out-dir ./results/plots \
  --no-show
```

This will create (at least):

- `~/results/plots/ttft_p50_p90_p99.png`
- `~/results/plots/e2e_p50_p90_p99.png`
- `~/results/plots/tbt_p50_p90_p99.png`

### Plot automatically during the sweep (recommended)

If you want plots produced directly by the sweep command, add `--plot-out-dir`.

By default, TTFT and E2E plots are capped to **2000 ms** on the y-axis.

```bash
python3 scripts/sweep_qps_queueing_estimator.py \
  --calib cycle_info/DuoPD.json \
  --qps 0.5 1 2 3 4 5 \
  --out-csv ./results/sweep_summary.csv \
  --plot-out-dir ./results/plots \
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
  --enable-duoPD
```

You can override the caps via:

- `--plot-ttft-ymax-ms`
- `--plot-e2e-ymax-ms`
