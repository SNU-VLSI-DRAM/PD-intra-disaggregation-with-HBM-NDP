#!/usr/bin/env python3
"""Discrete-event estimator for Poisson arrivals with dynamic prefill batching.

Contract
- Arrivals: Poisson with rate `--qps` (requests/sec).
- Calib: provides prefill latency per *formed batch size* and decode latency.
    - Prefill: `prefill_ms` for batch sizes 1..N.
    - Decode: `decode_ms` per request (may be constant across batches or vary).
- Dynamic batching: prefill runs as a single server. When it becomes free, it starts
    a batch made from all queued requests up to `--max-batch`.
- Dependency: each request's decode starts only after its own prefill completes.
- Decode resource: a single decode server (one request decodes at a time).

Metrics per request
- TTFT: time from arrival -> first token produced.
- E2E: time from arrival -> end of decode.
- TBT: (E2E - TTFT) / max(out_tokens-1, 1).
- TPOT: average of per-token TBT values for each request.

Defaults
- Request count fixed to 1000 (can be changed via --n for convenience).
- Output tokens fixed to calib lout (no distribution inputs).

Units
- Internally seconds; CLI takes ms where noted.
"""

import argparse
import json
import math
import os
import random
import statistics
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple


def _kv_cache_bytes_per_token(
    hidden_dim: int,
    n_layers: int,
    bytes_per_elem: int,
    gqa_size: int = 1,
) -> int:
    """KV cache bytes for one token for one request.

    Approx: KV per layer = 2 * hidden_dim elements (K and V).
    Total across layers: 2 * hidden_dim * n_layers elements.
    """
    base = 2 * int(hidden_dim) * int(n_layers) * int(bytes_per_elem)
    gqa = int(gqa_size) if int(gqa_size) > 0 else 1
    # In GQA/MQA, KV heads are fewer than Q heads, reducing KV cache size.
    # We approximate this as dividing KV cache bytes by the GQA grouping factor.
    return int(base // gqa)


class CalibPoint:
    def __init__(self, batch: int, lin: int, lout: int, prefill_s: float, decode_tbt_s: float):
        self.batch = batch
        self.lin = lin
        self.lout = lout
        self.prefill_s = prefill_s
        # Average time between tokens during decode (seconds/token)
        self.decode_tbt_s = decode_tbt_s


def _latency_for_batch(pts: Dict[int, CalibPoint], batch: int) -> Tuple[float, float, int, int]:
    """Return (prefill_s, decode_s, lin, lout) for a formed batch.

        Behavior:
        - If an exact batch size is provided in the calibration, use it.
        - If it's missing, fall back to linear scaling from batch=1 for prefill:
                prefill_s(batch) = prefill_s(batch=1) * batch
            while keeping decode latency the same as batch=1.

        This matches the simplified assumption:
            "prefill cost scales with batch size, decode is unchanged".
    """
    if batch <= 0:
        batch = 1

    if batch in pts:
        p = pts[batch]
        return p.prefill_s, p.decode_tbt_s, p.lin, p.lout

    if 1 not in pts:
        raise ValueError("Calibration must include batch=1 to enable fallback scaling")

    p1 = pts[1]
    return p1.prefill_s, p1.decode_tbt_s, p1.lin, p1.lout


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    w = idx - lo
    return (1.0 - w) * sorted_vals[lo] + w * sorted_vals[hi]


def _exp_interarrival(qps: float) -> float:
    if qps <= 0:
        return float("inf")
    return random.expovariate(qps)


def parse_calib_json(path: str) -> Dict[int, CalibPoint]:
    """Parse calibration file.

    Expected JSON:
    {
      "points": [
        {"batch":1,"lin":8192,"lout":128,"prefill_ms":12.3,"decode_ms":4.56},
        ...
      ]
    }

    Returns: dict batch -> CalibPoint
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pts = {}
    for p in data.get("points", []):
        batch = int(p["batch"])
        lin = int(p["lin"])
        lout = int(p["lout"])
        prefill_s = float(p["prefill_ms"]) / 1000.0
        # decode_ms is interpreted as average TBT (ms/token)
        decode_tbt_s = float(p["decode_ms"]) / 1000.0
        pts[batch] = CalibPoint(
            batch=batch,
            lin=lin,
            lout=lout,
            prefill_s=prefill_s,
            decode_tbt_s=decode_tbt_s,
        )
    if not pts:
        raise ValueError("No calibration points found in JSON")
    return pts


def _is_sm_ratio_calib(calib_path: str) -> bool:
    base = os.path.basename(calib_path).lower()
    return base in ("sm_9:1.json", "sm_7:3.json", "sm_5:5.json")

class Request:
    def __init__(self, rid: int, arrival_t: float):
        self.rid = rid
        self.arrival_t = arrival_t
        self.input_len = None
        self.lout = None
        self.prefill_done_t = None
        self.decode_start_t = None
        self.decode_done_t = None
        # Per-token decode TBT samples (seconds/token). Populated by the step-based decoder.
        self.decode_tbt_s_list = []
        # Chunked prefill bookkeeping
        self.prefill_remaining_tokens = None
        # Number of input tokens already prefilling/encoded into KV cache (chunked-prefill mode).
        self.prefill_processed_tokens = 0


def _prefill_time_for_batch_tokens(
    *,
    pts: Dict[int, CalibPoint],
    formed: int,
    total_tokens: int,
    input_len_mu: float,
    prefill_hidden_dim: int,
    prefill_base_n: int,
    # Optional memory-time override model.
    prefill_use_memory_model: bool,
    prefill_mem_ops_per_nh: float,
    prefill_mem_bw_scale: float,
    model_weight_bytes: float,
    model_hidden_dim: int,
    model_layers: int,
    model_bytes_per_elem: int,
    gqa_size: int,
    pd_duplex: bool,
) -> float:
    """Estimate prefill service time (seconds) for a batch processing total_tokens.

    We start from calibrated prefill latency for the formed batch size, then scale by
    an attention FLOPs proxy (same model as the existing non-chunked prefill stage).

    In chunked mode, total_tokens is the total number of tokens processed in this prefill
    *chunk batch* (sum of per-request chunk tokens).
    """
    sp_s, _, calib_lin, _ = _latency_for_batch(pts, formed)

    N = float(max(1, int(total_tokens)))
    H = float(max(1, int(prefill_hidden_dim)))
    N0 = float(max(1, int(prefill_base_n if prefill_base_n > 0 else calib_lin)))

    flops = 24.0 * N * (H**2) + 4.0 * (N**2) * H
    flops0 = 24.0 * N0 * (H**2) + 4.0 * (N0**2) * H
    if flops0 > 0.0:
        sp_s *= flops / flops0

    # If requested, cap by memory access time (prefill becomes memory-bound).
    # We reuse the decode-style bandwidth derivation:
    #   bandwidth = calib_bytes / calib_time
    # Then for this batch:
    #   mem_time = batch_bytes / bandwidth
    if prefill_use_memory_model:
        kv_bpt = _kv_cache_bytes_per_token(
            hidden_dim=model_hidden_dim,
            n_layers=model_layers,
            bytes_per_elem=model_bytes_per_elem,
            gqa_size=gqa_size,
        )
        if pd_duplex:
            kv_bpt *= 0.25

        # Derive effective BW from calibration point at lin==calib_lin.
        # Use decode_tbt_s from the same calib batch as a proxy for bandwidth.
        _, calib_dec_tbt_s, calib_lin, _ = _latency_for_batch(pts, formed)
        calib_bytes = float(model_weight_bytes) + float(calib_lin) * float(kv_bpt)
        bw_bytes_per_s = calib_bytes / float(calib_dec_tbt_s) if float(calib_dec_tbt_s) > 0.0 else float("inf")

        # Prefill memory traffic:
        # - plus an explicit per-token activation read/write term: ops * N * H * bytes.
        # The user's requested list of ops maps to a scalar multiplier.
        bytes_per_act = float(max(1, int(model_bytes_per_elem)))
        act_bytes = float(prefill_mem_ops_per_nh) * float(N) * float(H) * bytes_per_act
        step_bytes = float(model_weight_bytes) + float(N) * float(kv_bpt) + act_bytes

        # Optionally scale BW for prefill vs decode (e.g., worse locality).
        bw_bytes_per_s = float(bw_bytes_per_s) * float(prefill_mem_bw_scale)
        mem_s = float(step_bytes) / float(bw_bytes_per_s) if float(bw_bytes_per_s) > 0.0 else 0.0
        sp_s = max(float(sp_s), float(mem_s))

    return float(sp_s)


def _prefill_time_for_chunked_microbatch(
    *,
    pts: Dict[int, CalibPoint],
    formed: int,
    chunk_contexts_tokens: List[int],
    chunk_new_tokens: List[int],
    prefill_hidden_dim: int,
    prefill_base_n: int,
    # Optional memory-time override model.
    prefill_use_memory_model: bool,
    prefill_mem_ops_per_nh: float,
    prefill_mem_bw_scale: float,
    model_weight_bytes: float,
    model_hidden_dim: int,
    model_layers: int,
    model_bytes_per_elem: int,
    gqa_size: int,
    pd_duplex: bool,
) -> float:
    """Prefill time for a chunked-prefill microbatch that accounts for prior KV (attention context).

    Each request in the microbatch processes `chunk_new_tokens[i]` new tokens, but its attention
    sees `chunk_contexts_tokens[i]` previous KV tokens already in cache.

    We approximate attention FLOPs as:
      sum_i (N_new_i * (H^2) + N_new_i * N_ctx_i * H)
    which is the incremental cost for processing new tokens with attention over existing context.

    This intentionally differs from the non-chunked proxy (which uses N^2) because chunked prefill
    reuses KV and only pays cross-attention between new tokens and existing context.
    """
    sp_s, _, calib_lin, _ = _latency_for_batch(pts, formed)

    H = float(max(1, int(prefill_hidden_dim)))
    # Baseline context length used by the calibration point (or user override).
    N0 = float(max(1, int(prefill_base_n if prefill_base_n > 0 else calib_lin)))

    # Baseline (non-chunked) FLOPs proxy for N0 tokens.
    flops0 = 24.0 * N0 * (H**2) + 4.0 * (N0**2) * H

    # Chunked incremental FLOPs proxy.
    flops = 0.0
    for ctx, new in zip(chunk_contexts_tokens, chunk_new_tokens):
        n_new = float(max(0, int(new)))
        n_ctx = float(max(0, int(ctx)))
        # MLP/QKV-ish term scales with new tokens; attention term scales with new*context.
        flops += 24.0 * n_new * (H**2) + 4.0 * n_new * n_ctx * H

    if flops0 > 0.0:
        sp_s *= flops / flops0

    if prefill_use_memory_model:
        kv_bpt = _kv_cache_bytes_per_token(
            hidden_dim=model_hidden_dim,
            n_layers=model_layers,
            bytes_per_elem=model_bytes_per_elem,
            gqa_size=gqa_size,
        )
        if pd_duplex:
            kv_bpt *= 0.25

        # Bandwidth derived from the same calib batch's decode_tbt_s.
        _, calib_dec_tbt_s, calib_lin, _ = _latency_for_batch(pts, formed)
        calib_bytes = float(model_weight_bytes) + float(calib_lin) * float(kv_bpt)
        bw_bytes_per_s = calib_bytes / float(calib_dec_tbt_s) if float(calib_dec_tbt_s) > 0.0 else float("inf")

        bytes_per_act = float(max(1, int(model_bytes_per_elem)))
        # For chunked microbatch, sum activation bytes per request based on new token count.
        act_bytes = 0.0
        kv_bytes = 0.0
        for new in chunk_new_tokens:
            n_new = float(max(0, int(new)))
            act_bytes += float(prefill_mem_ops_per_nh) * n_new * float(H) * bytes_per_act
            kv_bytes += n_new * float(kv_bpt)
        step_bytes = float(model_weight_bytes) + kv_bytes + act_bytes

        bw_bytes_per_s = float(bw_bytes_per_s) * float(prefill_mem_bw_scale)
        mem_s = float(step_bytes) / float(bw_bytes_per_s) if float(bw_bytes_per_s) > 0.0 else 0.0
        sp_s = max(float(sp_s), float(mem_s))

    return float(sp_s)


def simulate(
    qps: float,
    pts: Dict[int, CalibPoint],
    n: int,
    max_batch: int,
    seed: int,
    include_prefill_in_ttft: bool,
    decode_scales_with_context: bool,
    input_len_mu: float,
    input_len_sigma: float,
    decode_max_batch: int,
    decode_window_s: float,
    lout_mu: float,
    lout_sigma: float,
    log_path: Optional[str],
    # Model params for KV-cache traffic.
    model_hidden_dim: int,
    model_layers: int,
    model_bytes_per_elem: int,
    # Approx deserialize/weight load bytes per token step (can be 0 if not desired).
    model_weight_bytes: float,
    # Prefill FLOPs scaling params
    prefill_hidden_dim: int,
    prefill_base_n: int,
    enable_duoPD: bool,
    skip_ttft_kv_transfer_penalty: bool,
    # KV-cache transfer params when DuoPD is disabled
    interconnect_bw_gbps_per_device: float,
    num_devices: int,
    gqa_size: int,
    pd_duplex: bool,
    # Chunked prefill (vLLM-like) knobs
    chunked_prefill: bool = False,
    prefill_chunk_tokens: int = 256,
    # Prefill memory-bound override knobs
    prefill_use_memory_model: bool = False,
    prefill_mem_ops_per_nh: float = 9.0,
    prefill_mem_bw_scale: float = 1.0,
) -> Dict[str, float]:
    random.seed(seed)

    # Generate Poisson arrival times
    arrivals: List[Request] = []
    t = 0.0
    for i in range(n):
        t += _exp_interarrival(qps)
        r = Request(i, t)
        if decode_scales_with_context:
            raw_in = (
                random.gauss(float(input_len_mu), float(input_len_sigma))
                if input_len_sigma > 0
                else float(input_len_mu)
            )
            r.input_len = max(1, int(round(raw_in)))
        # sample output length once per request
        if lout_sigma > 0:
            raw_out = random.gauss(float(lout_mu), float(lout_sigma))
            r.lout = max(1, int(round(raw_out)))
        else:
            r.lout = max(1, int(round(float(lout_mu))))
        arrivals.append(r)

    # ----------------------------
    # Stage 1/2 scheduling
    # ----------------------------
    # Default mode: (1) prefill-to-completion as a single server, then (2) decode steps.
    # Chunked-prefill mode: a single GPU timeline that alternates between:
    #   - decode token steps for active requests
    #   - prefill chunk batches for queued/partial requests
    # This mimics the core idea of vLLM chunked prefill (interleave prefill chunks with decode)
    # without modeling the full vLLM token-budget scheduler.

    ready_q: Deque[Request]
    decode_free_t = 0.0

    if not chunked_prefill:
        # Stage 1: prefill batching (single server)
        prefill_q: Deque[Request] = deque()
        prefill_free_t = 0.0
        prefill_done: List[Request] = []

        idx = 0
        while len(prefill_done) < n:
            # If queue empty, wait for next arrival
            if not prefill_q:
                if idx >= n:
                    break
                now = max(prefill_free_t, arrivals[idx].arrival_t)
                while idx < n and arrivals[idx].arrival_t <= now + 1e-15:
                    prefill_q.append(arrivals[idx])
                    idx += 1
            else:
                now = prefill_free_t

            # Add arrivals that happen before/at prefill_free_t (they'll be queued)
            while idx < n and arrivals[idx].arrival_t <= now + 1e-15:
                prefill_q.append(arrivals[idx])
                idx += 1

            if prefill_free_t > now + 1e-15:
                continue

            # Start a batch now
            formed = min(len(prefill_q), max_batch)
            batch = [prefill_q.popleft() for _ in range(formed)]

            # Prefill time scales with FLOPs, driven by total input length N in the batch.
            n_total = sum(
                int(r.input_len) if r.input_len is not None else int(round(float(input_len_mu)))
                for r in batch
            )
            sp_s = _prefill_time_for_batch_tokens(
                pts=pts,
                formed=formed,
                total_tokens=n_total,
                input_len_mu=input_len_mu,
                prefill_hidden_dim=prefill_hidden_dim,
                prefill_base_n=prefill_base_n,
                prefill_use_memory_model=prefill_use_memory_model,
                prefill_mem_ops_per_nh=prefill_mem_ops_per_nh,
                prefill_mem_bw_scale=prefill_mem_bw_scale,
                model_weight_bytes=model_weight_bytes,
                model_hidden_dim=model_hidden_dim,
                model_layers=model_layers,
                model_bytes_per_elem=model_bytes_per_elem,
                gqa_size=gqa_size,
                pd_duplex=pd_duplex,
            )
            start_t = now
            done_t = start_t + sp_s
            prefill_free_t = done_t
            for r in batch:
                r.prefill_done_t = done_t
                prefill_done.append(r)

        # Stage 2: decode (single server) but *batched execution*.
        # Requests become ready at prefill_done_t.
        prefill_done.sort(key=lambda r: (r.prefill_done_t or 0.0, r.arrival_t, r.rid))
        ready_q = deque(prefill_done)
        decode_free_t = 0.0
    else:
        # Chunked prefill mode: init remaining prefill tokens per request.
        for r in arrivals:
            in_len = int(r.input_len) if r.input_len is not None else int(round(float(input_len_mu)))
            r.prefill_remaining_tokens = max(1, int(in_len))

        idx = 0
        # requests with remaining prefill (>0) that have arrived
        # Scheduling policy: round-based chunking (vLLM-like): each prefill batch processes
        # one chunk per request across the queued set (up to max_batch).
        prefill_q = deque()
        ready_q = deque()  # requests whose full prefill is complete
        decode_free_t = 0.0

    # In non-chunked mode, we should already have:
    #   ready_q = deque(prefill_done)
    # and idx/prefill_q must NOT be used.
    if not chunked_prefill:
        idx = n
        prefill_q = deque()

    if decode_max_batch <= 0:
        decode_max_batch = 1
    if decode_window_s < 0:
        decode_window_s = 0.0

    # Active decode set (requests currently generating tokens)
    active: List[Request] = []
    remaining_tokens: Dict[int, int] = {}

    # Collect all per-token TBT samples across all requests (for p50/p90/p99).
    all_token_tbt_s: List[float] = []

    # Decode step loop (or combined loop in chunked mode)
    step_start_t = 0.0
    while True:
        if not chunked_prefill:
            if not (ready_q or active):
                break
        else:
            if not (idx < n or prefill_q or ready_q or active):
                break

        # In chunked mode, admit arrivals up to current time into prefill_q.
        # Important: when the system is idle we may jump time forward to the next arrival,
        # but that idle time must NOT be charged into decode TBT.
        if chunked_prefill:
            now_t = decode_free_t
            if not active and not ready_q and not prefill_q and idx < n:
                now_t = max(now_t, arrivals[idx].arrival_t)
                decode_free_t = now_t
            while idx < n and arrivals[idx].arrival_t <= now_t + 1e-15:
                prefill_q.append(arrivals[idx])
                idx += 1

        # If there are no active decode requests, run prefill chunks only (chunked mode).
        if chunked_prefill and not active and not ready_q and prefill_q:
            # Strict FIFO prefill-only: only serve the head-of-line request(s) in order.
            # We don't move a partially-prefilled request behind newer arrivals.
            formed = min(len(prefill_q), max_batch)
            batch = [prefill_q.popleft() for _ in range(formed)]
            chunk_sz = max(1, int(prefill_chunk_tokens))
            chunk_ctx: List[int] = []
            chunk_new: List[int] = []
            for r in batch:
                rem = int(r.prefill_remaining_tokens or 0)
                take = int(min(rem, chunk_sz))
                ctx = int(getattr(r, "prefill_processed_tokens", 0))
                chunk_ctx.append(ctx)
                chunk_new.append(take)

            sp_s = _prefill_time_for_chunked_microbatch(
                pts=pts,
                formed=formed,
                chunk_contexts_tokens=chunk_ctx,
                chunk_new_tokens=chunk_new,
                prefill_hidden_dim=prefill_hidden_dim,
                prefill_base_n=prefill_base_n,
                prefill_use_memory_model=prefill_use_memory_model,
                prefill_mem_ops_per_nh=prefill_mem_ops_per_nh,
                prefill_mem_bw_scale=prefill_mem_bw_scale,
                model_weight_bytes=model_weight_bytes,
                model_hidden_dim=model_hidden_dim,
                model_layers=model_layers,
                model_bytes_per_elem=model_bytes_per_elem,
                gqa_size=gqa_size,
                pd_duplex=pd_duplex,
            )
            start_t = decode_free_t
            done_t = start_t + sp_s
            decode_free_t = done_t

            for r in batch:
                rem = int(r.prefill_remaining_tokens or 0)
                take = min(rem, chunk_sz)
                rem -= take
                r.prefill_remaining_tokens = rem
                r.prefill_processed_tokens = int(getattr(r, "prefill_processed_tokens", 0)) + int(take)
                if rem <= 0:
                    r.prefill_done_t = done_t
                    ready_q.append(r)
                else:
                    # Round-based chunking: unfinished requests go to the tail.
                    prefill_q.append(r)
            continue
        # If no active requests, jump to next ready time.
        if not active:
            if ready_q:
                head = ready_q[0]
                head_ready = head.prefill_done_t if head.prefill_done_t is not None else float("inf")
                step_start_t = max(decode_free_t, head_ready)
            else:
                step_start_t = decode_free_t
        else:
            step_start_t = decode_free_t

        # Move newly-ready requests into active within the assembly window
        cutoff = step_start_t + decode_window_s
        while ready_q:
            r0 = ready_q[0]
            r0_ready = r0.prefill_done_t if r0.prefill_done_t is not None else float("inf")
            if r0_ready <= cutoff + 1e-15:
                r = ready_q.popleft()
                if r.decode_start_t is None:
                    r.decode_start_t = step_start_t
                remaining_tokens[r.rid] = int(r.lout) if r.lout is not None else 1
                active.append(r)
            else:
                break

        # Chunked-prefill mixed step:
        # If decode is active and prefill_q is non-empty, we run a "mixed" GPU step.
        # The step time (TBT) is modeled as decode_step_time + prefill_chunk_compute_time.
        # This includes *compute* only (no queueing delay), matching the desired behavior.
        mixed_prefill_batch: List[Request] = []
        mixed_prefill_compute_s = 0.0
        if chunked_prefill and active and prefill_q:
            # Round-based mixed prefill: pull from head-of-line; reinsert unfinished to the tail.
            formed_pf = min(len(prefill_q), max_batch)
            mixed_prefill_batch = [prefill_q.popleft() for _ in range(formed_pf)]
            chunk_sz = max(1, int(prefill_chunk_tokens))
            # In a mixed step, the *total* prefill token budget is limited to chunk_sz.
            # i.e., it is NOT chunk_sz * batch_size.
            total_chunk_tokens = 0
            per_req_take: Dict[int, int] = {}
            for r in mixed_prefill_batch:
                if total_chunk_tokens >= chunk_sz:
                    per_req_take[r.rid] = 0
                    continue
                rem = int(r.prefill_remaining_tokens or 0)
                take = min(rem, chunk_sz - total_chunk_tokens)
                per_req_take[r.rid] = int(take)
                total_chunk_tokens += int(take)
            chunk_ctx: List[int] = []
            chunk_new: List[int] = []
            for r in mixed_prefill_batch:
                chunk_ctx.append(int(getattr(r, "prefill_processed_tokens", 0)))
                chunk_new.append(int(per_req_take.get(r.rid, 0)))
            mixed_prefill_compute_s = _prefill_time_for_chunked_microbatch(
                pts=pts,
                formed=max(1, len(mixed_prefill_batch)),
                chunk_contexts_tokens=chunk_ctx,
                chunk_new_tokens=chunk_new,
                prefill_hidden_dim=prefill_hidden_dim,
                prefill_base_n=prefill_base_n,
                prefill_use_memory_model=prefill_use_memory_model,
                prefill_mem_ops_per_nh=prefill_mem_ops_per_nh,
                prefill_mem_bw_scale=prefill_mem_bw_scale,
                model_weight_bytes=model_weight_bytes,
                model_hidden_dim=model_hidden_dim,
                model_layers=model_layers,
                model_bytes_per_elem=model_bytes_per_elem,
                gqa_size=gqa_size,
                pd_duplex=pd_duplex,
            )

        # If still nothing active (can happen if decode_free_t < next ready), jump again.
        if not active:
            decode_free_t = step_start_t
            continue

        # Form a decode step batch from active requests
        active.sort(key=lambda r: (r.decode_start_t or 0.0, r.rid))
        step_batch = active[:decode_max_batch]
        formed = max(1, len(step_batch))
        _, dec_tbt_s_base, calib_lin, _calib_lout = _latency_for_batch(pts, formed)

        # Decode step time derived from *loading size*.
        #
        # Calibration point says: decode_ms is the TBT at lin==calib_lin (e.g., 8K)
        # for the target model (e.g., GPT-175B). We treat that as defining an
        # effective bandwidth for loading KV (and optionally weights).
        #
        # We then compute step load bytes for the batch as:
        #   total_bytes = weights_bytes + sum_i (ctx_len_i * kv_bytes_per_token)
        # and set:
        #   step_tbt_s = total_bytes / bandwidth_bytes_per_s

        # Derive an effective bandwidth from calibration.
        kv_bpt = _kv_cache_bytes_per_token(
            hidden_dim=model_hidden_dim,
            n_layers=model_layers,
            bytes_per_elem=model_bytes_per_elem,
            gqa_size=gqa_size,
        )
        if pd_duplex:
            # PD_duplex: decode KV-cache time is reduced to 1/4.
            kv_bpt *= 0.25

        # Calib assumes a single request at ctx==calib_lin for the calibrated TBT.
        calib_bytes = float(model_weight_bytes) + float(calib_lin) * float(kv_bpt)
        # Guard against div0.
        if float(dec_tbt_s_base) > 0.0:
            bw_bytes_per_s = calib_bytes / float(dec_tbt_s_base)
        else:
            bw_bytes_per_s = float("inf")

        # Batch total bytes.
        total_ctx = sum(
            int(r.input_len) if r.input_len is not None else int(round(float(input_len_mu)))
            for r in step_batch
        )

        if (enable_duoPD and len(step_batch) > 16):
            # When DuoPD is enabled, we assume that the effective bandwidth is improved
            # by a factor proportional to the batch size, up to a limit.
            step_bytes = float(model_weight_bytes) * 2 + float(total_ctx) * float(kv_bpt)
        else: step_bytes = float(model_weight_bytes) + float(total_ctx) * float(kv_bpt)

        step_tbt_s = step_bytes / bw_bytes_per_s if bw_bytes_per_s > 0.0 else 0.0
        if chunked_prefill and mixed_prefill_compute_s > 0.0:
            step_tbt_s += float(mixed_prefill_compute_s)

    # NOTE: KV-cache interconnect transfer penalty is applied to TTFT (first token)
    # per request, not to per-token TBT. See metrics loop below.

        step_end_t = step_start_t + step_tbt_s

        # Apply the mixed prefill chunk progress at the end of the mixed step.
        if chunked_prefill and mixed_prefill_batch:
            chunk_sz = max(1, int(prefill_chunk_tokens))
            for r in mixed_prefill_batch:
                rem = int(r.prefill_remaining_tokens or 0)
                # Apply the same per-request allocation used when computing total_chunk_tokens.
                # If a request got 0 tokens this mixed step, it makes no progress.
                take = 0
                if 'per_req_take' in locals():
                    take = int(per_req_take.get(r.rid, 0))
                take = min(rem, int(take))
                rem -= take
                r.prefill_remaining_tokens = rem
                r.prefill_processed_tokens = int(getattr(r, "prefill_processed_tokens", 0)) + int(take)
                if rem <= 0:
                    r.prefill_done_t = step_end_t
                    ready_q.append(r)
                else:
                    # Round-based chunking: unfinished requests go to the tail.
                    prefill_q.append(r)

        # Advance one token for each request in the step batch
        finished_ids = set()
        for r in step_batch:
            r.decode_tbt_s_list.append(step_tbt_s)
            all_token_tbt_s.append(step_tbt_s)
            remaining_tokens[r.rid] = remaining_tokens.get(r.rid, 1) - 1
            if remaining_tokens[r.rid] <= 0:
                r.decode_done_t = step_end_t
                finished_ids.add(r.rid)

        # Remove finished requests from active
        if finished_ids:
            active = [r for r in active if r.rid not in finished_ids]
            for rid in finished_ids:
                remaining_tokens.pop(rid, None)

        decode_free_t = step_end_t

    # Collect per-request metrics
    # TTFT definition
    # - Default (non-chunked): arrival -> first token (includes prefill if --ttft-includes-prefill).
    # - Chunked-prefill: TTFT should include the time to complete *all* prefill chunks for that
    #   request (plus any queueing delay), and then include the first decode token step time.
    #   Concretely: first_token_time = max(prefill_done_t, decode_start_t) + first_decode_step_tbt
    #   This matches: "complete all chunks + queueing + first token".
    ttft_s: List[float] = []
    e2e_s: List[float] = []
    # Per-request TPOT (mean token time) list.
    tpot_s: List[float] = []

    log_f = None
    token_log_f = None
    if log_path:
        log_f = open(log_path, "w", encoding="utf-8")
        # CSV header
        log_f.write(
            "rid,arrival_s,prefill_done_s,decode_start_s,decode_done_s,"
            "input_len,output_len,ttft_ms,e2e_ms,tpot_ms\n"
        )

        # Per-token decode log
        token_log_path = log_path + ".tokens.csv"
        token_log_f = open(token_log_path, "w", encoding="utf-8")
    if token_log_f is not None:
        token_log_f.write("rid,token_idx,tbt_ms\n")

    # For TTFT/TBT we use per-token time derived from the (possibly scaled) decode.
    # We re-derive per-token per-request inside the loop when scaling is enabled.
    # For TTFT/TBT, per-token time is approximated from batch=1 calibration.
    _, dec_tbt_s_1, calib_lin_1, calib_lout_1 = _latency_for_batch(pts, 1)
    base_per_token_s = float(dec_tbt_s_1)

    for r in arrivals:
        if r.decode_done_t is None or r.decode_start_t is None or r.prefill_done_t is None:
            continue

        per_token_s = base_per_token_s
        # scale first-token time based on this request's own input length and output length
        if decode_scales_with_context and calib_lin_1 > 0:
            ctx_len = int(r.input_len) if r.input_len is not None else int(round(float(input_len_mu)))
            per_token_s *= max(0.0, float(ctx_len) / float(calib_lin_1))
        if calib_lout_1 > 0:
            out_len = int(r.lout) if r.lout is not None else int(calib_lout_1)
            per_token_s *= max(0.0, float(out_len) / float(calib_lout_1))

        # First token appears after one decode step.
        # In chunked-prefill mode, a request cannot produce a token before its full prefill
        # has completed (after all chunks). So first token time must be based on prefill_done_t.
        if chunked_prefill and r.decode_tbt_s_list:
            first_step_s = float(r.decode_tbt_s_list[0])
            first_tok_t = max(float(r.prefill_done_t), float(r.decode_start_t)) + first_step_s
        else:
            first_tok_t = r.decode_start_t + per_token_s

        # If DuoPD is disabled, add a one-time KV-cache transfer penalty to TTFT.
        # We approximate KV bytes transferred as ctx_len * kv_bytes_per_token.
        if (not enable_duoPD) and (not chunked_prefill) and (not skip_ttft_kv_transfer_penalty):
            kv_bpt_1 = _kv_cache_bytes_per_token(
                hidden_dim=model_hidden_dim,
                n_layers=model_layers,
                bytes_per_elem=model_bytes_per_elem,
                gqa_size=gqa_size,
            )
            ctx_len = int(r.input_len) if r.input_len is not None else int(round(float(input_len_mu)))
            kv_transfer_bytes = float(ctx_len) * float(kv_bpt_1)
            bw_per_dev_Bps = float(interconnect_bw_gbps_per_device) * 1e9
            agg_bw_Bps = bw_per_dev_Bps * float(max(1, int(num_devices)))
            first_tok_t += kv_transfer_bytes / agg_bw_Bps if agg_bw_Bps > 0.0 else 0.0
        if include_prefill_in_ttft:
            ttft = first_tok_t - r.arrival_t
        else:
            # exclude prefill: queueing-to-decode-start
            ttft = first_tok_t - r.prefill_done_t 

        e2e = r.decode_done_t - r.arrival_t

        # Per-request TPOT: average of per-token TBT samples for that request.
        if r.decode_tbt_s_list:
            tpot = float(sum(r.decode_tbt_s_list)) / float(len(r.decode_tbt_s_list))
        else:
            out_len = int(r.lout) if r.lout is not None else 1
            tpot = (r.decode_done_t - r.decode_start_t) / max(out_len, 1)

        ttft_s.append(ttft)
        e2e_s.append(e2e)
        tpot_s.append(tpot)

        if log_f is not None:
            log_f.write(
                "{},{:.9f},{:.9f},{:.9f},{:.9f},{:.3f},{:.3f},{:.6f},{:.6f},{:.6f}\n".format(
                    int(r.rid),
                    float(r.arrival_t),
                    float(r.prefill_done_t),
                    float(r.decode_start_t),
                    float(r.decode_done_t),
                    float(r.input_len) if r.input_len is not None else float("nan"),
                    float(r.lout) if r.lout is not None else float("nan"),
                    1000.0 * float(ttft),
                    1000.0 * float(e2e),
                    1000.0 * float(tpot),
                )
            )

        # Dump per-token TBT list (inter-token latency sequence).
        # Note: this is the step-based decoder's recorded TBT for each produced token.
        if token_log_f is not None:
            for i_tok, tbt_s_i in enumerate(r.decode_tbt_s_list):
                token_log_f.write("{},{},{:.6f}\n".format(int(r.rid), int(i_tok), 1000.0 * float(tbt_s_i)))

    if log_f is not None:
        log_f.close()
    if token_log_f is not None:
        token_log_f.close()

    ttft_s.sort()
    e2e_s.sort()
    tpot_s.sort()
    all_token_tbt_s.sort()

    # Throughput (completed):
    # - req/s: completed requests divided by makespan (from first arrival to last completion).
    # - tok/s: total output tokens divided by makespan.
    # We use decode_done_t as completion time.
    completed = [r for r in arrivals if r.decode_done_t is not None]
    if completed:
        first_arrival_t = min(r.arrival_t for r in completed)
        last_done_t = max(float(r.decode_done_t) for r in completed)
        makespan_s = max(1e-12, float(last_done_t - first_arrival_t))
        total_out_tokens = sum(int(r.lout) if r.lout is not None else 0 for r in completed)
        throughput_qps = float(len(completed)) / makespan_s
        throughput_tps = float(total_out_tokens) / makespan_s
    else:
        throughput_qps = 0.0
        throughput_tps = 0.0

    return {
        "n": float(len(ttft_s)),
    "throughput_qps": float(throughput_qps),
    "throughput_tps": float(throughput_tps),
        "p50_ttft_ms": 1000.0 * _percentile(ttft_s, 0.50),
        "p90_ttft_ms": 1000.0 * _percentile(ttft_s, 0.90),
        "p99_ttft_ms": 1000.0 * _percentile(ttft_s, 0.99),
        "p50_e2e_ms": 1000.0 * _percentile(e2e_s, 0.50),
        "p90_e2e_ms": 1000.0 * _percentile(e2e_s, 0.90),
        "p99_e2e_ms": 1000.0 * _percentile(e2e_s, 0.99),
    # TBT percentiles are over *all* token steps of *all* requests.
    "p50_tbt_ms": 1000.0 * _percentile(all_token_tbt_s, 0.50) if all_token_tbt_s else 0.0,
    "p90_tbt_ms": 1000.0 * _percentile(all_token_tbt_s, 0.90) if all_token_tbt_s else 0.0,
    "p99_tbt_ms": 1000.0 * _percentile(all_token_tbt_s, 0.99) if all_token_tbt_s else 0.0,

    # TPOT percentiles are over per-request averages.
    "p50_tpot_ms": 1000.0 * _percentile(tpot_s, 0.50),
    "p90_tpot_ms": 1000.0 * _percentile(tpot_s, 0.90),
    "p99_tpot_ms": 1000.0 * _percentile(tpot_s, 0.99),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qps", type=float, required=True, help="Arrival rate (req/s), Poisson")
    ap.add_argument("--calib", type=str, required=True, help="Calibration JSON path")
    ap.add_argument("--n", type=int, default=1000, help="Number of requests to simulate")
    ap.add_argument("--max-batch", type=int, default=32, help="Max dynamic prefill batch size")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--decode-scales-with-context",
        action="store_true",
        help="If set, decode_ms scales linearly with a modeled per-request context length.",
    )
    ap.add_argument(
        "--input-len-mu",
        type=float,
        default=32768.0,
        help="Mean input/context length (tokens) for decode scaling.",
    )
    ap.add_argument(
        "--input-len-sigma",
        type=float,
        default=0.0,
        help="Stddev input/context length (tokens). 0 means fixed at mu.",
    )
    ap.add_argument(
        "--lout-mu",
        type=float,
        default=128.0,
        help="Mean output length (tokens) for decode batching/latency scaling.",
    )
    ap.add_argument(
        "--lout-sigma",
        type=float,
        default=0.0,
        help="Stddev output length (tokens). 0 means fixed at mu.",
    )

    # Prefill FLOPs scaling (based on total batch input length N and hidden dim H)
    ap.add_argument(
        "--prefill-hidden-dim",
        type=int,
        default=12288,
        help="Hidden dim H used in prefill FLOPs formula (default 12K).",
    )
    ap.add_argument(
        "--prefill-base-n",
        type=int,
        default=32768,
        help="Base N (tokens) for prefill calibration point (default 32768).",
    )

    ap.add_argument(
        "--prefill-use-memory-model",
        action="store_true",
        help=(
            "If set, prefill time becomes max(compute_time, memory_access_time). "
            "Memory_access_time is estimated using a decode-like bandwidth model + an explicit activation access term."
        ),
    )
    ap.add_argument(
        "--prefill-mem-ops-per-nh",
        type=float,
        default=9.0,
        help=(
            "Prefill activation memory accesses per token per hidden element (multiplier for N*H*bytes). "
            "Default 9 ~= QKV+attn+proj+LN1+FFN1+FFN2+FFN3+LN2."
        ),
    )
    ap.add_argument(
        "--prefill-mem-bw-scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor applied to the derived bandwidth when estimating prefill memory time. "
            "Use <1 to model worse prefill locality/bandwidth than decode."
        ),
    )

    ap.add_argument(
        "--chunked-prefill",
        action="store_true",
        help=(
            "If set, enable a vLLM-like chunked-prefill scheduler that interleaves prefill chunks and decode steps."
        ),
    )
    ap.add_argument(
        "--prefill-chunk-tokens",
        type=int,
        default=256,
        help="Chunk size (tokens) per request per prefill chunk batch when --chunked-prefill is enabled.",
    )
    ap.add_argument(
        "--decode-max-batch",
        type=int,
        default=0,
        help="Max decode batch size (0 means use --max-batch).",
    )
    ap.add_argument(
        "--decode-window-ms",
        type=float,
        default=0.0,
        help="Decode batch assembly window in ms.",
    )

    # Model/KV-cache parameters for load-size-based decode.
    ap.add_argument("--model-hidden-dim", type=int, default=12288, help="Model hidden dim (e.g., 12K for GPT-175B)")
    ap.add_argument("--model-layers", type=int, default=96, help="Number of transformer layers (e.g., 96 for GPT-175B)")
    ap.add_argument(
        "--model-bytes-per-elem",
        type=int,
        default=2,
        help="Bytes per element for KV cache (FP16=2, FP8=1, BF16=2, FP32=4)",
    )
    ap.add_argument(
        "--model-weight-bytes",
        type=float,
        default=3.5e11,
        help=(
            "Approx model weight bytes added to load per token step. "
            "Set to 3.5e11 for ~350GB if you want to model weight streaming; "
            "leave 0 if weights are assumed cached."
        ),
    )
    ap.add_argument(
        "--interconnect-bw-gbps-per-device",
        type=float,
        default=900.0,
        help="Interconnect bandwidth per device in GB/s (used when DuoPD is disabled).",
    )
    ap.add_argument(
        "--num-devices",
        type=int,
        default=1,
        help="Number of devices participating in KV cache transfer.",
    )

    ap.add_argument(
        "--gqa-size",
        type=int,
        default=1,
        help=(
            "GQA grouping factor (num Q heads / num KV heads). If > 1, KV-cache bytes per token are divided "
            "by this factor (fewer KV heads). Default: 1."
        ),
    )

    ap.add_argument(
        "--log",
        type=str,
        default=None,
        help=(
            "Optional CSV path to dump per-request TTFT/TBT/E2E and lengths. "
            "If a relative path (or filename) is given, it will be placed under ./results/."
        ),
    )
    ap.add_argument(
        "--ttft-includes-prefill",
        action="store_true",
        help="If set, TTFT is measured from arrival (includes prefill).",
    )

    ap.add_argument(
        "--enable-duoPD",
        default=False,
        action="store_true",
        help="If set, enables DuoPD for improved performance.",
    )
    ap.add_argument(
        "--skip-ttft-kv-transfer-penalty",
        default=False,
        action="store_true",
        help=(
            "If set, do not add the one-time KV-transfer TTFT penalty even when DuoPD is disabled "
            "(non-chunked path)."
        ),
    )
    ap.add_argument(
        "--pd-duplex",
        default=False,
        action="store_true",
        help="If set, enables PD_duplex (decode KV-cache time scaled by 1/4).",
    )

    args = ap.parse_args()

    # Normalize log path:
    # - If user passed a relative path or filename, put it under ~/results/.
    # - Always expand ~.
    # - Create the directory so logging doesn't fail.
    if args.log:
        log_arg = args.log
        if not os.path.isabs(log_arg):
            log_arg = os.path.join(os.path.expanduser("~/results"), log_arg)
        log_arg = os.path.expanduser(log_arg)
        os.makedirs(os.path.dirname(log_arg), exist_ok=True)
        args.log = log_arg

    pts = parse_calib_json(args.calib)
    if args.max_batch <= 0:
        raise SystemExit("--max-batch must be > 0")

    skip_ttft_kv_transfer_penalty = bool(args.skip_ttft_kv_transfer_penalty) or _is_sm_ratio_calib(args.calib)

    res = simulate(
        qps=args.qps,
        pts=pts,
        n=args.n,
        max_batch=args.max_batch,
        seed=args.seed,
        include_prefill_in_ttft=args.ttft_includes_prefill,
        decode_scales_with_context=args.decode_scales_with_context,
        input_len_mu=args.input_len_mu,
        input_len_sigma=args.input_len_sigma,
        decode_max_batch=args.decode_max_batch if args.decode_max_batch > 0 else args.max_batch,
        decode_window_s=args.decode_window_ms / 1000.0,
        lout_mu=args.lout_mu,
        lout_sigma=args.lout_sigma,
        log_path=args.log,
        model_hidden_dim=args.model_hidden_dim,
        model_layers=args.model_layers,
        model_bytes_per_elem=args.model_bytes_per_elem,
        model_weight_bytes=args.model_weight_bytes,
        prefill_hidden_dim=args.prefill_hidden_dim,
        prefill_base_n=args.prefill_base_n,
        enable_duoPD=args.enable_duoPD,
        skip_ttft_kv_transfer_penalty=skip_ttft_kv_transfer_penalty,
        interconnect_bw_gbps_per_device=args.interconnect_bw_gbps_per_device,
        num_devices=args.num_devices,
    gqa_size=args.gqa_size,
    pd_duplex=args.pd_duplex,
        chunked_prefill=args.chunked_prefill,
        prefill_chunk_tokens=args.prefill_chunk_tokens,
    prefill_use_memory_model=bool(args.prefill_use_memory_model),
    prefill_mem_ops_per_nh=float(args.prefill_mem_ops_per_nh),
    prefill_mem_bw_scale=float(args.prefill_mem_bw_scale),
    )

    print(
        "qps={:.6g} n={} max_batch={} enable_duoPD={} pd_duplex={} num_devices={} "
        "throughput_qps={:.6f} throughput_tps={:.6f} "
        "p50_TTFT_ms={:.3f} p90_TTFT_ms={:.3f} p99_TTFT_ms={:.3f} "
        "p50_E2E_ms={:.3f} p90_E2E_ms={:.3f} p99_E2E_ms={:.3f} "
        "p50_TBT_ms={:.6f} p90_TBT_ms={:.6f} p99_TBT_ms={:.6f} "
        "p50_TPOT_ms={:.6f} p90_TPOT_ms={:.6f} p99_TPOT_ms={:.6f}".format(
            args.qps,
            int(res["n"]),
            args.max_batch,
            1 if args.enable_duoPD else 0,
            1 if args.pd_duplex else 0,
            int(args.num_devices),
            res["throughput_qps"],
            res["throughput_tps"],
            res["p50_ttft_ms"],
            res["p90_ttft_ms"],
            res["p99_ttft_ms"],
            res["p50_e2e_ms"],
            res["p90_e2e_ms"],
            res["p99_e2e_ms"],
            res["p50_tbt_ms"],
            res["p90_tbt_ms"],
            res["p99_tbt_ms"],
            res["p50_tpot_ms"],
            res["p90_tpot_ms"],
            res["p99_tpot_ms"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
