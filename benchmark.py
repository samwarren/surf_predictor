#!/usr/bin/env python3
"""
Benchmark: vLLM vs HuggingFace Transformers — Batched Inference Latency.

Sends increasing numbers of concurrent surf-forecast prompts to both backends
and plots total latency vs batch size.

Usage:
    python benchmark.py                  # default batch sizes: 1,2,4,8,16
    python benchmark.py --sizes 1 2 4 8  # custom batch sizes
    python benchmark.py --output bench.png
"""

import argparse
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Different surf spots to create varied prompts
SPOTS = [
    ("Ocean Beach, SF", "46237", "W/NW swell, 15s period, 6ft faces"),
    ("Mavericks, Half Moon Bay", "46012", "Large NW groundswell, 18s period, 20ft+ faces"),
    ("Pacifica/Linda Mar", "46237", "Small W swell, 10s period, 3ft faces"),
    ("Santa Cruz – Steamer Lane", "46042", "W swell, 14s period, 5ft faces"),
    ("Bolinas", "46237", "NW swell, 12s period, 4ft faces"),
    ("Fort Point, SF", "46237", "Refracted NW swell, 16s period, 8ft faces"),
    ("Montara", "46012", "WNW swell, 13s period, 5ft faces"),
    ("Rockaway Beach, Pacifica", "46237", "SW windswell, 8s period, 2ft faces"),
    ("Stinson Beach", "46237", "NW swell, 14s period, 4ft faces"),
    ("Moss Landing", "46042", "W swell, 12s period, 3ft faces"),
    ("Pleasure Point, SC", "46042", "S swell, 15s period, 4ft faces"),
    ("Capitola", "46042", "SW swell, 11s period, 3ft faces"),
    ("Manresa", "46042", "W swell, 13s period, 4ft faces"),
    ("Waddell Creek", "46042", "NW swell, 16s period, 6ft faces"),
    ("Ano Nuevo", "46042", "WNW swell, 15s period, 7ft faces"),
    ("Mavs outer reef", "46012", "NW swell, 20s period, 30ft faces"),
]


def make_prompt(spot_name: str, conditions: str) -> str:
    """Generate a surf forecast prompt for a given spot."""
    return f"""You are an expert surf forecaster for {spot_name}.
Current conditions: {conditions}.

Rate the surf quality for the next 6 hours on a scale of 1-5:
1=Poor, 2=Poor+, 3=Fair, 4=Good, 5=Epic.

Output ONLY a comma-separated list of 6 integers (1-5), nothing else.
Predictions:"""


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def benchmark_vllm(prompts_list: list[list[str]], llm: LLM, sampling_params: SamplingParams) -> list[float]:
    """Benchmark vLLM with increasing batch sizes. Returns list of elapsed times."""
    times = []
    for prompts in prompts_list:
        t0 = time.time()
        llm.generate(prompts, sampling_params)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"    vLLM  batch={len(prompts):>2}  -> {elapsed:.2f}s")
    return times


def benchmark_hf(prompts_list: list[list[str]], model, tokenizer, device: str) -> list[float]:
    """Benchmark HuggingFace transformers (sequential per prompt). Returns list of elapsed times."""
    times = []
    for prompts in prompts_list:
        t0 = time.time()
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=50, temperature=0.1, do_sample=True)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"    HF    batch={len(prompts):>2}  -> {elapsed:.2f}s")
    return times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM vs HuggingFace batched inference")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                        help="Batch sizes to test (default: 1 2 4 8 16)")
    parser.add_argument("--output", type=str, default="benchmark_plot.png",
                        help="Output plot filename (default: benchmark_plot.png)")
    args = parser.parse_args()

    batch_sizes = sorted(args.sizes)
    max_needed = max(batch_sizes)
    all_prompts = [make_prompt(SPOTS[i % len(SPOTS)][0], SPOTS[i % len(SPOTS)][2])
                   for i in range(max_needed)]
    prompts_list = [all_prompts[:n] for n in batch_sizes]

    # --- Load vLLM ---
    print("\n[1/4] Loading vLLM model...")
    llm = LLM(model=MODEL_NAME)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=50)

    # Warmup
    print("[2/4] Warming up vLLM...")
    llm.generate([all_prompts[0]], sampling_params)

    print("[2/4] Running vLLM benchmark...")
    vllm_times = benchmark_vllm(prompts_list, llm, sampling_params)

    # Free vLLM resources
    del llm
    import gc; gc.collect()

    # --- Load HuggingFace ---
    print("\n[3/4] Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = "cpu"  # Force CPU to match vLLM (which has no MPS support)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)

    # Warmup
    print("[3/4] Warming up HuggingFace...")
    inputs = tokenizer(all_prompts[0], return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=50, temperature=0.1, do_sample=True)

    print("[3/4] Running HuggingFace benchmark...")
    hf_times = benchmark_hf(prompts_list, model, tokenizer, device)

    del model, tokenizer
    gc.collect()

    # --- Results table ---
    print("\n[4/4] Results\n")
    results = pd.DataFrame({
        "Batch Size": batch_sizes,
        "vLLM (s)": [f"{t:.2f}" for t in vllm_times],
        "HF (s)": [f"{t:.2f}" for t in hf_times],
        "Speedup": [f"{h/v:.2f}x" for v, h in zip(vllm_times, hf_times)],
    })
    from tabulate import tabulate
    print(tabulate(results, headers="keys", tablefmt="fancy_grid", showindex=False))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(batch_sizes, vllm_times, "o-", color="#2196F3", linewidth=2.5,
            markersize=8, label="vLLM (batched)")
    ax.plot(batch_sizes, hf_times, "s-", color="#FF5722", linewidth=2.5,
            markersize=8, label="HuggingFace (sequential)")

    ax.set_xlabel("Number of Concurrent Prompts", fontsize=13)
    ax.set_ylabel("Total Latency (seconds)", fontsize=13)
    ax.set_title("vLLM vs HuggingFace — Batched Inference Latency\n"
                 f"Model: {MODEL_NAME} | Both on CPU",
                 fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(batch_sizes)

    # Annotate speedup on the HF line
    for bs, vt, ht in zip(batch_sizes, vllm_times, hf_times):
        speedup = ht / vt if vt > 0 else 0
        if speedup > 1.1:
            ax.annotate(f"{speedup:.1f}x", xy=(bs, ht), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=10,
                        color="#FF5722", fontweight="bold")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()
