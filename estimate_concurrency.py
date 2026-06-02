#!/usr/bin/env python3
"""Estimate LLM inference max concurrency from GPU memory constraints."""

import argparse
import sys

MODEL_CONFIGS = {
    "deepseekv4_pro": {
        "full_name": "DeepSeek-V4-Pro",
        "mla": False,        "attention_mechanism": "CSA + HCA (混合注意力)",
        "num_layers": 61,
        "latent_kv_dim": None,
        "num_kv_heads": 128,
        "head_dim": 512,
        "hidden_dim": 7168,
        "intermediate_dim": 3072,
        "vocab_size": 129280,
        "moe": True,
        "num_experts": 385,
        "num_routed_experts": 384,
        "num_shared_experts": 1,
        "num_experts_per_token": 6,
        "activated_params_billions": 49.0,
        "total_params_billions": 1600.0,
        "context_length": 1048576
    },
    "deepseekv4_flash": {
        "full_name": "DeepSeek-V4-Flash",
        "mla": False,        "attention_mechanism": "CSA + HCA (混合注意力)",
        "num_layers": 43,
        "latent_kv_dim": None,
        "num_kv_heads": 64,
        "head_dim": 512,
        "hidden_dim": 4096,
        "intermediate_dim": 2048,
        "vocab_size": 129280,
        "moe": True,
        "num_experts": 257,
        "num_routed_experts": 256,
        "num_shared_experts": 1,
        "num_experts_per_token": 6,
        "activated_params_billions": 13.0,
        "total_params_billions": 284.0,
        "context_length": 1048576
    },
    "qwen3.6_35b_a3b": {
        "full_name": "Qwen3.6-35B-A3B",
        "mla": False,        "attention_mechanism": "Gated DeltaNet + Gated Attention",
        "num_layers": 40,
        "latent_kv_dim": None,
        "num_kv_heads": 2,
        "head_dim": 256,
        "hidden_dim": 2048,
        "intermediate_dim": 512,
        "vocab_size": 248320,
        "moe": True,
        "num_experts": 257,
        "num_routed_experts": 256,
        "num_shared_experts": 1,
        "num_experts_per_token": 9,
        "activated_params_billions": 3.0,
        "total_params_billions": 35.0,
        "context_length": 262144
    },
    "minimax_m2.5": {
        "full_name": "MiniMax-M2.5",
        "mla": False,
        "attention_mechanism": "标准多头注意力",
        "num_layers": 62,
        "latent_kv_dim": None,
        "num_kv_heads": 8,
        "head_dim": 128,
        "hidden_dim": 6144,
        "intermediate_dim": 16384,
        "vocab_size": 200064,
        "moe": True,
        "num_experts": 256,
        "num_routed_experts": 256,
        "num_shared_experts": 0,
        "num_experts_per_token": 8,
        "activated_params_billions": 10.0,
        "total_params_billions": 229.0,
        "context_length": 128000
    },
    "kimi_k2.6": {
        "full_name": "Kimi-K2.6",
        "mla": True,
        "attention_mechanism": "多潜变量注意力(MLA)",
        "num_layers": 61,
        "latent_kv_dim": 128,
        "rope_head_dim": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "hidden_dim": 5120,
        "intermediate_dim": 13824,
        "vocab_size": 160000,
        "moe": True,
        "num_experts": 385,
        "num_routed_experts": 384,
        "num_shared_experts": 1,
        "num_experts_per_token": 9,
        "activated_params_billions": 32.0,
        "total_params_billions": 1040.0,
        "context_length": 262144
    },
    "glm5.1": {
        "full_name": "GLM-5.1",
        "mla": True,
        "attention_mechanism": "多潜变量注意力(MLA) + DSA",
        "num_layers": 78,
        "latent_kv_dim": 512,
        "rope_head_dim": 64,
        "num_kv_heads": 64,
        "head_dim": 64,
        "hidden_dim": 6144,
        "intermediate_dim": 18432,
        "vocab_size": 152064,
        "moe": True,
        "num_experts": 257,
        "num_routed_experts": 256,
        "num_shared_experts": 1,
        "num_experts_per_token": 9,
        "activated_params_billions": 40.0,
        "total_params_billions": 754.0,
        "context_length": 202752
    },
    "kimi_k2.5": {
        "full_name": "Kimi-K2.5",
        "mla": True,
        "attention_mechanism": "多潜变量注意力(MLA)",
        "num_layers": 61,
        "latent_kv_dim": 128,
        "rope_head_dim": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "hidden_dim": 7168,
        "intermediate_dim": 16384,
        "vocab_size": 160000,
        "moe": True,
        "num_experts": 385,
        "num_routed_experts": 384,
        "num_shared_experts": 1,
        "num_experts_per_token": 8,
        "activated_params_billions": 32.0,
        "total_params_billions": 1040.0,
        "context_length": 262144
    },
    "glm5": {
        "full_name": "GLM-5",
        "mla": True,
        "attention_mechanism": "多潜变量注意力(MLA) + DSA",
        "num_layers": 78,
        "latent_kv_dim": 512,
        "rope_head_dim": 64,
        "num_kv_heads": 64,
        "head_dim": 64,
        "hidden_dim": 6144,
        "intermediate_dim": 12288,
        "vocab_size": 152064,
        "moe": True,
        "num_experts": 257,
        "num_routed_experts": 256,
        "num_shared_experts": 1,
        "num_experts_per_token": 8,
        "activated_params_billions": 40.0,
        "total_params_billions": 744.0,
        "context_length": 202752
    }
}


def get_dtype_bytes(dtype_str):
    bytes_map = {
        "fp16": 2,
        "bf16": 2,
        "fp8": 1,
        "int8": 1,
        "int4": 0.5,
        "nvfp4": 0.5,
    }
    if dtype_str not in bytes_map:
        raise ValueError(
            f"Unknown dtype: {dtype_str}. Options: {list(bytes_map.keys())}"
        )
    return bytes_map[dtype_str]


def get_kv_per_token(config):
    mechanism = config.get("attention_mechanism", "GQA")

    if config.get("mla") or "MLA" in mechanism:
        kv_per_layer = config["latent_kv_dim"]
        rope_head_dim = config.get("rope_head_dim", config["head_dim"])
        return config["num_layers"] * (kv_per_layer + rope_head_dim)

    if "Gated DeltaNet" in mechanism or "DeltaNet" in mechanism:
        return 0

    kv_per_layer = config["num_kv_heads"] * config["head_dim"]
    return 2 * config["num_layers"] * kv_per_layer


def get_linear_attention_state_size(config, kv_bytes):
    mechanism = config.get("attention_mechanism", "")
    if "Gated DeltaNet" not in mechanism and "DeltaNet" not in mechanism:
        return 0
    state_size = config["hidden_dim"] * 2 * config["head_dim"] * kv_bytes
    return state_size


def estimate_concurrency(
    model_key,
    gpu_memory_gb,
    tp_size,
    context_length,
    weight_dtype,
    kv_cache_dtype,
    mem_fraction_static,
    kv_cache_bytes_override,
    bytes_per_token_override,
    overhead_rate=0.90,
    overhead_fixed_gb=0.0,
    total_params_billions_override=None,
):
    if model_key not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model: {model_key}. Available: {available}")

    config = MODEL_CONFIGS[model_key]

    weight_bytes = get_dtype_bytes(weight_dtype)
    kv_bytes = get_dtype_bytes(kv_cache_dtype)

    if total_params_billions_override is not None:
        total_params = total_params_billions_override * 1e9
        total_params_billions = total_params_billions_override
    else:
        total_params = config["total_params_billions"] * 1e9
        total_params_billions = config["total_params_billions"]

    model_weight_gb = total_params * weight_bytes / (1024 ** 3)
    per_gpu_weight_gb = model_weight_gb / tp_size

    per_gpu_kv_gb = gpu_memory_gb * mem_fraction_static - per_gpu_weight_gb

    if per_gpu_kv_gb < 0:
        raise ValueError(
            f"GPU memory insufficient: per-GPU weight ({per_gpu_weight_gb:.1f} GB) "
            f"exceeds available static portion "
            f"({gpu_memory_gb * mem_fraction_static:.1f} GB)"
        )

    per_gpu_after_fixed = per_gpu_kv_gb - overhead_fixed_gb

    if per_gpu_after_fixed < 0:
        raise ValueError(
            f"Fixed overhead ({overhead_fixed_gb:.1f} GB) exceeds per-GPU "
            f"KV pool ({per_gpu_kv_gb:.1f} GB)"
        )

    per_gpu_effective_kv = per_gpu_after_fixed * overhead_rate
    total_effective_kv_gb = per_gpu_effective_kv * tp_size

    kv_per_token = get_kv_per_token(config)
    kv_per_token_bytes = kv_per_token * kv_bytes

    linear_state_bytes = get_linear_attention_state_size(config, kv_bytes)
    linear_state_gb = linear_state_bytes / (1024 ** 3)
    linear_state_per_gpu = linear_state_gb / tp_size
    per_gpu_effective_kv = per_gpu_effective_kv - linear_state_per_gpu

    if per_gpu_effective_kv < 0:
        per_gpu_effective_kv = 0

    kv_cache_total_bytes = total_effective_kv_gb * (1024 ** 3)

    if config.get("mla"):
        # MLA模式下，KV Cache在每个GPU上完整保存（不进行TP切分），
        # 因此并发数受限于单GPU的KV容量。
        kv_cache_budget_bytes = per_gpu_effective_kv * (1024 ** 3)
    else:
        kv_cache_budget_bytes = kv_cache_total_bytes

    if kv_cache_bytes_override is not None:
        kv_cache_budget_bytes = kv_cache_bytes_override

    single_token_bytes = kv_per_token_bytes
    if bytes_per_token_override is not None:
        single_token_bytes = bytes_per_token_override

    # DeepSeek V4 Pro: KV cache usage = tokens * 4444B + 21040 KB
    if model_key == "deepseekv4_pro":
        single_token_bytes = 4444
        kv_fixed_bytes = 21040 * 1024  # 21040 KB -> bytes
        kv_per_sequence_bytes = single_token_bytes * context_length + kv_fixed_bytes
    elif model_key == "deepseekv4_flash":
        single_token_bytes = 3360
        kv_fixed_bytes = 21040 * 1024  # 21040 KB -> bytes
        kv_per_sequence_bytes = single_token_bytes * context_length + kv_fixed_bytes
    else:
        kv_per_sequence_bytes = single_token_bytes * context_length

    if kv_per_sequence_bytes > 0:
        max_concurrency = int(kv_cache_budget_bytes / kv_per_sequence_bytes)
    else:
        max_concurrency = -1

    return {
        "model": model_key,
        "full_name": config["full_name"],
        "total_params_billions": total_params_billions,
        "params_override": total_params_billions_override is not None,
        "mla": config.get("mla", False),
        "rope_head_dim": config.get("rope_head_dim"),
        "attention_mechanism": config.get("attention_mechanism", "GQA"),
        "linear_state_bytes": linear_state_bytes,
        "linear_state_gb": linear_state_gb,
        "linear_state_per_gpu": linear_state_per_gpu,
        "gpu_memory_gb": gpu_memory_gb,
        "tp_size": tp_size,
        "context_length": context_length,
        "weight_dtype": weight_dtype,
        "kv_cache_dtype": kv_cache_dtype,
        "mem_fraction_static": mem_fraction_static,
        "overhead_rate": overhead_rate,
        "overhead_fixed_gb": overhead_fixed_gb,
        "weight_bytes_per_param": weight_bytes,
        "kv_cache_bytes_per_element": kv_bytes,
        "model_weight_gb": model_weight_gb,
        "per_gpu_weight_gb": per_gpu_weight_gb,
        "per_gpu_kv_gb": per_gpu_kv_gb,
        "per_gpu_after_fixed_gb": per_gpu_after_fixed,
        "per_gpu_effective_kv_gb": per_gpu_effective_kv,
        "total_effective_kv_gb": total_effective_kv_gb,
        "kv_per_token_dim": kv_per_token,
        "kv_per_token_bytes": kv_per_token_bytes,
        "single_token_bytes": single_token_bytes,
        "kv_per_sequence_bytes_gb": kv_per_sequence_bytes / (1024 ** 3),
        "max_concurrency": max_concurrency,
    }


def print_result(r):
    print(f"Model: {r['full_name']} ({r['model']})")
    print(f"MLA: {r['mla']}")
    if r['mla']:
        print(f"MLA rope_head_dim: {r['rope_head_dim']}")
    print(f"Attention mechanism: {r['attention_mechanism']}")
    if r.get("params_override"):
        print(f"Total params (user-specified): {r['total_params_billions']}B")
    else:
        print(f"Total params (declared): {r['total_params_billions']}B")
    print(f"GPU memory: {r['gpu_memory_gb']} GB x {r['tp_size']} GPUs "
          f"(TP={r['tp_size']})")
    print(f"Context length: {r['context_length']}")
    print(f"Weight dtype: {r['weight_dtype']} "
          f"({r['weight_bytes_per_param']}B/param)")
    print(f"KV cache dtype: {r['kv_cache_dtype']} "
          f"({r['kv_cache_bytes_per_element']}B/element)")
    print(f"Memory fraction (static): {r['mem_fraction_static']}")
    print(f"Overhead rate: {r['overhead_rate']}")
    print(f"Overhead fixed (per-GPU): {r['overhead_fixed_gb']} GB")
    if r.get("linear_state_bytes", 0) > 0:
        print(f"Linear attention fixed state (total): {r['linear_state_gb']:.6f} GB")
        print(f"Linear attention fixed state (per-GPU): {r['linear_state_per_gpu']:.6f} GB")
    print("---")
    print(f"Model weight (total): {r['model_weight_gb']:.1f} GB")
    print(f"Model weight (per-GPU): {r['per_gpu_weight_gb']:.1f} GB")
    print(f"KV cache pool (per-GPU, raw): {r['per_gpu_kv_gb']:.1f} GB")
    print(f"KV cache pool (per-GPU, after fixed deduction): "
          f"{r['per_gpu_after_fixed_gb']:.1f} GB")
    print(f"KV cache pool (per-GPU, effective): "
          f"{r['per_gpu_effective_kv_gb']:.1f} GB")
    print(f"KV cache pool (total, effective): "
          f"{r['total_effective_kv_gb']:.1f} GB")
    print(f"KV per token (dimension): {r['kv_per_token_dim']}")
    print(f"KV per token (bytes): {r['kv_per_token_bytes']}")
    print(f"KV per sequence: {r['kv_per_sequence_bytes_gb']:.4f} GB")
    print("---")
    if r['max_concurrency'] < 0:
        print("Estimated max concurrency: N/A (zero per-token KV, bounded by other memory)")
    else:
        print(f"Estimated max concurrency: {r['max_concurrency']}")
    print()


def list_models():
    print("Available models:")
    for key, cfg in MODEL_CONFIGS.items():
        attn = cfg.get("attention_mechanism", "GQA")
        print(f"  {key:12s}  {cfg['full_name']:15s}  "
              f"total={cfg['total_params_billions']:.0f}B  "
              f"activated={cfg['activated_params_billions']:.0f}B  "
              f"attn={attn}")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate LLM inference max concurrency from GPU memory"
    )
    parser.add_argument(
        "--model", type=str, help="Model key to estimate (optional if --total-params is given; provides KV architecture defaults)"
    )
    parser.add_argument(
        "--total-params", type=float, default=None,
        help="Override total model parameters in billions (overrides model config default)"
    )
    parser.add_argument(
        "--gpu-memory", type=float, default=80,
        help="GPU memory per GPU in GB (default: 80)"
    )
    parser.add_argument(
        "--tp-size", type=int, default=8,
        help="Tensor parallelism size (default: 8)"
    )
    parser.add_argument(
        "--context-length", type=int, default=16384,
        help="Context length (default: 16384)"
    )
    parser.add_argument(
        "--weight-dtype", type=str, default="fp8",
        help="Weight data type (default: fp8)"
    )
    parser.add_argument(
        "--kv-cache-dtype", type=str, default="fp8",
        help="KV cache data type (default: fp8)"
    )
    parser.add_argument(
        "--mem-fraction-static", type=float, default=0.85,
        help="Static memory fraction (default: 0.85)"
    )
    parser.add_argument(
        "--overhead-rate", type=float, default=0.90,
        help="Multiplier on available KV memory after fixed deduction "
             "(default: 0.90, meaning 90%% usable for KV cache)"
    )
    parser.add_argument(
        "--overhead-fixed-gb", type=float, default=0.0,
        help="Fixed GB deduction per GPU before applying overhead-rate "
             "(default: 0)"
    )
    parser.add_argument(
        "--kv-cache-bytes-override", type=float, default=None,
        help="Override total KV cache pool size in bytes"
    )
    parser.add_argument(
        "--bytes-per-token-override", type=float, default=None,
        help="Override bytes per token for KV cache"
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available models and exit"
    )

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    if not args.model:
        parser.error("--model is required unless --list-models is used")

    result = estimate_concurrency(
        model_key=args.model,
        gpu_memory_gb=args.gpu_memory,
        tp_size=args.tp_size,
        context_length=args.context_length,
        weight_dtype=args.weight_dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        mem_fraction_static=args.mem_fraction_static,
        kv_cache_bytes_override=args.kv_cache_bytes_override,
        bytes_per_token_override=args.bytes_per_token_override,
        overhead_rate=args.overhead_rate,
        overhead_fixed_gb=args.overhead_fixed_gb,
        total_params_billions_override=args.total_params,
    )
    print_result(result)


if __name__ == "__main__":
    main()
