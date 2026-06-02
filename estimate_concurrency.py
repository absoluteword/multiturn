#!/usr/bin/env python3
"""Estimate LLM inference max concurrency from GPU memory constraints."""

import argparse
import sys

MODEL_CONFIGS = {
    "deepseekv4_pro": {
        "full_name": "DeepSeek-V4-Pro",
        "mla": False,
        "attention_mechanism": "CSA + HCA (V4 混合注意力, sequence-length-dependent KV)",
        "num_layers": 61,
        "num_kv_heads": 1,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "compress_rate_csa": 4,
        "compress_rate_hca": 128,
        "sliding_window": 128,
        "num_hash_layers": 3,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-06,
        "o_groups": 16,
        "o_lora_rank": 1024,
        "q_lora_rank": 1536,
        "index_head_dim": 128,
        "index_n_heads": 64,
        "index_topk": 1024,
        "swiglu_limit": 10.0,
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 2.5,
        "expert_dtype": "fp4",
        "hidden_dim": 7168,
        "intermediate_dim": 3072,
        "vocab_size": 129280,
        "moe": True,
        "num_experts": 384,
        "num_routed_experts": 384,
        "num_shared_experts": 1,
        "num_experts_per_token": 6,
        "activated_params_billions": 49.0,
        "total_params_billions": 1600.0,
        "context_length": 1048576,
        "v4_slope_bytes": 4444,
        "v4_fixed_kb": 21040,
        "v4_estimate_seq_len": 65536
    },
    "deepseekv4_flash": {
        "full_name": "DeepSeek-V4-Flash",
        "mla": False,
        "attention_mechanism": "CSA + HCA (V4 混合注意力, sequence-length-dependent KV)",
        "num_layers": 43,
        "num_kv_heads": 1,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "compress_rate_csa": 4,
        "compress_rate_hca": 128,
        "sliding_window": 128,
        "num_hash_layers": 3,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-06,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "q_lora_rank": 1024,
        "index_head_dim": 128,
        "index_n_heads": 64,
        "index_topk": 512,
        "swiglu_limit": 10.0,
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 1.5,
        "expert_dtype": "fp4",
        "hidden_dim": 4096,
        "intermediate_dim": 2048,
        "vocab_size": 129280,
        "moe": True,
        "num_experts": 256,
        "num_routed_experts": 256,
        "num_shared_experts": 1,
        "num_experts_per_token": 6,
        "activated_params_billions": 13.0,
        "total_params_billions": 284.0,
        "context_length": 1048576,
        "v4_slope_bytes": 3104,
        "v4_fixed_kb": 13896,
        "v4_estimate_seq_len": 65536
    },
    "step3.7_flash": {
        "full_name": "Step-3.7-Flash",
        "mla": False,
        "attention_mechanism": "GQA (full + sliding window, 1:3 ratio)",
        "num_layers": 45,
        "num_full_attention_layers": 12,
        "num_sliding_attention_layers": 33,
        "sliding_window": 512,
        "num_kv_heads": 8,
        "head_dim": 128,
        "hidden_dim": 4096,
        "intermediate_dim": 11264,
        "moe_intermediate_size": 1280,
        "share_expert_dim": 1280,
        "vocab_size": 128896,
        "moe": True,
        "num_experts": 288,
        "num_routed_experts": 288,
        "num_shared_experts": 1,
        "num_experts_per_token": 8,
        "activated_params_billions": 11.0,
        "total_params_billions": 196.0,
        "context_length": 262144
    },
    "qwen3.5_35b_a3b": {
        "full_name": "Qwen3.5-35B-A3B",
        "mla": False,
        "attention_mechanism": "Hybrid (Gated DeltaNet + Gated Attention, 3:1 ratio)",
        "num_layers": 40,
        "num_gqa_layers": 10,
        "num_kv_heads": 2,
        "head_dim": 256,
        "linear_conv_kernel_dim": 4,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "hidden_dim": 2048,
        "intermediate_dim": 512,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "vocab_size": 248320,
        "moe": True,
        "num_experts": 256,
        "num_routed_experts": 256,
        "num_shared_experts": 1,
        "num_experts_per_token": 8,
        "activated_params_billions": 3.0,
        "total_params_billions": 35.0,
        "context_length": 262144
    },
    "qwen3.5_397b_a17b": {
        "full_name": "Qwen3.5-397B-A17B",
        "mla": False,
        "attention_mechanism": "Hybrid (Gated DeltaNet + Gated Attention, 3:1 ratio)",
        "num_layers": 60,
        "num_gqa_layers": 15,
        "num_kv_heads": 2,
        "head_dim": 256,
        "linear_conv_kernel_dim": 4,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_num_value_heads": 64,
        "linear_value_head_dim": 128,
        "hidden_dim": 4096,
        "intermediate_dim": 1024,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "vocab_size": 248320,
        "moe": True,
        "num_experts": 512,
        "num_routed_experts": 512,
        "num_shared_experts": 0,
        "num_experts_per_token": 10,
        "activated_params_billions": 17.0,
        "total_params_billions": 397.0,
        "context_length": 262144
    },
    "qwen3.5_122b_a10b": {
        "full_name": "Qwen3.5-122B-A10B",
        "mla": False,
        "attention_mechanism": "Hybrid (Gated DeltaNet + Gated Attention, 3:1 ratio)",
        "num_layers": 48,
        "num_gqa_layers": 12,
        "num_kv_heads": 2,
        "head_dim": 256,
        "linear_conv_kernel_dim": 4,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_num_value_heads": 64,
        "linear_value_head_dim": 128,
        "hidden_dim": 3072,
        "intermediate_dim": 1024,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "vocab_size": 248320,
        "moe": True,
        "num_experts": 256,
        "num_routed_experts": 256,
        "num_shared_experts": 0,
        "num_experts_per_token": 8,
        "activated_params_billions": 10.0,
        "total_params_billions": 122.0,
        "context_length": 262144
    },
    "minimax_m2.5": {
        "full_name": "MiniMax-M2.5",
        "mla": False,
        "attention_mechanism": "GQA (Grouped Query Attention)",
        "num_layers": 62,
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
        "total_params_billions": 230.0,
        "context_length": 128000
    },
    "minimax_m2": {
        "full_name": "MiniMax-M2",
        "mla": False,
        "attention_mechanism": "GQA (Grouped Query Attention)",
        "num_layers": 62,
        "num_kv_heads": 8,
        "head_dim": 128,
        "num_attention_heads": 48,
        "hidden_dim": 3072,
        "intermediate_dim": 8192,
        "vocab_size": 200064,
        "moe": True,
        "num_experts": 256,
        "num_routed_experts": 256,
        "num_shared_experts": 0,
        "num_experts_per_token": 8,
        "activated_params_billions": 10.0,
        "total_params_billions": 230.0,
        "context_length": 196608,
        "use_mtp": True,
        "num_mtp_modules": 3
    },
    "kimi_k2.6": {
        "full_name": "Kimi-K2.6",
        "mla": True,
        "attention_mechanism": "MLA (Multi-Latent Attention, DeepSeek-V3 style)",
        "num_layers": 61,
        "num_kv_heads": 64,
        "head_dim": 128,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "hidden_dim": 5120,
        "intermediate_dim": 13824,
        "vocab_size": 160000,
        "moe": True,
        "num_experts": 385,
        "num_routed_experts": 384,
        "num_shared_experts": 1,
        "num_experts_per_token": 9,
        "activated_params_billions": 32.0,
        "total_params_billions": 1000.0,
        "context_length": 262144
    },
    "glm5.1": {
        "full_name": "GLM-5.1",
        "mla": True,
        "attention_mechanism": "MLA + DSA (DeepSeek Sparse Attention with indexer)",
        "num_layers": 78,
        "num_kv_heads": 64,
        "head_dim": 64,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "index_topk": 2048,
        "hidden_dim": 6144,
        "intermediate_dim": 18432,
        "moe_intermediate_size": 2048,
        "vocab_size": 154880,
        "moe": True,
        "num_experts": 257,
        "num_routed_experts": 256,
        "num_shared_experts": 1,
        "num_experts_per_token": 9,
        "activated_params_billions": 40.0,
        "total_params_billions": 744.0,
        "context_length": 202752
    },
    "kimi_k2.5": {
        "full_name": "Kimi-K2.5",
        "mla": True,
        "attention_mechanism": "MLA (Multi-Latent Attention, DeepSeek-V3 style)",
        "num_layers": 61,
        "num_kv_heads": 64,
        "head_dim": 128,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "hidden_dim": 7168,
        "intermediate_dim": 16384,
        "vocab_size": 160000,
        "moe": True,
        "num_experts": 385,
        "num_routed_experts": 384,
        "num_shared_experts": 1,
        "num_experts_per_token": 8,
        "activated_params_billions": 32.0,
        "total_params_billions": 1000.0,
        "context_length": 262144
    },
    "glm5": {
        "full_name": "GLM-5",
        "mla": True,
        "attention_mechanism": "MLA + DSA (DeepSeek Sparse Attention with indexer)",
        "num_layers": 78,
        "num_kv_heads": 64,
        "head_dim": 64,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "index_topk": 2048,
        "hidden_dim": 6144,
        "intermediate_dim": 12288,
        "moe_intermediate_size": 2048,
        "vocab_size": 154880,
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
    num_layers = config.get("num_layers", 0)
    n_gqa_layers = config.get("num_gqa_layers", num_layers)
    is_hybrid = n_gqa_layers < num_layers

    if "v4_slope_bytes" in config and "v4_fixed_kb" in config:
        slope = config["v4_slope_bytes"]
        fixed_bytes = config["v4_fixed_kb"] * 1024
        seq_len = config.get("v4_estimate_seq_len", 65536)
        total_seq_bytes = slope * seq_len + fixed_bytes
        return total_seq_bytes / seq_len

    if config.get("mla") or "MLA" in mechanism:
        if "kv_lora_rank" in config:
            kv_lora_rank = config["kv_lora_rank"]
            qk_rope_head_dim = config.get(
                "qk_rope_head_dim", config.get("rope_head_dim", 0)
            )
            index_head_dim = config.get("index_head_dim", 0)
            per_layer = kv_lora_rank + qk_rope_head_dim + index_head_dim
            return config["num_layers"] * per_layer

        kv_per_layer = config.get("latent_kv_dim", 0)
        rope_head_dim = config.get("rope_head_dim", config.get("head_dim", 0))
        return config["num_layers"] * (kv_per_layer + rope_head_dim)

    if "Gated DeltaNet" in mechanism or "DeltaNet" in mechanism:
        if is_hybrid:
            kv_per_layer = config["num_kv_heads"] * config["head_dim"]
            return 2 * n_gqa_layers * kv_per_layer
        return 0

    kv_per_layer = config["num_kv_heads"] * config["head_dim"]
    return 2 * n_gqa_layers * kv_per_layer


def get_linear_attention_state_size(config, kv_bytes):
    mechanism = config.get("attention_mechanism", "")
    if "Gated DeltaNet" not in mechanism and "DeltaNet" not in mechanism:
        return 0
    if "num_gqa_layers" in config and config["num_gqa_layers"] < config.get("num_layers", 0):
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
        effective_kv_tp = 1
    else:
        num_kv_heads_cfg = config.get("num_kv_heads", 1) or 1
        effective_kv_tp = min(tp_size, num_kv_heads_cfg)
    kv_sharding_efficiency = effective_kv_tp / tp_size if tp_size > 0 else 0.0

    kv_cache_budget_bytes = per_gpu_effective_kv * (1024 ** 3) * effective_kv_tp

    if kv_cache_bytes_override is not None:
        kv_cache_budget_bytes = kv_cache_bytes_override

    single_token_bytes = kv_per_token_bytes
    if bytes_per_token_override is not None:
        single_token_bytes = bytes_per_token_override

    if "v4_slope_bytes" in config and "v4_fixed_kb" in config:
        single_token_bytes = config["v4_slope_bytes"]
        kv_fixed_bytes = config["v4_fixed_kb"] * 1024
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
        "attention_mechanism": config.get("attention_mechanism", "GQA"),
        "kv_lora_rank": config.get("kv_lora_rank"),
        "qk_nope_head_dim": config.get("qk_nope_head_dim"),
        "qk_rope_head_dim": config.get("qk_rope_head_dim"),
        "v_head_dim": config.get("v_head_dim"),
        "index_head_dim": config.get("index_head_dim"),
        "index_n_heads": config.get("index_n_heads"),
        "index_topk": config.get("index_topk"),
        "num_kv_heads": config.get("num_kv_heads"),
        "head_dim": config.get("head_dim"),
        "num_layers": config.get("num_layers"),
        "num_gqa_layers": config.get("num_gqa_layers", config.get("num_layers")),
        "kv_per_layer_dim": kv_per_token // config.get("num_layers", 1) if config.get("mla") and "kv_lora_rank" in config else None,
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
        "effective_kv_tp": effective_kv_tp,
        "kv_sharding_efficiency": kv_sharding_efficiency,
        "kv_cache_budget_gb": kv_cache_budget_bytes / (1024 ** 3),
        "per_seq_per_gpu_bytes_gb": (kv_per_sequence_bytes / effective_kv_tp) / (1024 ** 3) if effective_kv_tp > 0 else 0.0,
        "max_concurrency": max_concurrency,
    }


def print_result(r):
    print(f"Model: {r['full_name']} ({r['model']})")
    print(f"MLA: {r['mla']}")
    if r['mla']:
        if r.get('kv_lora_rank') is not None:
            print(f"MLA cache structure (per layer per token):")
            print(f"  kv_lora_rank (shared compressed latent): {r['kv_lora_rank']}")
            print(f"  qk_nope_head_dim (absorbed into latent):   {r.get('qk_nope_head_dim', '-')}")
            print(f"  qk_rope_head_dim (per-head RoPE, stored):  {r.get('qk_rope_head_dim', '-')}")
            print(f"  v_head_dim (absorbed into latent):         {r.get('v_head_dim', '-')}")
            if r.get('index_head_dim') is not None and r['index_head_dim'] > 0:
                print(f"  index_head_dim (DSA index K, stored):     {r['index_head_dim']}")
                print(f"  index_n_heads:                            {r.get('index_n_heads', '-')}")
                print(f"  index_topk:                               {r.get('index_topk', '-')}")
            print(f"  num_kv_heads (RoPE-K replicated this many): {r.get('num_kv_heads', '-')}")
            print(f"  num_layers: {r.get('num_layers', '-')}")
            if r.get('kv_per_layer_dim') is not None:
                if r.get('index_head_dim') is not None and r['index_head_dim'] > 0:
                    print(f"  kv_per_layer_dim = {r['kv_lora_rank']} + "
                          f"{r.get('qk_rope_head_dim', 0)} + "
                          f"{r['index_head_dim']} (DSA index K) "
                          f"= {r['kv_per_layer_dim']}")
                else:
                    print(f"  kv_per_layer_dim = {r['kv_lora_rank']} + "
                          f"{r.get('qk_rope_head_dim', 0)} "
                          f"= {r['kv_per_layer_dim']}")
        else:
            print(f"MLA rope_head_dim: {r.get('qk_rope_head_dim')}")
    print(f"Attention mechanism: {r['attention_mechanism']}")
    if r.get('num_gqa_layers') and r.get('num_layers') and r['num_gqa_layers'] < r['num_layers']:
        print(f"GQA layers: {r['num_gqa_layers']} of {r['num_layers']} "
              f"(linear attention in remaining {r['num_layers'] - r['num_gqa_layers']} layers; "
              f"linear state not counted per user instruction)")
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
    kv_per_token_dim = r['kv_per_token_dim']
    kv_per_token_bytes = r['kv_per_token_bytes']
    if isinstance(kv_per_token_dim, float):
        print(f"KV per token (bytes, at v4_estimate_seq_len={r.get('v4_estimate_seq_len', 65536)}): {kv_per_token_dim:,.1f}")
        print(f"KV per token (bytes @ {r['kv_cache_dtype']}): {kv_per_token_bytes:,.1f}")
    else:
        print(f"KV per token (dimension): {kv_per_token_dim:,d}")
        print(f"KV per token (bytes @ {r['kv_cache_dtype']}): {kv_per_token_bytes:,d}")
    print(f"KV per sequence (full, single-GPU equivalent): {r['kv_per_sequence_bytes_gb']:.4f} GB")
    eff = r['effective_kv_tp']
    shd = r['kv_sharding_efficiency']
    if r['mla']:
        sharding_note = "MLA: full cache replicated on every GPU"
    elif shd >= 0.999:
        sharding_note = f"GQA: KV fully sharded across TP={r['tp_size']} (num_kv_heads={r['num_kv_heads']} is sufficient)"
    elif eff == 1 and (r['num_kv_heads'] or 1) == 1:
        sharding_note = f"MQA: num_kv_heads=1, no head-level sharding possible (TP={r['tp_size']} replicates)"
    else:
        sharding_note = (f"GQA: KV sharded across effective_kv_tp={eff} GPUs (limited by "
                         f"num_kv_heads={r['num_kv_heads']} < TP={r['tp_size']}); "
                         f"sharding efficiency = {shd*100:.1f}%")
    print(f"KV sharding: {sharding_note}")
    print(f"KV cache budget (cluster, after sharding limit): {r['kv_cache_budget_gb']:.1f} GB")
    print(f"KV per sequence (per-GPU with sharding): {r['per_seq_per_gpu_bytes_gb']:.4f} GB")
    print("---")
    if r['max_concurrency'] < 0:
        print("Estimated max concurrency: N/A (zero per-token KV, bounded by other memory)")
    else:
        print(f"Estimated max concurrency: {r['max_concurrency']}")
    print()


def list_models():
    print("Available models (KV cache per token at FP8):")
    print()
    header = (
        f"  {'model':<18s}  {'full_name':<18s}  {'attn':<46s}  "
        f"{'L':>3s}  {'H':>4s}  {'KV/token':>12s}  {'KB/token':>10s}  "
        f"{'params':>10s}  {'act':>6s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key, cfg in MODEL_CONFIGS.items():
        attn = cfg.get("attention_mechanism", "GQA")
        kv_dim = get_kv_per_token(cfg)
        kv_bytes_fp8 = kv_dim
        kv_kb_fp8 = kv_bytes_fp8 / 1024.0
        mla = cfg.get("mla", False)
        h = cfg.get("num_kv_heads", "-")
        attn_short = attn if len(attn) <= 46 else attn[:43] + "..."
        if isinstance(kv_dim, float):
            kv_dim_str = f"{kv_dim:,.1f}"
        else:
            kv_dim_str = f"{kv_dim:,d}" if kv_dim > 0 else "0"
        print(
            f"  {key:<18s}  {cfg['full_name']:<18s}  {attn_short:<46s}  "
            f"{cfg['num_layers']:>3d}  {h:>4}  {kv_dim_str:>12s}  "
            f"{kv_kb_fp8:>8.1f}KB  "
            f"{cfg['total_params_billions']:>8.0f}B  "
            f"{cfg['activated_params_billions']:>4.0f}B"
        )
    print()
    print("L = num_layers, H = num_kv_heads (informational for MLA; cache is shared across heads)")
    print("KV/token = dimension count; multiply by bytes/element for actual size")
    print("  FP8/int8: 1 byte/elem | FP16/BF16: 2 bytes/elem | FP4/INT4/NVFP4: 0.5 bytes/elem")
    if any("v4_slope_bytes" in cfg for cfg in MODEL_CONFIGS.values()):
        print()
        print("V4 (CSA+HCA) models: KV is sequence-length-dependent; per-token values shown at 64K context")
        print("  formula: per_sequence = v4_slope_bytes * seq_len + v4_fixed_kb * 1024")
    print()
    if any("kv_lora_rank" in cfg for cfg in MODEL_CONFIGS.values()):
        print("MLA models: KV cache per layer per token = kv_lora_rank + qk_rope_head_dim")
        print("  (compressed latent is shared across all heads; RoPE-K is a single vector per token, not replicated)")
        if any(cfg.get("index_head_dim", 0) > 0 for cfg in MODEL_CONFIGS.values()):
            print("MLA + DSA (sparse indexer) also stores: + index_head_dim for index K")
            print("  (indexer selects top-k tokens for sparse attention; index K is a single vector per token)")


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
