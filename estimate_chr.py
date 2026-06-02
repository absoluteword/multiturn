import argparse

def estimate_cache_hit_rates(first_input_len, subsequent_input_len, output_len, num_rounds):
    result = []

    for round_idx in range(num_rounds):
        if round_idx == 0:
            accumulated_before = 0
            total_prompt = first_input_len
        else:
            accumulated_before = first_input_len + (round_idx - 1) * output_len + (round_idx - 1) * subsequent_input_len
            total_prompt = accumulated_before + output_len + subsequent_input_len

        if total_prompt > 0:
            hit_rate = accumulated_before / total_prompt
        else:
            hit_rate = 0.0

        if round_idx == 0:
            new_tokens = first_input_len
        else:
            new_tokens = subsequent_input_len

        result.append({
            "round": round_idx,
            "total_prompt_len": total_prompt,
            "cached_tokens": accumulated_before,
            "new_tokens": new_tokens,
            "cache_hit_rate": hit_rate,
        })

    return result


def print_results(results):
    print(f"{'Round':>6}  {'Total Prompt':>14}  {'Cached':>10}  {'New':>8}  {'Cache Hit Rate':>16}")
    print("-" * 64)
    for r in results:
        print(
            f"{r['round']:>6}  "
            f"{r['total_prompt_len']:>14}  "
            f"{r['cached_tokens']:>10}  "
            f"{r['new_tokens']:>8}  "
            f"{r['cache_hit_rate']:>15.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Estimate per-round KV cache hit rate for multi-turn conversations")
    parser.add_argument("--first-input-len", type=int, required=True, help="First round input length (tokens)")
    parser.add_argument("--subsequent-input-len", type=int, required=True, help="Subsequent round input length (tokens)")
    parser.add_argument("--output-len", type=int, required=True, help="Output length per round (tokens)")
    parser.add_argument("--num-rounds", type=int, required=True, help="Number of rounds")
    args = parser.parse_args()

    results = estimate_cache_hit_rates(
        first_input_len=args.first_input_len,
        subsequent_input_len=args.subsequent_input_len,
        output_len=args.output_len,
        num_rounds=args.num_rounds,
    )

    print_results(results)


if __name__ == "__main__":
    main()
