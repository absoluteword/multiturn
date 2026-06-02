import argparse
import asyncio
import json
import os
import queue
import random
import sys
import threading
import time
from datetime import datetime

import numpy as np
import requests
from tqdm.asyncio import tqdm

from sglang.bench_serving import RequestFuncOutput
from sglang.benchmark.datasets.random import sample_random_requests
from sglang.benchmark.utils import get_tokenizer
from sglang.test.kits.cache_hit_kit import (
    async_request_sglang_generate,
    gen_payload,
    gen_payload_openai,
)

# Optional import: aiohttp for robust vLLM streaming.
# If missing, fallback to sglang's helper (may give ttft=0 on vLLM).
try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    aiohttp = None  # type: ignore
    _HAS_AIOHTTP = False


def _skip_empty_sse(chunk: dict) -> bool:
    """Return True if this SSE chunk carries no actual token content."""
    choices = chunk.get("choices", [])
    if not choices:
        return True
    delta = choices[0].get("delta", {})
    content = delta.get("content", "")
    tool_calls = delta.get("tool_calls")
    reasoning = delta.get("reasoning")
    # Accept chunk only if it has non-empty content / reasoning / tool_calls
    if content:
        return bool(content)
    if reasoning:
        return bool(reasoning)
    if tool_calls:
        return True
    return False


async def async_request_vllm_chat_completions(payload, url, pbar):
    """Custom async request for vLLM OpenAI-compatible streaming endpoint.

    Measures TTFT as the time until the *first* SSE chunk with actual
    token content (skipping the role-only delta that vLLM may send
    before prefill completes, which causes sglang helper to report ttft≈0).
    """
    request_start = time.perf_counter()
    first_token_time = None
    generated_text = ""
    prompt_len = 0
    completion_len = 0
    cached_tokens = 0
    output_ids = []

    if aiohttp is None:
        raise RuntimeError(
            "aiohttp is required for --api-format openai with vLLM. "
            "Install it with: pip install aiohttp"
        )

    try:
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Parse usage FIRST — vLLM puts this in a final empty chunk
                    # (choices: []) that would otherwise be skipped.
                    usage = chunk.get("usage")
                    if usage:
                        prompt_len = usage.get("prompt_tokens", prompt_len)
                        completion_len = usage.get("completion_tokens", completion_len)
                        pdetails = usage.get("prompt_tokens_details")
                        if pdetails:
                            cached_tokens = pdetails.get("cached_tokens", cached_tokens)

                    # Skip chunks that carry no actual token content for TTFT purposes
                    if _skip_empty_sse(chunk):
                        continue

                    # First real token chunk -> record TTFT
                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            generated_text += content

    except Exception as e:
        output = RequestFuncOutput()
        output.success = False
        output.error = str(e)
        if pbar is not None:
            pbar.update(1)
        return output

    request_end = time.perf_counter()

    if first_token_time is None:
        # Edge case: stream ended without any content chunk
        first_token_time = request_end

    output = RequestFuncOutput()
    output.success = True
    output.ttft = first_token_time - request_start
    output.latency = request_end - request_start
    output.prompt_len = prompt_len
    output.generated_len = completion_len
    output.generated_text = generated_text
    output.output_ids = output_ids
    output.itl = []  # ITL estimated from latency-ttft in bench_multiturn.py
    output.cached_tokens = cached_tokens

    if pbar is not None:
        pbar.update(1)
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to benchmark concurrent requests to a server."
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=32,
        help="Number of concurrent clients",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=16,
        help="Maximum number of parallel requests",
    )
    parser.add_argument(
        "--request-length",
        type=int,
        default=1700,
        help="Length of each new request",
    )
    parser.add_argument(
        "--first-round-input-length",
        type=int,
        default=16000,
        help="Input length for the first round. If 0, use --request-length.",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=500,
        help="Length of each output",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=32,
        help="Number of rounds per client",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="poisson",
        choices=["poisson", "uniform"],
        help="Distribution type for request intervals (poisson or uniform)",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=10.0,
        help="Average number of requests per second",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server hostname or IP (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/models/GLM-5-NVFP4/",
        help="model path compatible with Hugging Face Transformers",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/data/models/ShareGPT_V3_unfiltenred_cleaned_split.json",
        help="local dataset to sample tokens from",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="performance_metrics.jsonl",
        help="File to log performance metrics",
    )
    parser.add_argument(
        "--disable-auto-run",
        action="store_true",
        default=True,
        help="If set, disable automatically testing with a range of request rates.",
    )
    parser.add_argument(
        "--disable-random-sample",
        action="store_true",
        default=True,
        help="If set, disable random sampling of requests from the ShareGPT dataset.",
    )
    parser.add_argument(
        "--enable-round-barrier",
        action="store_true",
        default=True,
        help="If set, only send i-th turn requests after all (i-1)-th turn requests finished.",
    )
    parser.add_argument(
        "--sub-question-input-length",
        type=int,
        default=0,
        help="Length of the sub question input for each request, if set 0 use request_length",
    )
    parser.add_argument(
        "--ready-queue-policy",
        type=str,
        default="random",
        help="Policy for popping requests from the ready queue (random or fifo)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Tag of a certain run in the log file",
    )
    parser.add_argument(
        "--min-rounds",
        type=int,
        default=0,
        help="Min rounds per client (0 = use --num-rounds)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="Max rounds per client (0 = use --num-rounds)",
    )
    parser.add_argument(
        "--range-ratio",
        type=float,
        default=0.8,
        help="Length variation ratio for prompts and outputs (1.0 = no variation, 0.5 = 50%% variation)",
    )
    parser.add_argument("--seed", type=int, default=42, help="The random seed.")
    parser.add_argument(
        "--lora-path",
        type=str,
        default="",
        help="String of LoRA path. Currently we only support benchmarking on a single LoRA adaptor.",
    )
    parser.add_argument(
        "--api-format",
        type=str,
        default="sglang",
        choices=["sglang", "openai"],
        help="API format to use: 'sglang' for native /generate endpoint, "
        "'openai' for OpenAI-compatible /v1/chat/completions endpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for txt result files",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default="",
        help="Served model name expected by the API server (e.g. vLLM --served-model-name). "
        "If empty, falls back to the basename of --model-path.",
    )
    return parser.parse_args()


def log_to_jsonl_file(data, file_path="performance_metrics.jsonl", tag=""):
    """Append the data with a timestamp and tag to the specified JSONL file."""
    timestamped_data = {"timestamp": datetime.now().isoformat(), "tag": tag, **data}
    try:
        with open(file_path, "a") as file:
            file.write(
                json.dumps(timestamped_data) + "\n"
            )  # Write as a single line in JSONL format
    except IOError as e:
        print(f"Error writing to JSONL file: {e}")


class ReadyQueue:
    """
    Thread-safe queue that can pop requests in different orders based on given policy.
    """

    def __init__(self, init_requests=None, policy="random"):
        self.lock = threading.Lock()
        self.requests = init_requests or []
        self.policy = policy

    def append(self, item):
        with self.lock:
            self.requests.append(item)

    def pop(self):
        with self.lock:
            if not self.requests:
                return None
            if self.policy == "random":
                index = random.randrange(len(self.requests))
                return self.requests.pop(index)
            elif self.policy == "fifo":
                return self.requests.pop(0)
            else:
                # todo, varying thinking time of clients
                raise ValueError(f"{self.policy} not implemented")


class WorkloadGenerator:
    def __init__(self, args):
        self.api_format = args.api_format
        self.model_path = args.model_path

        # vLLM/OpenAI API expects the served model name, not the raw path.
        # Use --served-model-name if provided, otherwise basename of --model-path.
        self.model_name = (
            args.served_model_name
            if args.served_model_name
            else os.path.basename(os.path.normpath(args.model_path))
        )

        # Construct the base URL and select request/payload functions
        if self.api_format == "openai":
            self.url = f"http://{args.host}:{args.port}/v1/chat/completions"
            # Use our own vLLM-compatible streaming helper to avoid
            # sglang's ttft=0 bug caused by role-only SSE chunks.
            self.request_func = async_request_vllm_chat_completions
        else:
            self.url = f"http://{args.host}:{args.port}/generate"
            self.request_func = async_request_sglang_generate

        self.tokenizer = get_tokenizer(args.model_path)
        self.distribution = args.distribution
        self.request_rate = args.request_rate
        self.start_time = None
        self.finished_time = None
        self.lora_path = args.lora_path

        self.sent_requests = 0
        self.completed_requests = 0

        # Resolve per-client round counts
        min_rounds = args.min_rounds
        max_rounds = args.max_rounds
        if min_rounds == 0 and max_rounds == 0:
            # Backward compat: all clients use --num-rounds
            min_rounds = args.num_rounds
            max_rounds = args.num_rounds
        elif min_rounds == 0:
            min_rounds = max_rounds
        elif max_rounds == 0:
            max_rounds = min_rounds
        if min_rounds < 1:
            raise ValueError(f"--min-rounds must be >= 1, got {min_rounds}")
        if min_rounds > max_rounds:
            raise ValueError(
                f"--min-rounds ({min_rounds}) must be <= --max-rounds ({max_rounds})"
            )

        self.min_rounds = min_rounds
        self.max_rounds = max_rounds

        if min_rounds == max_rounds:
            # All clients have the same round count; skip randint to preserve random state
            self.client_total_rounds = [min_rounds] * args.num_clients
        else:
            self.client_total_rounds = [
                random.randint(min_rounds, max_rounds) for _ in range(args.num_clients)
            ]

        # clients_per_round[r] = number of clients participating in round r
        self.clients_per_round = [
            sum(1 for t in self.client_total_rounds if t > r) for r in range(max_rounds)
        ]
        self.total_requests = sum(self.client_total_rounds)

        range_ratio = args.range_ratio

        first_round_input_length = args.first_round_input_length
        if first_round_input_length == 0:
            first_round_input_length = args.request_length

        # Use return_text=False to get token ids instead of text
        first_round_samples = sample_random_requests(
            input_len=first_round_input_length,
            output_len=args.output_length,
            num_prompts=args.num_clients,
            range_ratio=range_ratio,
            tokenizer=self.tokenizer,
            dataset_path=args.dataset_path,
            random_sample=not args.disable_random_sample,
            return_text=False,
        )
        # Store per-sample output_len for first round
        first_round_output_lens = [row.output_len for row in first_round_samples]
        # r.prompt is now List[int] when return_text=False
        self.candidate_inputs = [list(i.prompt) for i in first_round_samples]

        if args.sub_question_input_length != 0:
            sub_question_input_length = args.sub_question_input_length
        else:
            sub_question_input_length = args.request_length

        num_sub_questions = sum(max(t - 1, 0) for t in self.client_total_rounds)

        self.sub_question_inputs = sample_random_requests(
            input_len=sub_question_input_length,
            output_len=args.output_length,
            num_prompts=max(num_sub_questions, 1),
            range_ratio=range_ratio,
            tokenizer=self.tokenizer,
            dataset_path=args.dataset_path,
            random_sample=not args.disable_random_sample,
            return_text=False,
        )

        if self.api_format == "openai":
            # OpenAI mode: history is a messages list for /v1/chat/completions
            initial_messages = {
                i: [
                    {
                        "role": "user",
                        "content": self.tokenizer.decode(self.candidate_inputs[i]),
                    }
                ]
                for i in range(args.num_clients)
            }
            init_requests = [
                (
                    i,
                    gen_payload_openai(
                        initial_messages[i],
                        first_round_output_lens[i],
                        self.model_name,
                    ),
                )
                for i in range(args.num_clients)
            ]
            self.client_records = {
                i: {
                    "round": 0,
                    "history": initial_messages[i],
                    "total_rounds": self.client_total_rounds[i],
                    "pending_sub_q_len": 0,
                }
                for i in range(args.num_clients)
            }
        else:
            # SGLang mode: history is List[int] (token ids)
            init_requests = [
                (
                    i,
                    gen_payload(
                        self.candidate_inputs[i],
                        first_round_output_lens[i],
                        args.lora_path,
                    ),
                )
                for i in range(args.num_clients)
            ]
            self.client_records = {
                i: {
                    "round": 0,
                    "history": list(self.candidate_inputs[i]),
                    "total_rounds": self.client_total_rounds[i],
                    "pending_sub_q_len": 0,
                }
                for i in range(args.num_clients)
            }
        self.ready_queue = ReadyQueue(
            init_requests=init_requests, policy=args.ready_queue_policy
        )
        self.candidate_inputs = self.candidate_inputs[args.num_clients :]

        self.response_queue = queue.Queue()
        self.pbar = tqdm(total=self.total_requests)
        self.performance_metrics = {
            "ttft": [],
            "itl": [],
            "latency": [],
            "prompt_len": [],
            "cached_tokens": [],
            "generated_len": [],
            "miss_tokens": [],
            "hit_tokens": [],
            "start_time": [],
            "end_time": [],
        }
        self.enable_round_barrier = args.enable_round_barrier
        if self.enable_round_barrier:
            # Add round-specific metrics while preserving the original structure
            for i in range(self.max_rounds):
                self.performance_metrics[f"round_{i}"] = {
                    "ttft": [],
                    "itl": [],
                    "latency": [],
                    "prompt_len": [],
                    "cached_tokens": [],
                    "generated_len": [],
                    "predicted_chr": [],
                    "miss_tokens": [],
                    "hit_tokens": [],
                    "start_time": [],
                    "end_time": [],
                }
        self.num_clients = args.num_clients

        self.num_rounds = self.max_rounds
        self.max_parallel = args.max_parallel
        self.output_length = args.output_length

    async def handle_request(self, item):
        client_id, payload = item
        send_time = time.perf_counter()
        try:
            response = await self.request_func(payload, self.url, self.pbar)
            recv_time = time.perf_counter()
            if self.pbar.n == self.pbar.total:
                self.finished_time = recv_time
            self.response_queue.put((client_id, response, send_time, recv_time))
        except Exception as e:
            print(f"Request failed for client {client_id}: {e}")
            failed_response = RequestFuncOutput()
            failed_response.success = False
            failed_response.error = str(e)
            self.response_queue.put((client_id, failed_response, send_time, time.perf_counter()))

    def request_sender(self):
        async def request_loop():
            while True:
                if self.sent_requests - self.completed_requests < self.max_parallel:
                    new_request = self.ready_queue.pop()
                    if new_request:
                        asyncio.create_task(self.handle_request(new_request))
                        self.sent_requests += 1
                else:
                    await asyncio.sleep(0.05)
                    continue

                if self.pbar.n == self.pbar.total:
                    break

                # Calculate Poisson-distributed wait time
                if self.distribution == "poisson":
                    sleep_time = random.expovariate(self.request_rate)
                elif self.distribution == "uniform":
                    avg_interval = (
                        1.0 / self.request_rate if self.request_rate > 0 else 1.0
                    )
                    sleep_time = random.uniform(0, 2 * avg_interval)
                else:
                    raise ValueError("Invalid distribution type")
                await asyncio.sleep(sleep_time)  # Wait before sending the next request

        # Create and run the event loop for asynchronous requests
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(request_loop())
        loop.close()

    def response_handler(self):
        next_round_reqs = []
        current_barrier_round = 0
        barrier_round_completed = 0
        while True:
            try:
                client_id, response, send_time, recv_time = self.response_queue.get(
                    timeout=10
                )  # Block until response is available
                if not response.success:
                    print(f"Request failed for client {client_id}: {response.error}")
                    self.completed_requests += 1
                    continue
                # Extend history with response
                if self.api_format == "openai":
                    if response.generated_text:
                        self.client_records[client_id]["history"].append(
                            {"role": "assistant", "content": response.generated_text}
                        )
                else:
                    self.client_records[client_id]["history"].extend(
                        response.output_ids
                    )
                current_round = self.client_records[client_id]["round"]
                self.client_records[client_id]["round"] += 1

                # Compute per-token ITL from latency and generated_len,
                # because the raw response.itl may be degenerate when the
                # SGLang /generate endpoint batches all output_ids in one
                # SSE event (non-incremental streaming).
                gen_len = response.generated_len
                if gen_len > 1 and response.latency > response.ttft:
                    avg_itl = (response.latency - response.ttft) / (gen_len - 1)
                    per_token_itl = [avg_itl] * (gen_len - 1)
                else:
                    per_token_itl = response.itl

                # Compute predicted cache hit rate BEFORE appending metrics
                # Round 0: no history, predicted_chr = 0
                # Round 1+: only the new sub-question portion misses cache
                prompt_len = response.prompt_len
                sub_q_len = self.client_records[client_id].get("pending_sub_q_len", 0)
                if sub_q_len > 0 and prompt_len > 0:
                    predicted_chr = (prompt_len - sub_q_len) / prompt_len
                else:
                    predicted_chr = 0.0

                # For OpenAI/vLLM, only trust the server's cached_tokens if
                # actually reported (>0). If 0 or missing, assume no cache hit.
                if self.api_format == "openai":
                    estimated_cached_tokens = response.cached_tokens if response.cached_tokens > 0 else 0
                else:
                    estimated_cached_tokens = response.cached_tokens

                miss_tokens = max(0, response.prompt_len - estimated_cached_tokens)
                hit_tokens = estimated_cached_tokens

                self.performance_metrics["ttft"].append(response.ttft)
                self.performance_metrics["itl"].extend(per_token_itl)
                self.performance_metrics["latency"].append(response.latency)
                self.performance_metrics["prompt_len"].append(response.prompt_len)
                self.performance_metrics["cached_tokens"].append(
                    estimated_cached_tokens
                )
                self.performance_metrics["generated_len"].append(
                    response.generated_len
                )
                self.performance_metrics["miss_tokens"].append(miss_tokens)
                self.performance_metrics["hit_tokens"].append(hit_tokens)
                self.performance_metrics["start_time"].append(send_time)
                self.performance_metrics["end_time"].append(recv_time)

                if self.enable_round_barrier:
                    self.performance_metrics[f"round_{current_round}"][
                        "ttft"
                    ].append(response.ttft)
                    self.performance_metrics[f"round_{current_round}"][
                        "itl"
                    ].extend(per_token_itl)
                    self.performance_metrics[f"round_{current_round}"][
                        "latency"
                    ].append(response.latency)
                    self.performance_metrics[f"round_{current_round}"][
                        "prompt_len"
                    ].append(response.prompt_len)
                    self.performance_metrics[f"round_{current_round}"][
                        "cached_tokens"
                    ].append(estimated_cached_tokens)
                    self.performance_metrics[f"round_{current_round}"][
                        "generated_len"
                    ].append(response.generated_len)
                    self.performance_metrics[f"round_{current_round}"][
                        "predicted_chr"
                    ].append(predicted_chr)
                    self.performance_metrics[f"round_{current_round}"][
                        "miss_tokens"
                    ].append(miss_tokens)
                    self.performance_metrics[f"round_{current_round}"][
                        "hit_tokens"
                    ].append(hit_tokens)
                    self.performance_metrics[f"round_{current_round}"][
                        "start_time"
                    ].append(send_time)
                    self.performance_metrics[f"round_{current_round}"][
                        "end_time"
                    ].append(recv_time)
                self.completed_requests += 1

                client_total = self.client_records[client_id]["total_rounds"]
                if self.client_records[client_id]["round"] < client_total:
                    sub_q = self.sub_question_inputs.pop()
                    sub_q_len = len(sub_q.prompt)
                    self.client_records[client_id]["pending_sub_q_len"] = sub_q_len
                    if self.api_format == "openai":
                        # Append sub-question as a new user message
                        sub_q_text = self.tokenizer.decode(list(sub_q.prompt))
                        self.client_records[client_id]["history"].append(
                            {"role": "user", "content": sub_q_text}
                        )
                        new_req = (
                            client_id,
                            gen_payload_openai(
                                self.client_records[client_id]["history"],
                                sub_q.output_len,
                                self.model_name,
                            ),
                        )
                    else:
                        # Append sub-question token ids to client's history
                        sub_q_ids = list(sub_q.prompt)
                        self.client_records[client_id]["history"].extend(sub_q_ids)
                        new_req = (
                            client_id,
                            gen_payload(
                                self.client_records[client_id]["history"],
                                sub_q.output_len,
                                self.lora_path,
                            ),
                        )
                    if self.enable_round_barrier:
                        next_round_reqs.append(new_req)
                    else:
                        self.ready_queue.append(new_req)

                # Barrier logic: release next round when all clients for
                # current barrier round have completed
                if (
                    self.enable_round_barrier
                    and current_barrier_round < self.max_rounds
                ):
                    barrier_round_completed += 1
                    expected = self.clients_per_round[current_barrier_round]
                    if barrier_round_completed == expected:
                        print(
                            f"\n  Barrier: round {current_barrier_round} complete "
                            f"({expected} clients), releasing {len(next_round_reqs)} "
                            f"requests for round {current_barrier_round + 1}"
                        )
                        self._send_heartbeat(input_len=100, output_len=100)
                        time.sleep(10)
                        for req in next_round_reqs:
                            self.ready_queue.append(req)
                        next_round_reqs = []
                        current_barrier_round += 1
                        barrier_round_completed = 0
            except queue.Empty:
                if self.pbar.n == self.pbar.total:
                    break
            except ValueError as e:
                print(f"Error processing response for client {client_id}: {e}")
                continue

    def _send_heartbeat(self, input_len=100, output_len=20):
        """Send a small heartbeat request to the server.

        For OpenAI-compatible backends (vLLM), skip the heartbeat
        because /generate and the sglang payload are not supported.
        """
        if self.api_format == "openai":
            return
        heartbeat_input = [1] * input_len
        payload = gen_payload(heartbeat_input, output_len, self.lora_path)
        try:
            requests.post(self.url, json=payload, timeout=30)
        except Exception as e:
            print(f"Heartbeat request failed: {e}")

    def run(self):
        request_thread = threading.Thread(target=self.request_sender, daemon=True)
        response_thread = threading.Thread(target=self.response_handler, daemon=True)

        self.start_time = time.perf_counter()
        request_thread.start()
        response_thread.start()

        request_thread.join()
        response_thread.join()
        self.pbar.close()

        duration = self.finished_time - self.start_time
        sorted_ttft = sorted(self.performance_metrics["ttft"])
        sorted_latency = sorted(self.performance_metrics["latency"])
        sorted_itl = sorted(self.performance_metrics["itl"])
        sorted_prompt_len = sorted(self.performance_metrics["prompt_len"])
        sorted_output_len = sorted(self.performance_metrics["generated_len"])

        def percentile(sorted_vals, q):
            if not sorted_vals:
                return 0.0
            idx = int(q * len(sorted_vals))
            if idx >= len(sorted_vals):
                idx = len(sorted_vals) - 1
            return sorted_vals[idx]

        def max_or_zero(sorted_vals):
            return sorted_vals[-1] if sorted_vals else 0.0

        performance_data = {
            "summary": {
                "total_requests": len(self.performance_metrics["ttft"]),
                "request_rate": self.request_rate,
                "average_prompt_len": (
                    sum(self.performance_metrics["prompt_len"])
                    / len(self.performance_metrics["prompt_len"])
                    if self.performance_metrics["prompt_len"]
                    else 0.0
                ),
                "average_output_len": (
                    sum(self.performance_metrics["generated_len"])
                    / len(self.performance_metrics["generated_len"])
                    if self.performance_metrics["generated_len"]
                    else 0.0
                ),
                "p90_prompt_len": percentile(sorted_prompt_len, 0.9),
                "p99_prompt_len": percentile(sorted_prompt_len, 0.99),
                "p90_output_len": percentile(sorted_output_len, 0.9),
                "p99_output_len": percentile(sorted_output_len, 0.99),
                "average_ttft": sum(self.performance_metrics["ttft"])
                / len(self.performance_metrics["ttft"]),
                "p90_ttft": percentile(sorted_ttft, 0.9),
                "p99_ttft": percentile(sorted_ttft, 0.99),
                "median_ttft": percentile(sorted_ttft, 0.5),
                "max_ttft": max_or_zero(sorted_ttft),
                "average_itl": (
                    sum(self.performance_metrics["itl"])
                    / len(self.performance_metrics["itl"])
                    if self.performance_metrics["itl"]
                    else 0.0
                ),
                "p90_itl": percentile(sorted_itl, 0.9),
                "p99_itl": percentile(sorted_itl, 0.99),
                "median_itl": percentile(sorted_itl, 0.5),
                "max_itl": max_or_zero(sorted_itl),
                "average_latency": sum(self.performance_metrics["latency"])
                / len(self.performance_metrics["latency"]),
                "p90_latency": percentile(sorted_latency, 0.9),
                "p99_latency": percentile(sorted_latency, 0.99),
                "median_latency": percentile(sorted_latency, 0.5),
                "max_latency": max_or_zero(sorted_latency),
                "input_token_throughput": sum(self.performance_metrics["prompt_len"])
                / duration,
                "output_token_throughput": sum(
                    self.performance_metrics["generated_len"]
                )
                / duration,
                "total_token_throughput": (
                    (
                        sum(self.performance_metrics["miss_tokens"])
                        + sum(self.performance_metrics["generated_len"])
                    )
                    / duration
                ),
                "throughput": self.pbar.total / duration,
                "total_hit_tokens": sum(self.performance_metrics["hit_tokens"]),
                "total_miss_tokens": sum(self.performance_metrics["miss_tokens"]),
                "cache_hit_rate": (
                    0
                    if sum(self.performance_metrics["prompt_len"]) == 0
                    else sum(self.performance_metrics["cached_tokens"])
                    / sum(self.performance_metrics["prompt_len"])
                ),
            },
        }
        if self.enable_round_barrier:
            performance_data["round"] = {}
            for round_num in range(self.num_rounds):
                round_key = f"round_{round_num}"
                round_metrics = self.performance_metrics[round_key]
                round_start = (
                    min(round_metrics["start_time"])
                    if round_metrics["start_time"]
                    else 0
                )
                round_end = (
                    max(round_metrics["end_time"])
                    if round_metrics["end_time"]
                    else 0
                )
                round_duration = round_end - round_start

                performance_data["round"][round_key] = {
                    "average_ttft": (
                        sum(round_metrics["ttft"]) / len(round_metrics["ttft"])
                        if round_metrics["ttft"]
                        else 0
                    ),
                    "average_itl": (
                        sum(round_metrics["itl"]) / len(round_metrics["itl"])
                        if round_metrics["itl"]
                        else 0
                    ),
                    "p90_itl": percentile(sorted(round_metrics["itl"]), 0.9),
                    "p99_itl": percentile(sorted(round_metrics["itl"]), 0.99),
                    "median_itl": percentile(sorted(round_metrics["itl"]), 0.5),
                    "max_itl": max_or_zero(sorted(round_metrics["itl"])),
                    "cache_hit_rate": (
                        0
                        if sum(round_metrics["prompt_len"]) == 0
                        else sum(round_metrics["cached_tokens"])
                        / sum(round_metrics["prompt_len"])
                    ),
                    "predicted_chr": (
                        sum(round_metrics["predicted_chr"])
                        / len(round_metrics["predicted_chr"])
                        if round_metrics["predicted_chr"]
                        else 0
                    ),
                    "request_count": len(round_metrics["ttft"]),
                    "total_hit_tokens": sum(round_metrics["hit_tokens"]),
                    "total_miss_tokens": sum(round_metrics["miss_tokens"]),
                    "output_token_throughput": (
                        sum(round_metrics["generated_len"]) / round_duration
                        if round_duration > 0
                        else 0
                    ),
                    "total_token_throughput": (
                        (
                            sum(round_metrics["miss_tokens"])
                            + sum(round_metrics["generated_len"])
                        )
                        / round_duration
                        if round_duration > 0
                        else 0
                    ),
                    "round_duration": round_duration,
                }
        print("All requests completed")
        print("Performance metrics summary:")
        print(
            f"  Total requests: {performance_data['summary']['total_requests']} at {performance_data['summary']['request_rate']} requests per second"
        )
        print(
            f"  Average Prompt Length: {performance_data['summary']['average_prompt_len']:.2f} tokens"
        )
        print(
            f"  Average Output Length: {performance_data['summary']['average_output_len']:.2f} tokens"
        )
        print(
            f"  P90 Prompt Length: {performance_data['summary']['p90_prompt_len']:.0f} tokens"
        )
        print(
            f"  P99 Prompt Length: {performance_data['summary']['p99_prompt_len']:.0f} tokens"
        )
        print(
            f"  P90 Output Length: {performance_data['summary']['p90_output_len']:.0f} tokens"
        )
        print(
            f"  P99 Output Length: {performance_data['summary']['p99_output_len']:.0f} tokens"
        )
        print(f"  Average TTFT: {performance_data['summary']['average_ttft']:.2f}")
        print(f"  P90 TTFT: {performance_data['summary']['p90_ttft']:.2f}")
        print(f"  P99 TTFT: {performance_data['summary']['p99_ttft']:.2f}")
        print(f"  Median TTFT: {performance_data['summary']['median_ttft']:.2f}")
        print(f"  Max TTFT: {performance_data['summary']['max_ttft']:.2f}")
        print(f"  Average ITL: {performance_data['summary']['average_itl']:.4f}")
        print(f"  P90 ITL: {performance_data['summary']['p90_itl']:.4f}")
        print(f"  P99 ITL: {performance_data['summary']['p99_itl']:.4f}")
        print(f"  Median ITL: {performance_data['summary']['median_itl']:.4f}")
        print(f"  Max ITL: {performance_data['summary']['max_itl']:.4f}")
        print(
            f"  Average latency: {performance_data['summary']['average_latency']:.2f}"
        )
        print(f"  P90 latency: {performance_data['summary']['p90_latency']:.2f}")
        print(f"  P99 latency: {performance_data['summary']['p99_latency']:.2f}")
        print(f"  Median latency: {performance_data['summary']['median_latency']:.2f}")
        print(f"  Max latency: {performance_data['summary']['max_latency']:.2f}")
        print(
            f"  Input token throughput: {performance_data['summary']['input_token_throughput']:.2f} tokens per second"
        )
        print(
            f"  Output token throughput: {performance_data['summary']['output_token_throughput']:.2f} tokens per second"
        )
        print(
            f"  Total token throughput (miss+out): {performance_data['summary']['total_token_throughput']:.2f} tokens per second"
        )
        print(
            f"  Request Throughput: {performance_data['summary']['throughput']:.2f} requests per second"
        )
        print(
            f"  Total Hit Tokens: {performance_data['summary']['total_hit_tokens']}"
        )
        print(
            f"  Total Miss Tokens: {performance_data['summary']['total_miss_tokens']}"
        )
        print(f"  Cache Hit Rate: {performance_data['summary']['cache_hit_rate']:.6f}")

        if self.enable_round_barrier:
            # Print round-basedsummary
            print("Per-round metrics:")
            if "round" in performance_data:
                for round_num in range(self.num_rounds):
                    round_key = f"round_{round_num}"
                    if round_key in performance_data["round"]:
                        round_data = performance_data["round"][round_key]
                        avg_ttft = round_data["average_ttft"]
                        avg_itl = round_data["average_itl"]
                        p90_itl = round_data["p90_itl"]
                        p99_itl = round_data["p99_itl"]
                        median_itl = round_data["median_itl"]
                        max_itl = round_data["max_itl"]
                        cache_hit_rate = round_data["cache_hit_rate"]
                        predicted_chr = round_data["predicted_chr"]
                        request_count = round_data["request_count"]
                        total_hit_tokens = round_data["total_hit_tokens"]
                        total_miss_tokens = round_data["total_miss_tokens"]
                        output_token_throughput = round_data["output_token_throughput"]
                        total_token_throughput = round_data["total_token_throughput"]
                        round_duration = round_data["round_duration"]
                        clients_in_round = self.clients_per_round[round_num]
                        print(
                            f"  Round {round_num}: Average TTFT = {avg_ttft:.2f}s, "
                            f"Average ITL = {avg_itl:.4f}s, "
                            f"P90 ITL = {p90_itl:.4f}s, "
                            f"P99 ITL = {p99_itl:.4f}s, "
                            f"Median ITL = {median_itl:.4f}s, "
                            f"Max ITL = {max_itl:.4f}s, "
                            f"Cache Hit Rate = {cache_hit_rate:.6f} (predicted: {predicted_chr:.6f}), "
                            f"Hit Tokens = {total_hit_tokens}, "
                            f"Miss Tokens = {total_miss_tokens}, "
                            f"Output Tput = {output_token_throughput:.2f} tok/s, "
                            f"Total Tput = {total_token_throughput:.2f} tok/s, "
                            f"Duration = {round_duration:.2f}s "
                            f"({request_count} requests, "
                            f"{clients_in_round} clients)"
                        )
                    else:
                        print(f"  Round {round_num}: No requests completed")

        return performance_data


if __name__ == "__main__":
    args = parse_args()
    flush_cache_url = f"http://{args.host}:{args.port}/flush_cache"

    random.seed(args.seed)
    np.random.seed(args.seed)

    model_name = os.path.basename(os.path.normpath(args.model_path))
    
    bench_data_dir = os.path.join(args.output_dir, "bench_data")
    os.makedirs(bench_data_dir, exist_ok=True)
    
    txt_filename = f"multi_turn_{model_name}_input{args.request_length}_output{args.output_length}_clients{args.num_clients}_concurrency{args.max_parallel}.txt"
    txt_filepath = os.path.join(bench_data_dir, txt_filename)
    
    if not args.log_file or args.log_file == "performance_metrics.jsonl":
        jsonl_filename = txt_filename.replace('.txt', '.jsonl')
        args.log_file = os.path.join(bench_data_dir, jsonl_filename)
    else:
        args.log_file = os.path.join(bench_data_dir, args.log_file)
    
    print(f"Output will be saved to: {txt_filepath}")
    
    with open(txt_filepath, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        
        print("=" * 80)
        print(f"Configuration:")
        print(f"  Model: {args.model_path}")
        print(f"  Num clients: {args.num_clients}")
        print(f"  Max parallel: {args.max_parallel}")
        print(f"  Request length: {args.request_length}")
        print(f"  Output length: {args.output_length}")
        print(f"  First round input length: {args.first_round_input_length}")
        print(f"  Num rounds: {args.num_rounds}")
        print(f"  Request rate: {args.request_rate}")
        print(f"  Host: {args.host}")
        print(f"  Port: {args.port}")
        print(f"  Dataset: {args.dataset_path}")
        print(f"  Range ratio: {args.range_ratio}")
        print(f"  Seed: {args.seed}")
        print("=" * 80)
        print()
        
        if args.disable_auto_run:
            print("Running with specified request rate...")
            request_rates = [args.request_rate]
        else:
            print("Auto-running with different request rates...")
            request_rates = [16, 14, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

        for rate in request_rates:
            args.request_rate = rate
            if args.api_format != "openai":
                requests.post(flush_cache_url)
                time.sleep(1)
            else:
                # vLLM/openai backend does not have /flush_cache
                time.sleep(1)
            performance_data = WorkloadGenerator(args).run()
            log_to_jsonl_file(performance_data, args.log_file, tag=args.tag)
            print()
        
        sys.stdout = original_stdout
        
    print(f"Results saved to: {txt_filepath}")
