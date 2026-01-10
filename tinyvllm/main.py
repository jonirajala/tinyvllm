"""TinyVLLM CLI - LLM inference with continuous batching."""

import argparse
import sys

from tinygrad import Device

from tinyvllm.model.llama import create_llama
from tinyvllm.model.weights import load_llama_weights
from tinyvllm.model.tokenizer import load_tokenizer
from tinyvllm.core.sampling import SamplingParams
from tinyvllm.core.engine import LLMEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyVLLM - Minimal LLM inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--prompt", type=str, help="Text to complete (reads from stdin if not provided)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k filtering (0 to disable)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling threshold")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--device", type=str, help="Device to run on (CPU, CUDA, METAL)")
    parser.add_argument("--num-scheduler-steps", type=int, default=4,
                        help="Decode steps per scheduler cycle (1=low latency, 4+=high throughput)")
    parser.add_argument("--no-async-output", action="store_true", help="Disable async output processing")
    parser.add_argument("--num-blocks", type=int, default=None,
                        help="KV cache blocks (auto-calculated if not set)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="Fraction of GPU memory to use (default: 0.9)")
    args = parser.parse_args()

    if args.device:
        Device.DEFAULT = args.device.upper()

    print(f"Using device: {Device.DEFAULT}", file=sys.stderr)

    if args.prompt:
        prompt = args.prompt
    else:
        prompt = sys.stdin.read().strip()
        if not prompt:
            print("Error: No prompt provided", file=sys.stderr)
            sys.exit(1)

    config, weights = load_llama_weights(args.model)
    model = create_llama(config, weights)
    tokenizer = load_tokenizer(args.model)

    engine = LLMEngine(
        model, tokenizer,
        num_scheduler_steps=args.num_scheduler_steps,
        async_output=not args.no_async_output,
        num_blocks=args.num_blocks,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    engine.add_request(prompt, params)

    for output in engine.run():
        print(output.text)


if __name__ == "__main__":
    main()
