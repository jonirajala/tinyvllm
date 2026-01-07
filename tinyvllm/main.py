"""
TinyVLLM CLI - Run LLM inference with continuous batching.

Usage:
    python -m tinyvllm.main --model /path/to/llama --prompt "Hello"
    echo "Hello" | python -m tinyvllm.main --model /path/to/llama
"""

import argparse
import sys

from tinygrad import Device

from tinyvllm.model.llama import create_llama
from tinyvllm.model.weights import load_llama_weights
from tinyvllm.model.tokenizer import load_tokenizer
from tinyvllm.engine.sampling import SamplingParams
from tinyvllm.engine.engine import LLMEngine


def main():
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
    parser.add_argument("--no-async-output", action="store_true",
                        help="Disable async output processing (enabled by default)")
    args = parser.parse_args()

    # Set device if specified
    if args.device:
        Device.DEFAULT = args.device.upper()

    print(f"Using device: {Device.DEFAULT}", file=sys.stderr)

    # Get prompt from argument or stdin
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = sys.stdin.read().strip()
        if not prompt:
            print("Error: No prompt provided", file=sys.stderr)
            sys.exit(1)

    # Load model
    config, weights = load_llama_weights(args.model)
    model = create_llama(config, weights)
    tokenizer = load_tokenizer(args.model)

    # Create engine
    engine = LLMEngine(
        model, tokenizer,
        num_scheduler_steps=args.num_scheduler_steps,
        async_output=not args.no_async_output,
    )

    # Create sampling params
    params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    # Add request and run
    engine.add_request(prompt, params)

    for output in engine.run():
        print(output.text)


if __name__ == "__main__":
    main()
