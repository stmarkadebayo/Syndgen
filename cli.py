"""
Syndgen CLI Module

Command-line interface for the Syndgen synthetic data generation pipeline.
"""

import argparse
import sys
import logging
from core.schema import GenerationConfig, ExportConfig
from pipeline.core import SyndgenPipeline
from export.formats import DataExporter
from utils.helpers import (
    setup_logging,
    print_stats,
    load_config,
    get_ollama_client,
    list_ollama_models
)
import os
import subprocess
import platform
import time

def main():
    """Main CLI entry point"""
    config = load_config()
    ollama_config = config.get("ollama", {})
    generation_config = config.get("generation", {})
    export_config = config.get("export", {})

    parser = argparse.ArgumentParser(
        description="Syndgen - Localized Synthetic Data Generation & Distillation Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Main arguments
    parser.add_argument(
        "--mode",
        choices=["generate", "export", "batch"],
        default="generate",
        help="Operation mode"
    )

    # Generation arguments
    max_batch_size = generation_config.get("max_batch_size", 100)
    batch_default = min(10, max_batch_size)

    parser.add_argument(
        "--batch-size",
        type=int,
        default=batch_default,
        help="Number of samples to generate"
    )

    parser.add_argument(
        "--seed",
        type=str,
        help="Optional seed input for generation"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default=ollama_config.get("default_model", "deepseek-r1:1.5b"),
        help="LLM model to use"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=generation_config.get("default_temperature", 0.7),
        help="Sampling temperature"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=generation_config.get("default_max_tokens", 512),
        help="Maximum tokens to generate"
    )

    parser.add_argument(
        "--rejection-threshold",
        type=int,
        default=generation_config.get("default_rejection_threshold", 4),
        choices=range(1, 6),
        help="Minimum logic score to accept (1-5)"
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=generation_config.get("max_retries", 1),
        help="Maximum regeneration attempts"
    )

    # Export configuration
    default_formats = export_config.get("default_formats") or ["JSONL"]

    parser.add_argument(
        "--export-format",
        type=str,
        default=default_formats[0].lower(),
        choices=["jsonl", "parquet"],
        help="Export format"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=export_config.get("default_output_dir", "output"),
        help="Output directory"
    )

    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        default=export_config.get("include_reasoning", True),
        help="Include reasoning traces in export"
    )

    parser.add_argument(
        "--no-reasoning",
        dest="include_reasoning",
        action="store_false",
        help="Exclude reasoning traces from export"
    )

    parser.add_argument(
        "--compression",
        type=str,
        choices=["gzip", "bz2"],
        help="Compression format for parquet"
    )

    # Debug/verbose
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output showing reasoning traces and evaluation details"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output with full pipeline internals and timing"
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO if args.verbose else logging.WARNING
    setup_logging(log_level)

    # Check and start Ollama if needed
    ollama_available = check_and_start_ollama()

    # Determine operational mode
    if ollama_available:
        operational_mode = "LLM Mode (Ollama available)"
    else:
        operational_mode = "Enhanced Simulation Mode (Ollama not available)"

    print(f"[INFO] Operational Mode: {operational_mode}")
    logging.info(f"Starting Syndgen pipeline in {operational_mode}...")

    try:
        # Create configuration
        gen_config = GenerationConfig(
            model_name=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            rejection_threshold=args.rejection_threshold,
            max_retries=args.max_retries
        )

        export_config = ExportConfig(
            format=args.export_format,
            output_dir=args.output_dir,
            include_reasoning=args.include_reasoning,
            compression=args.compression
        )

        # Initialize pipeline and exporter
        pipeline = SyndgenPipeline(gen_config)
        exporter = DataExporter(export_config)

        if args.mode == "generate":
            # Generate single sample
            sample = pipeline.generate_sample(args.seed)

            print(f"Generated sample ID: {sample.id}")
            print(f"Valid: {sample.is_valid}")
            print(f"Logic score: {sample.reasoning_trace.logic_score}")

            # Show verbose/debug output
            if args.verbose or args.debug:
                print(f"\n[DETAILS] Sample Details:")
                print(f"   Scenario: {sample.scenario}")
                print(f"   Seed: {sample.seed or 'None'}")
                print(f"   Created: {sample.created_at}")
                print(f"   Generation Time: {sample.metadata.get('generation_time', 0):.3f}s")

                if args.verbose:
                    print(f"\n[TRACE] Reasoning Trace (Logic Score: {sample.reasoning_trace.logic_score}):")
                    for i, thought in enumerate(sample.reasoning_trace.thoughts, 1):
                        print(f"   {i}. {thought}")
                    print(f"   Confidence: {sample.reasoning_trace.confidence_score:.2f}")

                if args.debug:
                    print(f"\n[DEBUG] Debug Information:")
                    print(f"   Model: {gen_config.model_name}")
                    print(f"   Temperature: {gen_config.temperature}")
                    print(f"   Max Tokens: {gen_config.max_tokens}")
                    print(f"   Rejection Threshold: {gen_config.rejection_threshold}")

            print(f"\n[OUTPUT] Final Output:")
            print(sample.final_output)

        elif args.mode == "export":
            # Generate and export batch
            samples = pipeline.generate_batch(args.batch_size, args.seed)
            valid_samples = [s for s in samples if s.is_valid]

            print(f"Generated {len(samples)} samples, {len(valid_samples)} valid")

            # Export in all formats
            filename = exporter.export_samples(valid_samples)
            sft_filename = exporter.export_sft_format(valid_samples)
            dpo_filename = exporter.export_dpo_format(valid_samples)

            print(f"\nExported to:")
            print(f"  Primary: {filename}")
            print(f"  SFT: {sft_filename}")
            print(f"  DPO: {dpo_filename}")

        elif args.mode == "batch":
            # Generate and export batch with stats
            result = exporter.batch_export(pipeline, args.batch_size)

            print(f"\nBatch export completed:")
            print(f"  Primary: {result['primary_export']}")
            print(f"  SFT: {result['sft_export']}")
            print(f"  DPO: {result['dpo_export']}")

            print("\nPipeline Statistics:")
            print_stats(result['stats'])

        # Print final stats
        stats = pipeline.get_stats()
        print("\nFinal Statistics:")
        try:
            print_stats(stats)
        except Exception as e:
            print(f"Error printing stats: {e}")
            # Print basic stats without datetime objects
            print(f"  Total Generated: {stats.get('total_generated', 0)}")
            print(f"  Total Valid: {stats.get('total_valid', 0)}")
            print(f"  Total Rejected: {stats.get('total_rejected', 0)}")
            print(f"  Rejection Rate: {stats.get('rejection_rate', 0):.1%}")
            print(f"  Avg Generation Time: {stats.get('avg_generation_time', 0):.3f}s")
            print(f"  Avg Evaluation Time: {stats.get('avg_evaluation_time', 0):.3f}s")
            print(f"  Runtime: {stats.get('runtime', 0):.2f}s")

    except Exception as e:
        logging.error(f"Error in Syndgen pipeline: {e}")
        sys.exit(1)

def check_and_start_ollama():
    """
    Check if Ollama is running and start it if needed.
    Returns True if Ollama is available, False otherwise.
    """
    try:
        client = get_ollama_client()
        if not client:
            print("[WARN] Ollama package not installed. Install it to enable LLM mode.")
            return False
        client.list()
        print("[OK] Ollama server is running")
        return True
    except Exception:
        print("[WARN] Ollama server not detected. Attempting to start automatically...")

        try:
            # Start Ollama server based on platform
            if platform.system() == "Windows":
                print("[INFO] Starting Ollama server on Windows...")
                # Try to start Ollama server directly without new window
                try:
                    # First check if ollama.exe exists in PATH
                    subprocess.run(["where", "ollama"], check=True, capture_output=True)
                    # Start Ollama server in background
                    subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NO_WINDOW)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print("[WARN] Ollama not found in PATH. Please ensure Ollama is installed and in your PATH.")
                    print("       Download from: https://ollama.com/download")
                    return False
            else:
                print("[INFO] Starting Ollama server on Unix-like system...")
                subprocess.Popen(["ollama", "serve"])

            # Wait for server to start
            print("[INFO] Waiting for Ollama server to start...")
            time.sleep(8)  # Increased wait time

            # Verify it started and try to pull model
            try:
                client = get_ollama_client()
                if not client:
                    print("[WARN] Ollama package not installed. Install it to enable LLM mode.")
                    return False
                client.list()
                print("[OK] Ollama server started successfully!")

                # Check if model exists, if not pull it
                try:
                    model_names = list_ollama_models(client)
                    if "deepseek-r1:1.5b" not in model_names:
                        print("[INFO] Pulling DeepSeek model (this may take a few minutes)...")
                        client.pull("deepseek-r1:1.5b")
                        print("[OK] Model pulled successfully!")
                except Exception:
                    print("[WARN] Could not check/pull model automatically")

                return True
            except Exception:
                print("[WARN] Ollama server started but not responding. Continuing in simulation mode.")
                return False
        except Exception as e:
            print(f"[ERROR] Could not start Ollama automatically: {e}")
            print("[INFO] Please start Ollama manually:")
            print("       1. Run: ollama serve")
            print("       2. In another terminal: ollama pull deepseek-r1:1.5b")
            print("       3. Restart Syndgen")
            return False


def print_help():

    """Print help information"""
    print("""
Syndgen - Localized Synthetic Data Generation & Distillation Engine

Usage:
  python main.py [OPTIONS]

Modes:
  generate      Generate a single sample
  export        Generate and export a batch of samples
  batch         Generate, export, and show detailed statistics

Examples:
  # Generate single sample
  python main.py --mode generate

  # Generate and export 50 samples
  python main.py --mode export --batch-size 50

  # Generate batch with custom settings
  python main.py --mode batch --batch-size 100 --temperature 0.9 --rejection-threshold 3

  # Export to parquet with compression
  python main.py --mode export --export-format parquet --compression gzip
""")

if __name__ == "__main__":
    main()
