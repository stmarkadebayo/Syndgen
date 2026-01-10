"""
Syndgen CLI Module

Command-line interface for the Syndgen synthetic data generation pipeline.
"""

import argparse
import sys
import logging
from typing import Optional
from core.schema import GenerationConfig, ExportConfig
from pipeline.core import SyndgenPipeline
from export.formats import DataExporter
from utils.helpers import setup_logging, print_stats
import os
import subprocess
import platform
import time
import ollama

def main():
    """Main CLI entry point"""
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
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
        default="deepseek-r1-1.5b",
        help="LLM model to use"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )

    parser.add_argument(
        "--rejection-threshold",
        type=int,
        default=4,
        choices=range(1, 6),
        help="Minimum logic score to accept (1-5)"
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum regeneration attempts"
    )

    # Export configuration
    parser.add_argument(
        "--export-format",
        type=str,
        default="jsonl",
        choices=["jsonl", "parquet"],
        help="Export format"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory"
    )

    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        default=True,
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
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
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

    print(f"🔧 Operational Mode: {operational_mode}")
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
            print("\nFinal Output:")
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
        # Test if Ollama server is available
        ollama.show()
        print("✅ Ollama server is running")
        return True
    except Exception:
        print("🔄 Ollama server not detected. Attempting to start automatically...")

        try:
            # Start Ollama server based on platform
            if platform.system() == "Windows":
                print("💻 Starting Ollama server on Windows...")
                # Try to start Ollama server directly without new window
                try:
                    # First check if ollama.exe exists in PATH
                    subprocess.run(["where", "ollama"], check=True, capture_output=True)
                    # Start Ollama server in background
                    subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NO_WINDOW)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print("⚠️ Ollama not found in PATH. Please ensure Ollama is installed and in your PATH.")
                    print("   Download from: https://ollama.com/download")
                    return False
            else:
                print("🐧 Starting Ollama server on Unix-like system...")
                subprocess.Popen(["ollama", "serve"])

            # Wait for server to start
            print("⏳ Waiting for Ollama server to start...")
            time.sleep(8)  # Increased wait time

            # Verify it started and try to pull model
            try:
                ollama.show()
                print("✅ Ollama server started successfully!")

                # Check if model exists, if not pull it
                try:
                    models = ollama.list()['models']
                    model_names = [model['name'] for model in models]
                    if "deepseek-r1-1.5b" not in model_names:
                        print("📥 Pulling DeepSeek model (this may take a few minutes)...")
                        ollama.pull("deepseek-r1-1.5b")
                        print("✅ Model pulled successfully!")
                except:
                    print("⚠️ Could not check/pull model automatically")

                return True
            except:
                print("❌ Ollama server started but not responding. Continuing in simulation mode.")
                return False
        except Exception as e:
            print(f"❌ Could not start Ollama automatically: {e}")
            print("💡 Please start Ollama manually:")
            print("   1. Run: ollama serve")
            print("   2. In another terminal: ollama pull deepseek-r1-1.5b")
            print("   3. Restart Syndgen")
            return False

def print_help():
    """Print help information"""
    print("""
Syndgen - Localized Synthetic Data Generation & Distillation Engine

Usage:
  python -m syndgen [OPTIONS]

Modes:
  generate      Generate a single sample
  export        Generate and export a batch of samples
  batch         Generate, export, and show detailed statistics

Examples:
  # Generate single sample
  python -m syndgen --mode generate

  # Generate and export 50 samples
  python -m syndgen --mode export --batch-size 50

  # Generate batch with custom settings
  python -m syndgen --mode batch --batch-size 100 --temperature 0.9 --rejection-threshold 3

  # Export to parquet with compression
  python -m syndgen --mode export --export-format parquet --compression gzip
""")

if __name__ == "__main__":
    main()
