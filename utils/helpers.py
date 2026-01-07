"""
Syndgen Utilities Module

Helper functions and utilities for the Syndgen pipeline.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import ollama
import os

def setup_logging(level: int = logging.INFO):
    """Set up logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def print_stats(stats: Dict[str, Any]):
    """Print pipeline statistics in a readable format"""
    print(f"  Total Generated: {stats.get('total_generated', 0)}")
    print(f"  Total Valid: {stats.get('total_valid', 0)}")
    print(f"  Total Rejected: {stats.get('total_rejected', 0)}")
    print(f"  Rejection Rate: {stats.get('rejection_rate', 0):.1%}")
    print(f"  Avg Generation Time: {stats.get('avg_generation_time', 0):.3f}s")
    print(f"  Avg Evaluation Time: {stats.get('avg_evaluation_time', 0):.3f}s")
    print(f"  Runtime: {stats.get('runtime', 0):.2f}s")

    # Handle datetime objects if present
    if 'start_time' in stats and stats['start_time'] is not None:
        print(f"  Start Time: {format_timestamp(stats['start_time'])}")
    if 'end_time' in stats and stats['end_time'] is not None:
        print(f"  End Time: {format_timestamp(stats['end_time'])}")

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display"""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def validate_sample_quality(sample: Dict[str, Any]) -> bool:
    """
    Validate sample quality based on basic criteria

    Args:
        sample: Sample dictionary

    Returns:
        bool: True if sample meets quality criteria
    """
    # Check if sample has required fields
    required_fields = ['id', 'scenario', 'final_output', 'is_valid']
    if not all(field in sample for field in required_fields):
        return False

    # Check if final output has reasonable length
    final_output = sample.get('final_output', '')
    if len(final_output.strip()) < 50:  # Minimum 50 characters
        return False

    return True

def calculate_diversity_score(samples: list) -> float:
    """
    Calculate diversity score for a list of samples (simplified version)

    Args:
        samples: List of sample dictionaries

    Returns:
        float: Diversity score (0-1)
    """
    if not samples or len(samples) < 2:
        return 0.0

    # Simple diversity calculation based on unique scenarios
    unique_scenarios = set()
    for sample in samples:
        scenario = sample.get('scenario', '')
        if scenario:
            # Use first 50 characters as a simple hash
            unique_scenarios.add(scenario[:50])

    diversity_score = len(unique_scenarios) / len(samples)
    return diversity_score

def get_sample_summary(sample: Dict[str, Any]) -> str:
    """
    Get a summary of a sample for display

    Args:
        sample: Sample dictionary

    Returns:
        str: Formatted summary
    """
    summary = [
        f"ID: {sample.get('id', 'N/A')}",
        f"Valid: {sample.get('is_valid', False)}",
        f"Logic Score: {sample.get('reasoning_trace', {}).get('logic_score', 'N/A')}",
        f"Scenario: {sample.get('scenario', 'N/A')[:100]}...",
        f"Output Length: {len(sample.get('final_output', ''))} chars"
    ]
    return "\n".join(summary)

def setup_ollama_client() -> Optional[Any]:
    """
    Set up and configure the Ollama client for LLM integration

    Returns:
        Optional[Any]: Configured Ollama client or None if not available
    """
    try:
        # Check if Ollama is available
        ollama_client = ollama

        # Configure default settings
        ollama_host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

        # Test connection
        try:
            # Simple test to check if Ollama server is running
            models = ollama.list()
            logging.info(f"Ollama client initialized. Available models: {[m['name'] for m in models['models']]}")
            return ollama_client
        except Exception as e:
            logging.warning(f"Ollama server not available: {e}")
            return None

    except ImportError:
        logging.warning("Ollama package not available. Falling back to simulation mode.")
        return None
