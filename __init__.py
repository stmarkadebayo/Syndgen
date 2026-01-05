"""
Syndgen - Localized Synthetic Data Generation & Distillation Engine

Core package for synthetic data generation with Chain-of-Thought distillation
and automated critic validation.
"""

__version__ = "1.0.0"
__author__ = "Syndgen Team"
__license__ = "MIT"

# Import main components for easy access
from .core.schema import (
    GenerationConfig,
    ExportConfig,
    GeneratedSample,
    ReasoningTrace,
    CriticEvaluation,
    SFTFormat,
    DPOFormat,
    PipelineStats
)
from .pipeline.core import SyndgenPipeline
from .export.formats import DataExporter
from .cli import main

# Package-level constants
DEFAULT_MODEL = "deepseek-r1-1.5b"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_EXPORT_FORMAT = "jsonl"

__all__ = [
    # Core schema
    'GenerationConfig',
    'ExportConfig',
    'GeneratedSample',
    'ReasoningTrace',
    'CriticEvaluation',
    'SFTFormat',
    'DPOFormat',
    'PipelineStats',

    # Pipeline
    'SyndgenPipeline',

    # Export
    'DataExporter',

    # CLI
    'main',

    # Constants
    'DEFAULT_MODEL',
    'DEFAULT_OUTPUT_DIR',
    'DEFAULT_EXPORT_FORMAT',
    '__version__'
]
