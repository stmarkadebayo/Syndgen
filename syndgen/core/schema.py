"""
Syndgen Core Schema Module

This module defines Pydantic models for data validation and structure
in the Syndgen synthetic data generation pipeline.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

class ReasoningTrace(BaseModel):
    """Model for capturing Chain-of-Thought reasoning traces"""
    thoughts: List[str] = Field(..., description="List of reasoning steps")
    confidence_score: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)
    logic_score: Optional[int] = Field(None, description="Critic logic score (1-5)")

class GeneratedSample(BaseModel):
    """Model for a single generated data sample"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier")
    seed: Optional[str] = Field(None, description="Original seed input")
    scenario: str = Field(..., description="Generated scenario or context")
    reasoning_trace: ReasoningTrace = Field(..., description="Chain-of-Thought reasoning")
    final_output: str = Field(..., description="Final generated output")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    is_valid: bool = Field(default=False, description="Validation status")

    @validator('final_output')
    def validate_output_length(cls, v):
        """Ensure output has reasonable length"""
        if len(v.strip()) < 10:
            raise ValueError("Output too short - must be at least 10 characters")
        return v

class CriticEvaluation(BaseModel):
    """Model for critic evaluation results"""
    sample_id: str = Field(..., description="Reference to evaluated sample")
    logic_score: int = Field(..., description="Logic consistency score (1-5)", ge=1, le=5)
    coherence_score: int = Field(..., description="Coherence score (1-5)", ge=1, le=5)
    feedback: str = Field(..., description="Critic feedback and reasoning")
    passes_validation: bool = Field(..., description="Whether sample passes validation")
    evaluation_time: float = Field(..., description="Evaluation time in seconds", gt=0)

class SFTFormat(BaseModel):
    """Model for Supervised Fine-Tuning format"""
    instruction: str = Field(..., description="Instruction or prompt")
    input: Optional[str] = Field(None, description="Optional input data")
    output: str = Field(..., description="Expected output")
    reasoning_trace: Optional[ReasoningTrace] = Field(None, description="Optional reasoning trace")

class DPOFormat(BaseModel):
    """Model for Direct Preference Optimization format"""
    prompt: str = Field(..., description="Prompt or instruction")
    chosen: str = Field(..., description="Preferred/chosen response")
    rejected: str = Field(..., description="Rejected/less preferred response")
    reasoning_trace: Optional[ReasoningTrace] = Field(None, description="Optional reasoning trace")

class GenerationConfig(BaseModel):
    """Configuration for data generation"""
    model_name: str = Field("deepseek-r1-1.5b", description="LLM model to use")
    temperature: float = Field(0.7, description="Sampling temperature", ge=0, le=2)
    max_tokens: int = Field(512, description="Maximum tokens to generate", ge=50, le=2048)
    top_p: float = Field(0.9, description="Nucleus sampling probability", ge=0, le=1)
    rejection_threshold: int = Field(4, description="Minimum logic score to accept (1-5)", ge=1, le=5)
    max_retries: int = Field(1, description="Maximum regeneration attempts", ge=0, le=3)

class ExportConfig(BaseModel):
    """Configuration for data export"""
    format: str = Field("jsonl", description="Export format (jsonl, parquet)")
    output_dir: str = Field("output", description="Output directory")
    include_reasoning: bool = Field(True, description="Include reasoning traces in export")
    batch_size: int = Field(100, description="Batch size for export", ge=1, le=1000)
    compression: Optional[str] = Field(None, description="Compression format (gzip, bz2)")

class PipelineStats(BaseModel):
    """Statistics for the generation pipeline"""
    total_generated: int = Field(0, description="Total samples generated")
    total_valid: int = Field(0, description="Total valid samples")
    total_rejected: int = Field(0, description="Total rejected samples")
    rejection_rate: float = Field(0.0, description="Rejection rate (0-1)")
    avg_generation_time: float = Field(0.0, description="Average generation time (seconds)")
    avg_evaluation_time: float = Field(0.0, description="Average evaluation time (seconds)")
    start_time: datetime = Field(default_factory=datetime.now, description="Pipeline start time")
    end_time: Optional[datetime] = Field(None, description="Pipeline end time")
