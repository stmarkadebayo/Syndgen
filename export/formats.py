"""
Syndgen Export Formats Module

Implements export functionality for different data formats:
- JSONL (JSON Lines)
- Parquet
- SFT (Supervised Fine-Tuning)
- DPO (Direct Preference Optimization)
"""

import json
import os
from typing import List, Optional, Union
import pandas as pd
from ..core.schema import GeneratedSample, SFTFormat, DPOFormat, ExportConfig
from ..pipeline.core import SyndgenPipeline
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataExporter:
    """Main export class for Syndgen data"""

    def __init__(self, config: ExportConfig):
        self.config = config
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """Ensure output directory exists"""
        os.makedirs(self.config.output_dir, exist_ok=True)

    def export_samples(self, samples: List[GeneratedSample], format_type: Optional[str] = None):
        """
        Export samples in the specified format

        Args:
            samples: List of generated samples
            format_type: Optional format override
        """
        export_format = format_type or self.config.format
        filename = self._generate_filename(export_format)

        if export_format == "jsonl":
            self._export_jsonl(samples, filename)
        elif export_format == "parquet":
            self._export_parquet(samples, filename)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        logger.info(f"Exported {len(samples)} samples to {filename}")
        return filename

    def _generate_filename(self, format_type: str) -> str:
        """Generate output filename with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(
            self.config.output_dir,
            f"syndgen_export_{timestamp}.{format_type}"
        )

    def _export_jsonl(self, samples: List[GeneratedSample], filename: str):
        """Export samples in JSONL format"""
        with open(filename, 'w', encoding='utf-8') as f:
            for sample in samples:
                sample_dict = sample.dict()

                # Optionally exclude reasoning traces
                if not self.config.include_reasoning:
                    sample_dict.pop('reasoning_trace', None)

                f.write(json.dumps(sample_dict, ensure_ascii=False) + '\n')

    def _export_parquet(self, samples: List[GeneratedSample], filename: str):
        """Export samples in Parquet format"""
        # Convert samples to pandas DataFrame
        sample_dicts = []
        for sample in samples:
            sample_dict = sample.dict()

            # Optionally exclude reasoning traces
            if not self.config.include_reasoning:
                sample_dict.pop('reasoning_trace', None)

            sample_dicts.append(sample_dict)

        df = pd.DataFrame(sample_dicts)

        # Handle compression
        compression = self.config.compression if self.config.compression else None

        df.to_parquet(filename, compression=compression)

    def export_sft_format(self, samples: List[GeneratedSample], filename: Optional[str] = None):
        """
        Export samples in SFT (Supervised Fine-Tuning) format

        Args:
            samples: List of generated samples
            filename: Optional custom filename
        """
        if filename is None:
            filename = self._generate_filename("sft.jsonl")

        sft_samples = []
        for sample in samples:
            # Convert to SFT format
            sft_sample = SFTFormat(
                instruction=f"Answer the following question: {self._extract_question(sample.final_output)}",
                input=self._extract_question(sample.final_output),
                output=self._extract_answer(sample.final_output),
                reasoning_trace=sample.reasoning_trace if self.config.include_reasoning else None
            )
            sft_samples.append(sft_sample)

        # Export as JSONL
        with open(filename, 'w', encoding='utf-8') as f:
            for sft_sample in sft_samples:
                f.write(json.dumps(sft_sample.dict(), ensure_ascii=False) + '\n')

        logger.info(f"Exported {len(sft_samples)} samples in SFT format to {filename}")
        return filename

    def export_dpo_format(self, samples: List[GeneratedSample], filename: Optional[str] = None):
        """
        Export samples in DPO (Direct Preference Optimization) format

        Args:
            samples: List of generated samples
            filename: Optional custom filename
        """
        if filename is None:
            filename = self._generate_filename("dpo.jsonl")

        dpo_samples = []
        for sample in samples:
            # For DPO, we need chosen and rejected responses
            # In this simulation, we'll use the valid samples as "chosen"
            # and create slightly modified versions as "rejected"

            if sample.is_valid:
                question = self._extract_question(sample.final_output)
                answer = self._extract_answer(sample.final_output)

                # Create a "rejected" version (slightly worse answer)
                rejected_answer = self._create_rejected_answer(answer)

                dpo_sample = DPOFormat(
                    prompt=f"Question: {question}\n\nPlease provide a detailed answer:",
                    chosen=answer,
                    rejected=rejected_answer,
                    reasoning_trace=sample.reasoning_trace if self.config.include_reasoning else None
                )
                dpo_samples.append(dpo_sample)

        # Export as JSONL
        with open(filename, 'w', encoding='utf-8') as f:
            for dpo_sample in dpo_samples:
                f.write(json.dumps(dpo_sample.dict(), ensure_ascii=False) + '\n')

        logger.info(f"Exported {len(dpo_samples)} samples in DPO format to {filename}")
        return filename

    def _extract_question(self, final_output: str) -> str:
        """Extract question from final output"""
        if "Question:" in final_output:
            question_part = final_output.split("Question:")[1].split("\n\n")[0].strip()
            return question_part
        return "What is the answer to this question?"

    def _extract_answer(self, final_output: str) -> str:
        """Extract answer from final output"""
        if "Answer:" in final_output:
            answer_part = final_output.split("Answer:")[1].strip()
            return answer_part
        return "This is the answer to the question."

    def _create_rejected_answer(self, original_answer: str) -> str:
        """Create a slightly worse version of the answer for DPO rejected sample"""
        # Simple approach: shorten the answer and make it less detailed
        sentences = original_answer.split('. ')
        if len(sentences) > 1:
            return sentences[0] + '.'
        return "Short answer."

    def batch_export(self, pipeline: SyndgenPipeline, batch_size: int = 100):
        """
        Generate and export a batch of samples

        Args:
            pipeline: SyndgenPipeline instance
            batch_size: Number of samples to generate and export
        """
        logger.info(f"Generating batch of {batch_size} samples...")
        samples = pipeline.generate_batch(batch_size)

        # Filter valid samples if needed
        valid_samples = [s for s in samples if s.is_valid]
        logger.info(f"Generated {len(samples)} samples, {len(valid_samples)} valid")

        # Export in primary format
        primary_filename = self.export_samples(valid_samples)

        # Also export in SFT and DPO formats
        sft_filename = self.export_sft_format(valid_samples)
        dpo_filename = self.export_dpo_format(valid_samples)

        return {
            'primary_export': primary_filename,
            'sft_export': sft_filename,
            'dpo_export': dpo_filename,
            'stats': pipeline.get_stats()
        }
