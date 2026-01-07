"""
Syndgen Pipeline Core Module

Implements the tri-stage pipeline architecture:
1. Seed Layer
2. Inference Layer (Generator)
3. Audit Layer (Critic)
"""

import time
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from core.schema import (
    GeneratedSample,
    ReasoningTrace,
    CriticEvaluation,
    GenerationConfig,
    PipelineStats
)
import logging
import ollama
import re
from utils.helpers import setup_ollama_client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyndgenPipeline:
    """Main pipeline class for synthetic data generation"""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.stats = PipelineStats()
        self.memory_buffer = set()  # For tracking generated topics to ensure diversity
        self.ollama_client = setup_ollama_client()

    def generate_sample(self, seed: Optional[str] = None) -> GeneratedSample:
        """
        Generate a single data sample through the tri-stage pipeline

        Args:
            seed: Optional seed input for generation

        Returns:
            GeneratedSample with complete data
        """
        start_time = time.time()

        # Stage 1: Seed Layer
        seed_data = self._seed_layer(seed)

        # Stage 2: Inference Layer (Generator)
        scenario, reasoning_trace = self._inference_layer(seed_data)

        # Create initial sample
        sample = GeneratedSample(
            seed=seed_data,
            scenario=scenario,
            reasoning_trace=reasoning_trace,
            final_output=self._generate_final_output(scenario, reasoning_trace),
            metadata={"generation_time": time.time() - start_time}
        )

        # Stage 3: Audit Layer (Critic)
        evaluation = self._audit_layer(sample)

        # Update sample based on evaluation
        sample.is_valid = evaluation.passes_validation
        sample.reasoning_trace.logic_score = evaluation.logic_score

        # Update stats
        self._update_stats(sample, evaluation)

        return sample

    def _seed_layer(self, seed: Optional[str]) -> str:
        """
        Seed Layer: Process seed input or generate from scratch

        Args:
            seed: Optional seed input

        Returns:
            Processed seed data
        """
        if seed:
            # Use provided seed
            return seed
        else:
            # Generate a basic seed (in real implementation, this would use LLM)
            return "Generate a sample question-answer pair about machine learning concepts"

    def _inference_layer(self, seed: str) -> tuple[str, ReasoningTrace]:
        """
        Inference Layer: Generate scenario and reasoning trace using LLM

        Args:
            seed: Seed input

        Returns:
            Tuple of (scenario, reasoning_trace)
        """
        try:
            # Create prompt for Chain-of-Thought generation
            cot_prompt = f"""You are an expert AI assistant specializing in {seed}.
            Please generate a high-quality question-answer pair with detailed reasoning.

            Follow this exact format:
            <think>
            [Step-by-step reasoning process with at least 3-5 detailed thoughts]
            </think>

            <scenario>
            [Detailed scenario describing the context and user intent]
            </scenario>

            <output>
            Question: [Clear, well-formulated question]
            Answer: [Detailed, accurate answer]
            </output>

            Ensure the reasoning trace shows clear logical progression and the final output is high-quality and informative."""

            # Call Ollama API
            # Fix model name format (replace last hyphen with colon for Ollama)
            model_name = self.config.model_name.rsplit('-', 1)[0] + ':' + self.config.model_name.rsplit('-', 1)[1]
            response = ollama.generate(
                model=model_name,
                prompt=cot_prompt,
                options={
                    'temperature': self.config.temperature,
                    'top_p': 0.9,
                    'num_predict': self.config.max_tokens
                }
            )

            # Parse the response to extract reasoning, scenario, and output
            generated_text = response['response']

            # Extract reasoning trace
            reasoning_match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL)
            reasoning_steps = []
            if reasoning_match:
                reasoning_text = reasoning_match.group(1).strip()
                reasoning_steps = [step.strip() for step in reasoning_text.split('\n') if step.strip()]
            else:
                # Fallback: extract bullet points or numbered lists
                reasoning_steps = re.findall(r'[\d\*\-]\s*(.*)', generated_text)
                if not reasoning_steps:
                    reasoning_steps = ["Analyzing the topic", "Formulating key concepts", "Generating appropriate question and answer"]

            # Extract scenario
            scenario_match = re.search(r'<scenario>(.*?)</scenario>', generated_text, re.DOTALL)
            scenario = scenario_match.group(1).strip() if scenario_match else "Expert AI generating educational content about " + seed

            # Extract final output
            output_match = re.search(r'<output>(.*?)</output>', generated_text, re.DOTALL)
            final_output = output_match.group(1).strip() if output_match else self._generate_final_output_from_reasoning(reasoning_steps, seed)

            # Create reasoning trace
            reasoning_trace = ReasoningTrace(
                thoughts=reasoning_steps,
                confidence_score=min(0.95, 0.7 + len(reasoning_steps) * 0.05)  # Higher confidence for more detailed reasoning
            )

            return scenario, reasoning_trace

        except Exception as e:
            logger.error(f"Error in LLM inference: {e}")
            # Fallback to simulation mode
            reasoning_steps = [
                "Analyzing the topic of " + seed,
                "Identifying key concepts and relationships",
                "Formulating appropriate question and answer pair",
                "Ensuring logical consistency and accuracy"
            ]

            reasoning_trace = ReasoningTrace(
                thoughts=reasoning_steps,
                confidence_score=0.8
            )

            scenario = f"Generating educational content about {seed} for learning purposes"
            return scenario, reasoning_trace

    def _generate_final_output(self, scenario: str, reasoning_trace: ReasoningTrace) -> str:
        """
        Generate final output based on scenario and reasoning

        Args:
            scenario: Generated scenario
            reasoning_trace: Reasoning trace

        Returns:
            Final generated output
        """
        # This method is now mostly used as fallback
        # The main output comes from LLM in _inference_layer
        if not hasattr(self, '_llm_final_output'):
            # Fallback Q&A generation
            topic = "machine learning concepts" if "machine learning" in scenario.lower() else "AI concepts"
            return (
                f"Question: What are the key concepts in {topic}?\n\n"
                f"Answer: {topic.replace('concepts', '').strip()} includes fundamental ideas like "
                f"algorithms, data processing, model evaluation, and practical applications. "
                f"Understanding these concepts is essential for building effective AI systems."
            )
        else:
            return self._llm_final_output

    def _generate_final_output_from_reasoning(self, reasoning_steps: List[str], seed: str) -> str:
        """Fallback method to generate output from reasoning steps"""
        try:
            # Create a prompt to generate final output from reasoning
            reasoning_text = '\n'.join([f"{i+1}. {step}" for i, step in enumerate(reasoning_steps)])
            output_prompt = f"""Based on this reasoning process:
            {reasoning_text}

            Please generate a high-quality question and answer pair about {seed}.

            Format:
            Question: [Your question here]
            Answer: [Your detailed answer here]"""

            # Fix model name format (replace hyphen with colon for Ollama)
            model_name = self.config.model_name.replace('-', ':')
            response = ollama.generate(
                model=model_name,
                prompt=output_prompt,
                options={
                    'temperature': self.config.temperature,
                    'num_predict': 256
                }
            )

            return response['response'].strip()

        except Exception:
            # Ultimate fallback
            return (
                f"Question: What are the key aspects of {seed}?\n\n"
                f"Answer: {seed} involves several important components that work together "
                f"to achieve specific goals in artificial intelligence and machine learning."
            )

    def _audit_layer(self, sample: GeneratedSample) -> CriticEvaluation:
        """
        Audit Layer: Critic evaluation of generated sample using LLM

        Args:
            sample: Generated sample to evaluate

        Returns:
            CriticEvaluation with scores and feedback
        """
        evaluation_start = time.time()

        try:
            # Create critic prompt for LLM-based evaluation
            reasoning_text = '\n'.join([f"{i+1}. {thought}" for i, thought in enumerate(sample.reasoning_trace.thoughts)])
            critic_prompt = f"""You are an expert AI critic evaluating the quality of generated content.
            Please analyze the following sample and provide detailed evaluation:

            <sample>
            Scenario: {sample.scenario}

            Reasoning Trace:
            {reasoning_text}

            Final Output:
            {sample.final_output}
            </sample>

            Evaluation Criteria:
            1. Logic Score (1-5): Assess the logical consistency and reasoning quality
            2. Coherence Score (1-5): Evaluate how well the reasoning supports the final output
            3. Feedback: Provide specific, constructive criticism
            4. Validation: Does this sample meet quality standards? (yes/no)

            Respond in this exact format:
            <evaluation>
            logic_score: [1-5]
            coherence_score: [1-5]
            feedback: [Detailed feedback here]
            passes_validation: [yes/no]
            </evaluation>"""

            # Call Ollama for critic evaluation
            # Fix model name format (replace hyphen with colon for Ollama)
            model_name = self.config.model_name.replace('-', ':')
            response = ollama.generate(
                model=model_name,
                prompt=critic_prompt,
                options={
                    'temperature': 0.3,  # Lower temperature for more consistent evaluations
                    'num_predict': 256
                }
            )

            # Parse evaluation response
            eval_text = response['response']

            # Extract scores and feedback using regex
            logic_score_match = re.search(r'logic_score:\s*(\d)', eval_text)
            coherence_score_match = re.search(r'coherence_score:\s*(\d)', eval_text)
            feedback_match = re.search(r'feedback:\s*(.*?)(?=\n|$)', eval_text, re.DOTALL)
            validation_match = re.search(r'passes_validation:\s*(yes|no)', eval_text)

            # Parse scores with fallback
            logic_score = int(logic_score_match.group(1)) if logic_score_match else 3
            coherence_score = int(coherence_score_match.group(1)) if coherence_score_match else 3

            # Parse feedback
            feedback_text = feedback_match.group(1).strip() if feedback_match else "Automated evaluation completed"

            # Parse validation
            passes_validation = validation_match.group(1) == 'yes' if validation_match else (logic_score >= self.config.rejection_threshold)

            # Ensure scores are within valid range
            logic_score = max(1, min(5, logic_score))
            coherence_score = max(1, min(5, coherence_score))

        except Exception as e:
            logger.error(f"Error in LLM critic evaluation: {e}")
            # Fallback to basic evaluation
            has_question = "Question:" in sample.final_output
            has_answer = "Answer:" in sample.final_output
            reasoning_coherent = len(sample.reasoning_trace.thoughts) >= 3

            logic_score = 5 if (has_question and has_answer and reasoning_coherent) else 3
            coherence_score = 5 if reasoning_coherent else 3
            passes_validation = logic_score >= self.config.rejection_threshold

            feedback_text = "Fallback evaluation: " + ("Sample meets basic criteria" if passes_validation else "Sample needs improvement")

        evaluation_time = time.time() - evaluation_start
        evaluation_time = max(evaluation_time, 0.001)  # Ensure positive time

        return CriticEvaluation(
            sample_id=sample.id,
            logic_score=logic_score,
            coherence_score=coherence_score,
            feedback=feedback_text,
            passes_validation=passes_validation,
            evaluation_time=evaluation_time
        )

    def _update_stats(self, sample: GeneratedSample, evaluation: CriticEvaluation):
        """Update pipeline statistics"""
        self.stats.total_generated += 1
        if sample.is_valid:
            self.stats.total_valid += 1
        else:
            self.stats.total_rejected += 1

        # Update rejection rate
        if self.stats.total_generated > 0:
            self.stats.rejection_rate = self.stats.total_rejected / self.stats.total_generated

        # Update average times (simple moving average)
        gen_time = sample.metadata.get('generation_time', 0)
        eval_time = evaluation.evaluation_time

        if self.stats.total_generated == 1:
            self.stats.avg_generation_time = gen_time
            self.stats.avg_evaluation_time = eval_time
        else:
            # Exponential moving average
            alpha = 0.1  # Smoothing factor
            self.stats.avg_generation_time = (
                alpha * gen_time +
                (1 - alpha) * self.stats.avg_generation_time
            )
            self.stats.avg_evaluation_time = (
                alpha * eval_time +
                (1 - alpha) * self.stats.avg_evaluation_time
            )

    def generate_batch(self, batch_size: int = 10, seed: Optional[str] = None) -> List[GeneratedSample]:
        """
        Generate a batch of samples

        Args:
            batch_size: Number of samples to generate
            seed: Optional seed for all samples

        Returns:
            List of generated samples
        """
        samples = []
        for i in range(batch_size):
            logger.info(f"Generating sample {i+1}/{batch_size}")
            sample = self.generate_sample(seed)

            # If sample fails validation and retries are allowed, try again
            if not sample.is_valid and self.config.max_retries > 0:
                for retry in range(self.config.max_retries):
                    logger.info(f"Retry {retry+1} for sample {i+1}")
                    retry_sample = self.generate_sample(seed)
                    if retry_sample.is_valid:
                        sample = retry_sample
                        break

            samples.append(sample)

        return samples

    def get_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics as dictionary"""
        stats_dict = self.stats.dict()
        stats_dict['runtime'] = (
            (self.stats.end_time - self.stats.start_time).total_seconds()
            if self.stats.end_time else
            (datetime.now() - self.stats.start_time).total_seconds()
        )
        return stats_dict

    def reset_stats(self):
        """Reset pipeline statistics"""
        self.stats = PipelineStats()
