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
import re
from utils.helpers import setup_ollama_client, is_ollama_running, list_ollama_models

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
        scenario, reasoning_trace, final_output = self._inference_layer(seed_data)

        # Create initial sample
        sample = GeneratedSample(
            seed=seed_data,
            scenario=scenario,
            reasoning_trace=reasoning_trace,
            final_output=final_output or self._generate_final_output(scenario, reasoning_trace),
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

    def _inference_layer(self, seed: str) -> tuple[str, ReasoningTrace, Optional[str]]:
        """
        Inference Layer: Generate scenario and reasoning trace using LLM

        Args:
            seed: Seed input

        Returns:
            Tuple of (scenario, reasoning_trace)
        """
        # Check if we should use simulation mode
        if not self._is_ollama_available():
            logger.info("Using enhanced simulation mode for data generation")
            return self._enhanced_simulation_inference(seed)

        try:
            # Try multiple models in order of preference
            available_models = self._get_available_models()
            model_name = self._select_best_model(available_models, self.config.model_name)

            if not model_name:
                logger.warning("No suitable LLM models available, falling back to enhanced simulation")
                return self._enhanced_simulation_inference(seed)

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
            response = self.ollama_client.generate(
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

            return scenario, reasoning_trace, final_output

        except Exception as e:
            logger.error(f"Error in LLM inference: {e}")
            logger.info("Falling back to enhanced simulation mode")
            return self._enhanced_simulation_inference(seed)

    def _simulation_critic_evaluation(self, sample: GeneratedSample, evaluation_start: float) -> CriticEvaluation:
        """
        Simulation mode critic evaluation that provides basic quality assessment
        without requiring LLM access.

        Args:
            sample: Generated sample to evaluate
            evaluation_start: Time when evaluation started

        Returns:
            CriticEvaluation with basic assessment
        """
        # Basic quality checks
        has_question = "question:" in sample.final_output.lower()
        has_answer = "answer:" in sample.final_output.lower()
        reasoning_length = len(sample.reasoning_trace.thoughts)
        output_length = len(sample.final_output.strip())

        # Scoring logic
        logic_score = 3  # Base score
        if has_question and has_answer:
            logic_score += 1  # +1 for proper Q&A format
        if reasoning_length >= 3:
            logic_score += 1  # +1 for detailed reasoning
        if output_length > 100:
            logic_score = min(5, logic_score + 1)  # +1 for substantial content, cap at 5

        coherence_score = 3  # Base score
        if reasoning_length >= 3:
            coherence_score += 1  # +1 for coherent reasoning steps
        if has_question and has_answer:
            coherence_score += 1  # +1 for logical Q&A structure

        # Validation based on quality threshold
        passes_validation = logic_score >= self.config.rejection_threshold

        # Generate feedback
        feedback_parts = []
        if has_question and has_answer:
            feedback_parts.append("Proper question-answer format maintained")
        else:
            feedback_parts.append("Consider using clear question-answer format")

        if reasoning_length >= 3:
            feedback_parts.append("Reasoning trace shows good depth")
        else:
            feedback_parts.append("Consider adding more detailed reasoning steps")

        if output_length > 100:
            feedback_parts.append("Content length is substantial")
        else:
            feedback_parts.append("Consider expanding the answer for more detail")

        feedback_text = ". ".join(feedback_parts)

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

    def _generate_final_output(self, scenario: str, reasoning_trace: ReasoningTrace) -> str:
        """
        Generate final output based on scenario and reasoning

        Args:
            scenario: Generated scenario
            reasoning_trace: Reasoning trace

        Returns:
            Final generated output
        """
        # Check if we have final output from enhanced simulation
        # Fallback Q&A generation
        topic = "machine learning concepts" if "machine learning" in scenario.lower() else "AI concepts"
        return (
            f"Question: What are the key concepts in {topic}?\n\n"
            f"Answer: {topic.replace('concepts', '').strip()} includes fundamental ideas like "
            f"algorithms, data processing, model evaluation, and practical applications. "
            f"Understanding these concepts is essential for building effective AI systems."
        )

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

            # Use model name as-is (Ollama expects the full model name with hyphens)
            model_name = self.config.model_name
            if not self.ollama_client:
                raise RuntimeError("Ollama client not available")
            response = self.ollama_client.generate(
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

        # Check if we should use simulation mode for evaluation
        if not self._is_ollama_available():
            logger.info("Using simulation mode for critic evaluation")
            return self._simulation_critic_evaluation(sample, evaluation_start)

        try:
            # Try to get available models for evaluation
            available_models = self._get_available_models()
            model_name = self._select_best_model(available_models, self.config.model_name)

            if not model_name:
                logger.warning("No suitable LLM models available for critic evaluation, falling back to simulation")
                return self._simulation_critic_evaluation(sample, evaluation_start)

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
            response = self.ollama_client.generate(
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
            return self._simulation_critic_evaluation(sample, evaluation_start)

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

    def _is_ollama_available(self) -> bool:
        """Check if Ollama is available and running"""
        return is_ollama_running(self.ollama_client)

    def _get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        return list_ollama_models(self.ollama_client)

    def _select_best_model(self, available_models: List[str], preferred_model: str) -> Optional[str]:
        """Select the best available model from preferences"""
        if preferred_model in available_models:
            return preferred_model

        # Fallback model preferences
        fallback_models = [
            'deepseek-r1:1.5b',
            'llama2:7b',
            'mistral:7b',
            'codellama:7b',
            'llama2:13b'
        ]

        for model in fallback_models:
            if model in available_models:
                logger.info(f"Preferred model '{preferred_model}' not available, using '{model}' instead")
                return model

        return None

    def _enhanced_simulation_inference(self, seed: str) -> tuple[str, ReasoningTrace, str]:
        """
        Enhanced simulation mode that generates diverse, realistic synthetic data
        without requiring LLM access.
        """
        import random

        # Diverse topic categories for better variety
        topic_templates = {
            "machine_learning": {
                "questions": [
                    "What are the key differences between supervised and unsupervised learning?",
                    "How does gradient descent work in neural networks?",
                    "What is the bias-variance tradeoff in machine learning?",
                    "How do decision trees handle categorical features?",
                    "What are the advantages of ensemble methods like Random Forest?"
                ],
                "scenarios": [
                    "A data scientist explaining ML concepts to a junior developer",
                    "An AI researcher discussing model training techniques",
                    "A machine learning engineer optimizing model performance",
                    "An educator teaching ML fundamentals to students"
                ]
            },
            "programming": {
                "questions": [
                    "What are the main differences between lists and tuples in Python?",
                    "How does garbage collection work in modern programming languages?",
                    "What are the benefits of functional programming paradigms?",
                    "How do design patterns improve software architecture?",
                    "What are the trade-offs between compiled and interpreted languages?"
                ],
                "scenarios": [
                    "A senior developer mentoring a junior programmer",
                    "A software architect designing scalable systems",
                    "A coding instructor explaining language features",
                    "A tech lead reviewing code quality standards"
                ]
            },
            "data_science": {
                "questions": [
                    "What statistical tests should be used for different data distributions?",
                    "How do you handle missing data in a real-world dataset?",
                    "What are the key principles of data normalization?",
                    "How do you detect and handle outliers in data analysis?",
                    "What are the differences between correlation and causation?"
                ],
                "scenarios": [
                    "A data analyst cleaning and preparing datasets",
                    "A statistician explaining hypothesis testing",
                    "A data engineer designing ETL pipelines",
                    "A business analyst interpreting data insights"
                ]
            },
            "ai_concepts": {
                "questions": [
                    "What are the main approaches to artificial intelligence?",
                    "How do neural networks simulate human learning?",
                    "What are the ethical considerations in AI development?",
                    "How do reinforcement learning algorithms work?",
                    "What are the current limitations of AI systems?"
                ],
                "scenarios": [
                    "An AI ethicist discussing responsible AI development",
                    "A researcher explaining neural network architectures",
                    "An AI engineer implementing reinforcement learning",
                    "A philosopher debating AI consciousness and intelligence"
                ]
            }
        }

        # Determine topic category from seed
        seed_lower = seed.lower()
        if "machine learning" in seed_lower or "ml" in seed_lower:
            category = "machine_learning"
        elif "programming" in seed_lower or "code" in seed_lower or "python" in seed_lower:
            category = "programming"
        elif "data" in seed_lower and ("science" in seed_lower or "analysis" in seed_lower):
            category = "data_science"
        else:
            category = "ai_concepts"

        templates = topic_templates[category]

        # Select random but diverse content
        question = random.choice(templates["questions"])
        scenario = random.choice(templates["scenarios"])

        # Generate detailed reasoning steps
        reasoning_steps = [
            f"Understanding the core question about {category.replace('_', ' ')}",
            "Analyzing the key concepts and their relationships",
            "Considering practical applications and examples",
            "Formulating a comprehensive and accurate answer",
            "Ensuring the explanation is clear and educational"
        ]

        # Create sophisticated final output based on category and question
        if category == "machine_learning":
            if "supervised" in question.lower() and "unsupervised" in question.lower():
                answer = """Supervised learning uses labeled training data to learn patterns and make predictions, while unsupervised learning finds hidden patterns in unlabeled data. Supervised learning includes tasks like classification and regression, whereas unsupervised learning focuses on clustering and dimensionality reduction. The key difference is the availability of labeled examples during training."""
            elif "gradient descent" in question.lower():
                answer = """Gradient descent is an optimization algorithm that minimizes the loss function by iteratively moving in the direction of the steepest descent. In neural networks, it updates weights by computing gradients through backpropagation. The learning rate controls step size, and techniques like momentum and Adam improve convergence."""
            elif "bias-variance" in question.lower():
                answer = """The bias-variance tradeoff represents the balance between model complexity and generalization. High bias leads to underfitting (oversimplified models), while high variance causes overfitting (models that memorize training data). The optimal model finds the sweet spot that minimizes total error on unseen data."""
            elif "decision trees" in question.lower():
                answer = """Decision trees handle categorical features by creating branches for each category value. For high-cardinality features, techniques like one-hot encoding or hierarchical splitting are used. Information gain or Gini impurity measures guide the splitting decisions to maximize predictive power."""
            else:  # ensemble methods
                answer = """Ensemble methods like Random Forest combine multiple weak learners to create a stronger model. They reduce overfitting through averaging predictions and introduce diversity via bootstrapping and random feature selection. Bagging, boosting, and stacking are common ensemble strategies."""
        elif category == "programming":
            if "lists" in question.lower() and "tuples" in question.lower():
                answer = """Lists are mutable sequences that can be modified after creation, while tuples are immutable and cannot be changed. Lists use square brackets and support operations like append() and remove(), whereas tuples use parentheses and are hashable for use as dictionary keys. Choose lists for dynamic collections and tuples for fixed data structures."""
            elif "garbage collection" in question.lower():
                answer = """Modern languages use automatic garbage collection to manage memory. Reference counting tracks object usage, while mark-and-sweep algorithms identify unreachable objects. Generational GC optimizes performance by focusing on short-lived objects. Languages like Python, Java, and Go implement sophisticated GC strategies."""
            elif "functional programming" in question.lower():
                answer = """Functional programming emphasizes immutability, pure functions, and higher-order functions. Benefits include easier testing, concurrency support, and mathematical reasoning. Languages like Haskell, Scala, and functional JavaScript demonstrate these principles through features like lambda expressions and monads."""
            elif "design patterns" in question.lower():
                answer = """Design patterns provide proven solutions to common software design problems. Creational patterns like Factory and Singleton manage object creation, while structural patterns like Adapter and Composite organize code relationships. Behavioral patterns like Observer and Strategy define communication between objects."""
            else:  # compiled vs interpreted
                answer = """Compiled languages translate code to machine language before execution, offering better performance but slower development cycles. Interpreted languages execute code line-by-line, providing faster iteration but potentially slower runtime performance. Modern approaches like JIT compilation blur these lines."""
        elif category == "data_science":
            if "statistical tests" in question.lower():
                answer = """Different data distributions require specific statistical tests. Normal distributions use t-tests and ANOVA, while non-parametric tests like Mann-Whitney U and Kruskal-Wallis handle non-normal data. Chi-square tests work with categorical variables, and correlation tests examine relationships between continuous variables."""
            elif "missing data" in question.lower():
                answer = """Missing data can be handled through deletion, imputation, or model-based approaches. Listwise deletion removes incomplete cases, while mean/median imputation fills missing values. Advanced techniques like multiple imputation and KNN imputation preserve data relationships and reduce bias."""
            elif "normalization" in question.lower():
                answer = """Data normalization scales features to comparable ranges. Min-max scaling rescales to [0,1], while z-score standardization centers data around zero with unit variance. Robust scaling handles outliers, and normalization ensures gradient-based algorithms converge efficiently."""
            elif "outliers" in question.lower():
                answer = """Outliers are detected using statistical methods like z-scores, IQR ranges, and isolation forests. Handling approaches include removal, transformation, or robust modeling. Understanding outlier causes helps determine whether they represent errors, rare events, or important phenomena."""
            else:  # correlation vs causation
                answer = """Correlation measures the relationship between variables, while causation implies one variable influences another. Spurious correlations can appear due to confounding factors, and establishing causation requires controlled experiments or rigorous statistical methods like regression discontinuity design."""
        else:  # ai_concepts
            if "approaches" in question.lower() and "artificial intelligence" in question.lower():
                answer = """AI encompasses symbolic AI (rule-based systems), connectionist AI (neural networks), evolutionary algorithms, and hybrid approaches. Modern AI combines deep learning for pattern recognition with symbolic reasoning for logical inference. Each approach has strengths for different problem domains."""
            elif "neural networks" in question.lower() and "learning" in question.lower():
                answer = """Neural networks simulate human learning through interconnected nodes called neurons. Forward propagation passes inputs through layers, while backpropagation adjusts weights using gradient descent. Activation functions introduce non-linearity, enabling complex function approximation and feature learning."""
            elif "ethical" in question.lower() and "ai" in question.lower():
                answer = """AI ethics addresses bias in training data, transparency in decision-making, privacy concerns, and societal impacts. Fairness, accountability, and human oversight are crucial. Ethical frameworks like those from the IEEE and EU AI Act guide responsible AI development and deployment."""
            elif "reinforcement learning" in question.lower():
                answer = """Reinforcement learning trains agents through reward-based feedback. Agents explore environments, learn from consequences, and maximize cumulative rewards. Q-learning and policy gradients are core algorithms, with applications in robotics, game playing, and autonomous systems."""
            else:  # limitations
                answer = """Current AI systems lack true understanding, struggle with abstract reasoning, and can exhibit unpredictable failures. They require massive data and computation, and face challenges with common sense, creativity, and emotional intelligence. Safety, alignment, and robustness remain active research areas."""

        final_output = f"Question: {question}\n\nAnswer: {answer}"

        # Store the final output for use by _generate_final_output method
        reasoning_trace = ReasoningTrace(
            thoughts=reasoning_steps,
            confidence_score=0.85  # High confidence for curated content
        )

        return scenario, reasoning_trace, final_output
