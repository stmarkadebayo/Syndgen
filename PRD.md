
# PRD: Project SynDGen

**Subtitle:** Localized Synthetic Data Generation & Distillation Engine

**Version:** 1.0.0

**Author:** [Your Name]

**Status:** Draft / Implementation Phase

---

## 1. Executive Summary

**Syndgen** is a local-first synthetic data generation pipeline powered by the **DeepSeek R1 1.5B** reasoning model. It automates the creation of high-quality, logic-dense datasets for fine-tuning smaller specialized models or testing software systems. Unlike standard generators, SynGen utilizes **Chain-of-Thought (CoT)** distillation and an automated **"LLM-as-a-Judge"** critic loop to ensure data veracity without API costs or privacy leaks.

## 2. Problem Statement

* **Data Scarcity:** Training specialized ML models requires massive amounts of labeled data, which is often expensive or unavailable.
* **Privacy Concerns:** Using OpenAI/Anthropic APIs to generate data from sensitive "seed" information risks data exposure.
* **Low Quality:** Standard LLM outputs often contain hallucinations or lack the logical "steps" required for reasoning-heavy tasks.

## 3. Goals & Objectives

* **G1:** Generate 1,000+ verified samples per hour on consumer-grade hardware (Ollama + 1.5B model).
* **G2:** Implement a **Rejection Sampling** mechanism to automatically discard low-quality data.
* **G3:** Store data in industry-standard formats (`.jsonl`, `.parquet`) ready for Hugging Face integration.

## 4. Technical Architecture

### 4.1 The Pipeline

The system operates as a **tri-stage pipeline**:

1. **Seed Layer:** Takes user-defined parameters or "seed" examples.
2. **Inference Layer (Generator):** R1 1.5B generates a scenario, an internal reasoning trace (`<think>`), and a final output.
3. **Audit Layer (Critic):** A second pass (or parallel instance) evaluates the logic of the reasoning trace against the final output.

### 4.2 Tech Stack

* **Inference Engine:** Ollama (DeepSeek R1 1.5B)
* **Orchestration:** Python 3.10+
* **Data Handling:** Pandas / PyArrow
* **Validation:** Pydantic (for structured JSON enforcement)

---

## 5. Feature Requirements

### FR1: Reasoning-Augmented Generation

The system must capture the `<think>` tags from DeepSeek R1. This allows the final dataset to include the "Internal Monologue," which is crucial for training "Reasoning Models."

### FR2: The "Critic" Validation Loop

Every generated record must be scored by a secondary prompt.

* **Metric:** Logic Score (1–5).
* **Action:** If Score < 4, the record is discarded or sent back for one "Regeneration" attempt.

### FR3: Export Modules

Support for multiple downstream formats:

* **SFT Format:** Instruction/Response pairs for fine-tuning.
* **DPO Format:** (Prompt, Chosen, Rejected) pairs for Preference Optimization.

---

## 6. Success Metrics (KPIs)

* **Inference Speed:** Average seconds per valid record (Target: < 5s on 8GB VRAM).
* **Rejection Rate:** Percentage of samples flagged by the Critic (Target: 10–20% for high-quality thresholds).
* **Diversity Score:** Semantic uniqueness between generated samples (measured via cosine similarity).

---

## 7. Strategic Roadmap

* **Phase 1:** Core CLI tool for generating simple Q&A pairs.
* **Phase 2:** Implementation of the "Critic" loop and Rejection Sampling.
* **Phase 3:** Integration of a local UI (Gradio/Streamlit) for real-time monitoring of the "Thinking" process.
* **Phase 4:** Multi-model support (using R1 1.5B to generate and Llama 3 to critique).

---

## 8. Risk Assessment

| Risk | Mitigation Strategy |
| --- | --- |
| **Model Hallucination** | Use strict Pydantic schemas and high-temperature filtering. |
| **Data Repetition** | Implement a "Memory Buffer" of previously generated topics to ensure variety. |
| **Hardware Bottleneck** | Use 4-bit quantization (GGUF) via Ollama to maintain speed. |

---

### Would you like me to generate the **`schema.py`** (using Pydantic) for this project so you can ensure the data output is perfectly structured for the JSON export?