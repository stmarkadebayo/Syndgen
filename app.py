"""
Syndgen Web Interface

A Streamlit-based web interface for the Syndgen synthetic data generation pipeline.
"""

import streamlit as st
import time
from typing import List, Optional
import pandas as pd
import json

# Import Syndgen components
from core.schema import GenerationConfig, ExportConfig
from pipeline.core import SyndgenPipeline
from export.formats import DataExporter
from utils.helpers import (
    setup_logging,
    load_config,
    configure_ollama_env,
    get_ollama_client,
    is_ollama_running,
    list_ollama_models
)
import logging
import subprocess
import platform
import sys
import os

config = load_config()
ollama_config = config.get('ollama', {})
generation_config = config.get('generation', {})
export_config = config.get('export', {})
ui_config = config.get('ui', {})
configure_ollama_env(ollama_config)

# Configure page
st.set_page_config(
    page_title=ui_config.get('page_title', 'Syndgen - Synthetic Data Generator'),
    page_icon=ui_config.get('page_icon', 'S'),
    layout=ui_config.get('layout', 'wide'),
    initial_sidebar_state=ui_config.get('sidebar_state', 'expanded')
)

def check_ollama_status():
    """Check if Ollama is available and running"""
    client = get_ollama_client()
    return is_ollama_running(client)

def start_ollama():
    """Attempt to start Ollama server"""
    try:
        if platform.system() == "Windows":
            subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            subprocess.Popen(["ollama", "serve"])
        time.sleep(5)  # Wait for startup
        return check_ollama_status()
    except Exception:
        return False

def pull_model(model_name: str) -> bool:
    """Attempt to pull a model via Ollama."""
    client = get_ollama_client()
    if not client:
        return False
    try:
        client.pull(model_name)
        return True
    except Exception:
        return False


# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'samples' not in st.session_state:
    st.session_state.samples = []
if 'stats' not in st.session_state:
    st.session_state.stats = {}
if 'last_config' not in st.session_state:
    st.session_state.last_config = None
if 'ollama_status_cache' not in st.session_state:
    st.session_state.ollama_status_cache = {'status': False, 'timestamp': 0, 'models': []}

def main():
    """Main Streamlit application"""

    # Title and description
    st.title("Syndgen - Synthetic Data Generator")
    st.markdown("""
    **Localized Synthetic Data Generation & Distillation Engine**

    Generate high-quality synthetic Q&A pairs with Chain-of-Thought reasoning and LLM-as-judge evaluation.
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        default_model = ollama_config.get("default_model", "deepseek-r1:1.5b")

        # Ollama status and model detection
        ollama_status = check_ollama_status()
        available_models = []
        if ollama_status:
            try:
                available_models = list_ollama_models(get_ollama_client())
                st.success(f"[OK] Ollama Connected ({len(available_models)} models available)")
                operational_mode = "LLM Mode"

                if default_model not in available_models:
                    st.warning(f"[WARN] Default model '{default_model}' not found")
                    if st.button("Pull Default Model"):
                        with st.spinner(f"Pulling {default_model}..."):
                            if pull_model(default_model):
                                st.success("Model pulled successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to pull model. Please pull it manually.")
            except Exception:
                st.warning("[WARN] Ollama running but unable to list models")
                operational_mode = "Enhanced Simulation Mode"
        else:
            st.error("[ERROR] Ollama Not Available")
            if st.button("Try to Start Ollama"):
                with st.spinner("Starting Ollama..."):
                    if start_ollama():
                        st.success("Ollama started successfully!")
                        if pull_model(default_model):
                            st.success("Model pulled successfully!")
                        else:
                            st.warning("Model pull failed. You can try again or pull manually.")
                        st.rerun()
                    else:
                        st.error("Failed to start Ollama. Please start it manually.")
            operational_mode = "Enhanced Simulation Mode"

        st.info(f"**Mode:** {operational_mode}")

        # Generation parameters
        st.subheader("Generation Settings")

        # Dynamic model selection based on available models
        fallback_models = ollama_config.get("fallback_models", ["deepseek-r1:1.5b", "llama2:7b", "mistral:7b"])
        default_model = ollama_config.get("default_model", "deepseek-r1:1.5b")
        if available_models:
            # Put available models first, then fallback options
            model_options = available_models + [m for m in fallback_models if m not in available_models]
        else:
            model_options = fallback_models

        model_name = st.selectbox(
            "Model",
            model_options,
            index=model_options.index(default_model) if default_model in model_options else 0,
            help="LLM model to use for generation"
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=generation_config.get("default_temperature", 0.7),
            step=0.1,
            help="Sampling temperature (higher = more creative)"
        )

        max_tokens = st.slider(
            "Max Tokens",
            min_value=128,
            max_value=1024,
            value=generation_config.get("default_max_tokens", 512),
            step=64,
            help="Maximum tokens to generate"
        )

        rejection_threshold = st.slider(
            "Quality Threshold",
            min_value=1,
            max_value=5,
            value=generation_config.get("default_rejection_threshold", 4),
            help="Minimum logic score to accept (1-5)"
        )

        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=generation_config.get("max_batch_size", 100),
            value=min(5, generation_config.get("max_batch_size", 100)),
            help="Number of samples to generate"
        )

        # Export settings
        st.subheader("Export Settings")

        export_formats = st.multiselect(
            "Export Formats",
            ["JSONL", "Parquet", "SFT", "DPO"],
            default=export_config.get("default_formats", ["JSONL"]),
            help="Formats to export generated data"
        )

        include_reasoning = st.checkbox(
            "Include Reasoning Traces",
            value=export_config.get("include_reasoning", True),
            help="Include Chain-of-Thought reasoning in exports"
        )

        output_dir = st.text_input(
            "Output Directory",
            value=export_config.get("default_output_dir", "output"),
            help="Directory to save exported files"
        )

        # Generate button
        generate_button = st.button(
            "Generate Samples",
            type="primary",
            use_container_width=True
        )

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Generation Results")

        if generate_button:
            with st.spinner("Generating samples..."):
                try:
                    # Create configuration
                    gen_config = GenerationConfig(
                        model_name=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        rejection_threshold=rejection_threshold,
                        max_retries=1
                    )

                    export_settings = ExportConfig(
                        format="jsonl",  # Default, will be overridden
                        output_dir=output_dir,
                        include_reasoning=include_reasoning
                    )

                    # Initialize pipeline
                    pipeline = SyndgenPipeline(gen_config)
                    st.session_state.pipeline = pipeline

                    # Generate samples
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    samples = []
                    for i in range(batch_size):
                        status_text.text(f"Generating sample {i+1}/{batch_size}...")
                        sample = pipeline.generate_sample()
                        samples.append(sample)
                        progress_bar.progress((i + 1) / batch_size)

                    progress_bar.empty()
                    status_text.empty()

                    # Filter valid samples
                    valid_samples = [s for s in samples if s.is_valid]
                    st.session_state.samples = valid_samples

                    # Show results
                    st.success(f"[OK] Generated {len(samples)} samples, {len(valid_samples)} valid")

                    # Display samples
                    for i, sample in enumerate(valid_samples[:5]):  # Show first 5
                        with st.expander(f"Sample {i+1}: {sample.id[:8]}...", expanded=(i==0)):

                            # Quality indicators
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                score_label = "High" if sample.reasoning_trace.logic_score >= 4 else "Medium" if sample.reasoning_trace.logic_score >= 3 else "Low"
                                st.metric("Logic Score", f"{score_label} {sample.reasoning_trace.logic_score}/5")
                            with col_b:
                                st.metric("Confidence", f"{sample.reasoning_trace.confidence_score:.2f}")
                            with col_c:
                                st.metric("Valid", "Yes" if sample.is_valid else "No")

                            # Scenario
                            st.markdown(f"**Scenario:** {sample.scenario}")

                            # Reasoning trace
                            if st.checkbox(f"Show reasoning trace", key=f"reasoning_{i}"):
                                st.markdown("**Reasoning Trace:**")
                                for j, thought in enumerate(sample.reasoning_trace.thoughts, 1):
                                    st.markdown(f"{j}. {thought}")

                            # Final output
                            st.markdown("**Generated Content:**")
                            st.code(sample.final_output, language="text")

                except Exception as e:
                    error_msg = str(e)
                    if "model" in error_msg.lower() and "not found" in error_msg.lower():
                        st.error("[ERROR] Model not found. Please select an available model or ensure Ollama is running with the correct model.")
                        st.info("[TIP] Try running: `ollama pull deepseek-r1:1.5b` in your terminal")
                    elif "connection" in error_msg.lower():
                        st.error("[ERROR] Connection failed. Please check if Ollama is running.")
                        st.info("[TIP] Try clicking 'Try to Start Ollama' or run: `ollama serve` in another terminal")
                    else:
                        st.error(f"[ERROR] Generation failed: {error_msg}")
                        with st.expander("Show full error details"):
                            st.exception(e)

        # Display existing samples if available
        elif st.session_state.samples:
            valid_samples = st.session_state.samples
            st.info(f"Showing {len(valid_samples)} previously generated samples")

            for i, sample in enumerate(valid_samples[:5]):
                with st.expander(f"Sample {i+1}: {sample.id[:8]}...", expanded=(i==0)):
                    st.metric("Logic Score", f"{sample.reasoning_trace.logic_score}/5")
                    st.markdown(f"**Scenario:** {sample.scenario}")
                    st.code(sample.final_output, language="text")

    with col2:
        st.header("Statistics")

        if st.session_state.pipeline:
            try:
                stats = st.session_state.pipeline.get_stats()

                # Key metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Generated", stats.get('total_generated', 0))
                    st.metric("Valid Samples", stats.get('total_valid', 0))
                with col2:
                    st.metric("Rejection Rate", f"{stats.get('rejection_rate', 0):.1%}")
                    st.metric("Avg Gen Time", f"{stats.get('avg_generation_time', 0):.3f}s")

                # Performance chart placeholder
                if stats.get('total_generated', 0) > 0:
                    st.subheader("Performance")
                    perf_data = {
                        'Metric': ['Generation Time', 'Evaluation Time'],
                        'Average (seconds)': [
                            stats.get('avg_generation_time', 0),
                            stats.get('avg_evaluation_time', 0)
                        ]
                    }
                    st.bar_chart(pd.DataFrame(perf_data).set_index('Metric'))

            except Exception as e:
                st.warning("Statistics not available")

        # Export section
        st.header("Export Data")

        if st.session_state.samples and export_formats:
            if st.button("Export Selected Formats", use_container_width=True):
                with st.spinner("Exporting data..."):
                    try:
                        export_settings = ExportConfig(
                            format="jsonl",
                            output_dir=output_dir,
                            include_reasoning=include_reasoning
                        )

                        exporter = DataExporter(export_settings)

                        exported_files = []

                        for fmt in export_formats:
                            if fmt == "JSONL":
                                filename = exporter._generate_filename("jsonl")
                                exporter._export_jsonl(st.session_state.samples, filename)
                                exported_files.append(("JSONL", filename))
                            elif fmt == "Parquet":
                                filename = exporter._generate_filename("parquet")
                                exporter._export_parquet(st.session_state.samples, filename)
                                exported_files.append(("Parquet", filename))
                            elif fmt == "SFT":
                                filename = exporter.export_sft_format(st.session_state.samples)
                                exported_files.append(("SFT", filename))
                            elif fmt == "DPO":
                                filename = exporter.export_dpo_format(st.session_state.samples)
                                exported_files.append(("DPO", filename))

                        st.success("[OK] Export completed!")
                        st.markdown("**Exported files:**")
                        for fmt, filename in exported_files:
                            st.code(f"{fmt}: {filename}", language="text")

                    except Exception as e:
                        st.error(f"[ERROR] Export failed: {str(e)}")

        # Help section
        st.header("Help")

        with st.expander("How to use"):
            st.markdown("""
            1. **Configure** your generation parameters in the sidebar
            2. **Click "Generate Samples"** to create synthetic data
            3. **Review** the generated samples and their quality scores
            4. **Export** the data in your preferred formats

            **Tips:**
            - Higher temperature = more creative but potentially less coherent
            - Quality threshold filters out low-scoring samples
            - Reasoning traces show the model's thought process
            """)

        with st.expander("About Syndgen"):
            st.markdown("""
            **Syndgen** generates high-quality synthetic data using:

            - **Chain-of-Thought Reasoning**: Step-by-step thinking process
            - **LLM-as-Judge Evaluation**: Automated quality assessment
            - **Rejection Sampling**: Ensures only high-quality data
            - **Multiple Export Formats**: Ready for ML training pipelines

            Perfect for creating training data when real data is scarce or sensitive.
            """)

if __name__ == "__main__":
    # Set up logging to suppress most messages in Streamlit
    setup_logging(logging.WARNING)

    main()
