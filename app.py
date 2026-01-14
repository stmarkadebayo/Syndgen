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
from utils.helpers import setup_logging
import logging
import ollama
import subprocess
import platform
import sys
import os

# Configure page
st.set_page_config(
    page_title="Syndgen - Synthetic Data Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_ollama_status():
    """Check if Ollama is available and running"""
    try:
        ollama.show()
        return True
    except Exception:
        return False

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

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'samples' not in st.session_state:
    st.session_state.samples = []
if 'stats' not in st.session_state:
    st.session_state.stats = {}

def main():
    """Main Streamlit application"""

    # Title and description
    st.title("🤖 Syndgen - Synthetic Data Generator")
    st.markdown("""
    **Localized Synthetic Data Generation & Distillation Engine**

    Generate high-quality synthetic Q&A pairs with Chain-of-Thought reasoning and LLM-as-judge evaluation.
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Ollama status
        ollama_status = check_ollama_status()
        if ollama_status:
            st.success("✅ Ollama Connected")
            operational_mode = "LLM Mode"
        else:
            st.error("❌ Ollama Not Available")
            if st.button("🔄 Try to Start Ollama"):
                with st.spinner("Starting Ollama..."):
                    if start_ollama():
                        st.success("Ollama started successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to start Ollama. Please start it manually.")
            operational_mode = "Enhanced Simulation Mode"

        st.info(f"**Mode:** {operational_mode}")

        # Generation parameters
        st.subheader("🎯 Generation Settings")

        model_name = st.selectbox(
            "Model",
            ["deepseek-r1:1.5b", "llama2:7b", "mistral:7b"],
            help="LLM model to use for generation"
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Sampling temperature (higher = more creative)"
        )

        max_tokens = st.slider(
            "Max Tokens",
            min_value=128,
            max_value=1024,
            value=512,
            step=64,
            help="Maximum tokens to generate"
        )

        rejection_threshold = st.slider(
            "Quality Threshold",
            min_value=1,
            max_value=5,
            value=4,
            help="Minimum logic score to accept (1-5)"
        )

        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=100,
            value=5,
            help="Number of samples to generate"
        )

        # Export settings
        st.subheader("📤 Export Settings")

        export_formats = st.multiselect(
            "Export Formats",
            ["JSONL", "Parquet", "SFT", "DPO"],
            default=["JSONL"],
            help="Formats to export generated data"
        )

        include_reasoning = st.checkbox(
            "Include Reasoning Traces",
            value=True,
            help="Include Chain-of-Thought reasoning in exports"
        )

        output_dir = st.text_input(
            "Output Directory",
            value="output",
            help="Directory to save exported files"
        )

        # Generate button
        generate_button = st.button(
            "🚀 Generate Samples",
            type="primary",
            use_container_width=True
        )

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📝 Generation Results")

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

                    export_config = ExportConfig(
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
                    st.success(f"✅ Generated {len(samples)} samples, {len(valid_samples)} valid")

                    # Display samples
                    for i, sample in enumerate(valid_samples[:5]):  # Show first 5
                        with st.expander(f"Sample {i+1}: {sample.id[:8]}...", expanded=(i==0)):

                            # Quality indicators
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                score_color = "🟢" if sample.reasoning_trace.logic_score >= 4 else "🟡" if sample.reasoning_trace.logic_score >= 3 else "🔴"
                                st.metric("Logic Score", f"{score_color} {sample.reasoning_trace.logic_score}/5")
                            with col_b:
                                st.metric("Confidence", f"{sample.reasoning_trace.confidence_score:.2f}")
                            with col_c:
                                st.metric("Valid", "✅ Yes" if sample.is_valid else "❌ No")

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
                    st.error(f"❌ Generation failed: {str(e)}")
                    st.exception(e)

        # Display existing samples if available
        elif st.session_state.samples:
            valid_samples = st.session_state.samples
            st.info(f"📊 Showing {len(valid_samples)} previously generated samples")

            for i, sample in enumerate(valid_samples[:5]):
                with st.expander(f"Sample {i+1}: {sample.id[:8]}...", expanded=(i==0)):
                    st.metric("Logic Score", f"{sample.reasoning_trace.logic_score}/5")
                    st.markdown(f"**Scenario:** {sample.scenario}")
                    st.code(sample.final_output, language="text")

    with col2:
        st.header("📊 Statistics")

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
                    st.subheader("📈 Performance")
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
        st.header("💾 Export Data")

        if st.session_state.samples and export_formats:
            if st.button("📤 Export Selected Formats", use_container_width=True):
                with st.spinner("Exporting data..."):
                    try:
                        export_config = ExportConfig(
                            format="jsonl",
                            output_dir=output_dir,
                            include_reasoning=include_reasoning
                        )

                        exporter = DataExporter(export_config)

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

                        st.success("✅ Export completed!")
                        st.markdown("**Exported files:**")
                        for fmt, filename in exported_files:
                            st.code(f"{fmt}: {filename}", language="text")

                    except Exception as e:
                        st.error(f"❌ Export failed: {str(e)}")

        # Help section
        st.header("❓ Help")

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
