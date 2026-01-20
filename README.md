# Syndgen - Localized Synthetic Data Generation & Distillation Engine

![Syndgen Logo](https://via.placeholder.com/150?text=SYNDGEN)

**Local-first synthetic data generation pipeline powered by DeepSeek R1 1.5B reasoning model**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)


---

## Executive Summary

**Syndgen** is a localized synthetic data generation pipeline that automates the creation of high-quality, logic-dense datasets for fine-tuning smaller specialized models or testing software systems. Unlike standard generators, Syndgen utilizes **Chain-of-Thought (CoT)** distillation and an automated **"LLM-as-a-Judge"** critic loop to ensure data veracity without API costs or privacy leaks.

---

## Key Features

### **Tri-Stage Pipeline Architecture**
1. **Seed Layer**: Processes user-defined parameters or "seed" examples
2. **Inference Layer**: DeepSeek R1 1.5B generates scenarios with reasoning traces
3. **Audit Layer**: Secondary evaluation ensures logical consistency

### **Quality Control System**
- **Rejection Sampling**: Automatic filtering of low-quality data
- **Logic Scoring**: 1-5 scale evaluation by critic model
- **Confidence Metrics**: Built-in quality assurance

### **Multiple Export Formats**
- **JSONL**: Standard JSON Lines format
- **Parquet**: Columnar storage for big data
- **SFT**: Supervised Fine-Tuning format
- **DPO**: Direct Preference Optimization format

### **Performance Monitoring**
- Real-time statistics tracking
- Rejection rate analysis
- Generation speed metrics

---

## Installation

### Prerequisites
- Python 3.10+
- pip (Python package manager)

### Install Core Dependencies
```bash
pip install -r requirements.txt
```

### Optional: Install Ollama for LLM Integration
```bash
# Install Ollama (for actual LLM functionality)
curl -fsSL https://ollama.com/install.sh | sh

# Pull DeepSeek R1 1.5B model
ollama pull deepseek-r1:1.5b
```

---

## Quick Start

### Web Interface (Recommended)
```bash
# Install dependencies (including Streamlit)
pip install -r requirements.txt

# Launch the web interface
streamlit run app.py
```

### Command Line Interface

#### Generate a Single Sample
```bash
python main.py --mode generate
```

#### Generate with Detailed Reasoning (--verbose)
```bash
python main.py --mode generate --verbose
```

#### Generate with Full Debug Info (--debug)
```bash
python main.py --mode generate --debug
```

#### Generate and Export a Batch
```bash
python main.py --mode export --batch-size 50
```

#### Full Batch with Statistics
```bash
python main.py --mode batch --batch-size 100 --temperature 0.9
```

---

## Configuration Options

### Generation Configuration
| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `deepseek-r1:1.5b` | LLM model to use |
| `--temperature` | `0.7` | Sampling temperature (0.0-2.0) |
| `--max-tokens` | `512` | Maximum tokens to generate |
| `--rejection-threshold` | `4` | Minimum logic score to accept (1-5) |
| `--max-retries` | `1` | Maximum regeneration attempts |

### Export Configuration
| Option | Default | Description |
|--------|---------|-------------|
| `--export-format` | `jsonl` | Export format (jsonl, parquet) |
| `--output-dir` | `output` | Output directory |
| `--include-reasoning` | `true` | Include reasoning traces |
| `--compression` | `None` | Compression (gzip, bz2) |

### Example: Custom Configuration
```bash
python main.py \
  --mode batch \
  --batch-size 200 \
  --temperature 0.8 \
  --rejection-threshold 4 \
  --export-format parquet \
  --compression gzip \
  --output-dir custom_output
```

---

## Project Structure

```
syndgen/
 core/                  # Core data models and schemas
    schema.py          # Pydantic models for data validation
 export/                # Export functionality
    formats.py         # Multiple export format implementations
 pipeline/              # Main pipeline logic
    core.py            # Tri-stage pipeline implementation
 utils/                 # Utility functions
    helpers.py         # Helper functions and utilities
 cli.py                 # Command-line interface
 __init__.py            # Package initialization
 main.py                # Main entry point
 requirements.txt       # Dependencies
 .env                   # Environment configuration
 README.md              # Project documentation
```

---

## Architecture Overview

### 1. Seed Layer
- Processes input seeds or generates default prompts
- Ensures diversity through memory buffer tracking

### 2. Inference Layer (Generator)
- Uses DeepSeek R1 1.5B for scenario generation
- Captures Chain-of-Thought reasoning traces
- Generates final outputs with structured formatting

### 3. Audit Layer (Critic)
- Evaluates logical consistency
- Scores samples on 1-5 scale
- Provides detailed feedback
- Implements rejection sampling

---

## Data Quality Metrics

### Sample Evaluation Criteria
- **Logic Score**: 1-5 scale for logical consistency
- **Coherence Score**: 1-5 scale for reasoning quality
- **Validation Status**: Pass/fail based on threshold
- **Confidence Metrics**: Model confidence scores

### Pipeline Statistics
- **Rejection Rate**: Percentage of filtered samples
- **Generation Speed**: Samples per second
- **Diversity Score**: Semantic uniqueness metrics

---

## Development Roadmap

### Phase 1: Core CLI Tool (Complete)
- Basic Q&A pair generation
- Tri-stage pipeline implementation
- Multiple export formats

### Phase 2: Critic Loop Enhancement
- Advanced rejection sampling
- Multi-pass validation
- Enhanced quality metrics

### Phase 3: UI Integration
- Gradio/Streamlit interface
- Real-time monitoring dashboard
- Interactive generation controls

### Phase 4: Multi-Model Support
- Multiple LLM integration
- Model comparison features
- Hybrid generation approaches

---

## Performance Targets

| Metric | Target | Current Status |
|--------|--------|----------------|
| Samples/hour | 1,000+ | Simulation mode |
| Rejection Rate | 10-20% | Configurable |
| Generation Time | <5s/sample | Optimized |
| Diversity Score | >0.85 | Implemented |

---

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Commit your changes**: `git commit -m 'Add some feature'`
4. **Push to the branch**: `git push origin feature/your-feature`
5. **Open a pull request**

### Development Setup
```bash
# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest

# Format code
black .

# Lint code
flake8 .
```

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Support & Contact

For issues, questions, or suggestions:
- **GitHub Issues**: [https://github.com/stmarkadebayo/Syndgen/issues](https://github.com/stmarkadebayo/Syndgen/issues)
- **Email**: [your.email@example.com](mailto:your.email@example.com)

---

## Example Use Cases

### 1. Machine Learning Dataset Generation
```bash
# Generate 500 Q&A pairs for ML training
python main.py --mode export --batch-size 500 --export-format sft
```

### 2. Software Testing Data
```bash
# Generate diverse test cases with reasoning traces
python main.py --mode batch --batch-size 100 --include-reasoning
```

### 3. Research Data Collection
```bash
# Generate high-quality samples with strict validation
python main.py --mode batch --batch-size 200 --rejection-threshold 5
```

---

## Future Enhancements

- **Multi-LLM Support**: Integration with Llama, Mistral, etc.
- **Advanced Validation**: Semantic similarity checking
- **Cloud Integration**: Optional API-based generation
- **Custom Templates**: Domain-specific generation patterns
- **Memory Optimization**: Efficient batch processing

---

**Syndgen** - Empowering local, private, and high-quality synthetic data generation for the AI era! 
