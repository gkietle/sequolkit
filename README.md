# SequolKit: Semantic Query Understanding and Optimization Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)

A research implementation for semantic database query understanding and optimization using Large Language Models (LLMs).

## 📚 Abstract

SequolKit is a novel research toolkit that leverages state-of-the-art Large Language Models to enhance database query understanding and optimization through semantic analysis. Our system introduces a unique approach to database interaction by combining semantic schema enrichment, natural language processing, and query optimization techniques. The toolkit demonstrates significant improvements in query understanding accuracy and optimization efficiency through its innovative use of LLMs.

## 🎯 Key Features

- **Semantic Schema Enrichment**: Advanced analysis of database schemas to provide rich semantic descriptions of tables and columns
- **Natural Language to SQL Translation**: State-of-the-art conversion of natural language queries to optimized SQL
- **Baseline Comparison**: Comprehensive evaluation framework for comparing semantic enhancements
- **Docker-based Deployment**: Easy deployment with containerized services

## 🏗️ Architecture

The system consists of three main components:

1. **Embedding Service**: Handles helper functions for database utilities
2. **Engine Service**: Manages query processing and LLM interactions
3. **Ollama Service**: Provides LLM inference capabilities

## 🚀 Quick Start

### Prerequisites

- Python 3.10
- Docker and Docker Compose
- NVIDIA GPU (recommended)
- CUDA Toolkit (for GPU support)

### Database Setup

1. Download the required database files from our [Google Drive](https://drive.google.com/drive/folders/1wc-k2cx8evGCcI7tvIejFbwxtRRWspO5?usp=sharing)
2. Extract the downloaded files into the `database` directory in the project root
3. For evaluation data, download the Spider dataset from [here](https://yale-lily.github.io/spider) and place it in `evaluation/spider_data/`
4. Ensure the database files are properly placed in the following structure:
   ```
   database/
   ├── schema.sql
   ├── data/
   │   └── [database files]
   └── README.md

   evaluation/
   └── spider_data/
       └── test_database/
           ├── baseball_1/
           ├── soccer_1/
           └── wta_1/
   ```

> **Note**: Large database files (>50MB) are not tracked in git. Please download them separately from the provided Google Drive link.

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sequolkit.git
cd sequolkit

# Create and activate conda environment
conda create -n sequolkit python=3.10 -y
conda activate sequolkit

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt')"

# Start services using Docker Compose
docker compose up --build -d

# Pull required LLM models
docker exec -it sequolkit-ai-ollama ollama pull qwen2.5-coder
```

### Configuration

Create a `.env` file in the root directory:

```env
OLLAMA_MODEL=qwen2.5-coder:14b
OLLAMA_HOST=http://localhost:9292
EMBED_HOST_API=http://localhost:9191
SERVICE_PORT=5000
```

## 🔬 Research Components

### Schema Enrichment

Our schema enrichment component employs advanced LLM techniques to analyze and enhance database schemas with semantic descriptions, improving query understanding accuracy.

### SQL Agent Workflow

The SQL Agent implements a sophisticated natural language to SQL translation pipeline with semantic understanding capabilities.

### Baseline Workflow

A comprehensive evaluation framework for comparing the effectiveness of semantic enhancements against traditional approaches.

## 📊 Evaluation and Inference

### Evaluation Framework

SequolKit uses a comprehensive evaluation framework based on the Spider dataset methodology. The evaluation includes:

1. **Component Matching**: Evaluates individual SQL components (SELECT, FROM, WHERE, etc.)
2. **Exact Matching**: Compares complete SQL queries for exact matches
3. **Execution Accuracy**: Tests query execution results against ground truth

### Running Evaluations

1. **Setup Evaluation Environment**:
   ```bash
   cd evaluation
   pip install -r requirements.txt
   ```

2. **Prepare Test Data**:
   - Place your test queries in `evaluation/evaluation_examples/`
   - Ensure database files are in the correct location
   - Update `tables.json` with your schema information

3. **Run Evaluation**:
   ```bash
   python evaluation.py --gold [gold_file] --pred [predicted_file] --etype [evaluation_type] --db [database_dir] --table [table_file]
   ```

   Parameters:
   - `gold_file`: File containing ground truth SQL queries
   - `predicted_file`: Your model's predicted SQL queries
   - `evaluation_type`: "match" for exact matching, "exec" for execution accuracy, or "all" for both
   - `database_dir`: Directory containing SQLite databases
   - `table_file`: JSON file with database schema information

### Inference Process

1. **Model Inference**:
   ```bash
   python inferences_testing.py --input [input_file] --output [output_file] --model [model_name]
   ```

2. **Output Format**:
   - Each prediction should be a valid SQL query
   - Include database ID with each prediction
   - Follow the format: `[SQL_QUERY]\t[DB_ID]`

3. **Evaluation Metrics**:
   - Exact Match Accuracy
   - Component-wise Accuracy
   - Execution Accuracy
   - Hardness-based Analysis

### Best Practices

1. **Preprocessing**:
   - Normalize table and column names
   - Handle special characters and SQL keywords
   - Validate query syntax before evaluation

2. **Evaluation**:
   - Use both exact matching and execution accuracy
   - Consider query complexity in analysis
   - Document any special cases or limitations

3. **Results Analysis**:
   - Analyze performance by query complexity
   - Track component-wise accuracy
   - Compare against baseline models

For detailed evaluation metrics and examples, refer to the [evaluation documentation](evaluation/README.md).

## 📊 API Endpoints

- `POST /query`: Natural language to SQL translation
- `POST /query-baseline`: Baseline comparison endpoint
- `POST /schema-enrichment`: Schema semantic enrichment
- `GET /settings`: View current LLM settings
- `POST /settings`: Update LLM settings
- `GET /health-check`: Service health check

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{...}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LlamaIndex](https://github.com/run-llama/llama_index)
- Uses [Ollama](https://github.com/ollama/ollama) for LLM inference
- Inspired by recent advances in LLM-based database systems

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📧 Contact

For questions and feedback, please open an issue or contact the maintainers. 