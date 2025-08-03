# 📄 Document Portal

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-🦜🔗-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-🎈-red.svg)

*A comprehensive AI-powered document processing and analysis platform*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-reference) • [Contributing](#-contributing)

</div>

---

## 🚀 Overview

Document Portal is an intelligent document processing platform that leverages state-of-the-art AI models to analyze, compare, and extract insights from PDF documents. Built with LangChain and powered by multiple LLM providers, it offers both web interface and programmatic access for document intelligence tasks.

### ✨ Key Capabilities

- 🔍 **Document Analysis** - Extract metadata, summaries, and structured insights
- 📊 **Document Comparison** - Page-by-page diff analysis between PDF versions  
- 💬 **Document Chat** - Interactive Q&A with single or multiple documents
- 🎯 **Smart Retrieval** - Advanced RAG with MMR and contextual compression
- 📈 **Evaluation** - Built-in metrics for retrieval and generation quality
- 🔧 **Enterprise Ready** - Robust logging, error handling, and configuration

## 🎯 Features

### Core Modules

| Module | Description | Key Features |
|--------|-------------|--------------|
| **Document Analyzer** | Intelligent document processing | Metadata extraction, summarization, structured output |
| **Document Compare** | Version comparison tool | Page-wise diff, change detection, visual comparison |
| **Single Document Chat** | Interactive document Q&A | Context-aware responses, citation tracking |
| **Multi Document Chat** | Cross-document intelligence | Multi-source reasoning, document synthesis |

### Advanced Retrieval

- **🎯 MMR (Maximal Marginal Relevance)** - Optimized relevance vs diversity
- **🗜️ Contextual Compression** - Intelligent context filtering  
- **📊 Evaluation Metrics** - Automated quality assessment
- **🔍 Hybrid Search** - Semantic + keyword matching

### Infrastructure

- **🏗️ Singleton Logger** - Centralized structured logging with JSON output
- **⚡ Custom Exceptions** - Comprehensive error handling with severity levels
- **🔧 Configuration Management** - YAML-based settings with environment support
- **🎨 Web Interface** - Streamlit-powered user-friendly UI

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- OpenAI API Key or Groq API Key
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/AutoQAce/document_loader.git
cd document_portal

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.copy .env
# Edit .env with your API keys

# Install in development mode
pip install -e .

# Run the application
streamlit run streamlit_ui.py
```

### Environment Setup

Create a `.env` file with your API credentials:

```env
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here
DATA_STORAGE_PATH=./data
```

## 📖 Usage

### Web Interface

Launch the Streamlit application for an intuitive web experience:

```bash
streamlit run streamlit_ui.py
```

Navigate to `http://localhost:8501` to access:
- Document upload and analysis
- Interactive comparison tools  
- Chat interfaces
- Real-time processing status

### Programmatic Usage

#### Document Analysis

```python
from src.document_analyzer.data_ingestion import DocumentHandler
from src.document_analyzer.data_analysis import DocumentAnalyzer

# Initialize components
handler = DocumentHandler()
analyzer = DocumentAnalyzer()

# Process document
with open("document.pdf", "rb") as f:
    file_path = handler.save_pdf(f)
    text = handler.read_pdf(file_path)
    
# Extract insights
metadata = analyzer.analyze_metadata(text)
print(f"Document Type: {metadata['document_type']}")
print(f"Summary: {metadata['summary']}")
```

#### Document Comparison

```python
from src.document_compare.data_ingestion import DocumentIngestion
from src.document_compare.data_comparator import DocumentComparator

# Setup comparison
ingestion = DocumentIngestion()
comparator = DocumentComparator()

# Process files
ref_path, act_path = ingestion.save_uploaded_files(ref_file, actual_file)
combined_text = ingestion.combine_documents()

# Generate comparison
comparison = comparator.compare_documents(combined_text)
```

#### Document Chat

```python
from src.single_document_chat.data_ingestion import SingleDocumentProcessor
from src.single_document_chat.retrieval import DocumentRetriever

# Initialize chat system
processor = SingleDocumentProcessor()
retriever = DocumentRetriever()

# Setup knowledge base
processor.ingest_document("document.pdf")
vectorstore = processor.get_vectorstore()

# Query document
response = retriever.query("What are the key findings?", vectorstore)
```

## 🏗️ Project Structure

```
document_portal/
├── 📁 src/                          # Core modules
│   ├── 📁 document_analyzer/        # Document analysis & metadata extraction
│   ├── 📁 document_compare/         # Version comparison tools
│   ├── 📁 single_document_chat/     # Single doc Q&A system
│   └── 📁 multi_document_chat/      # Multi-doc intelligence
├── 📁 config/                       # Configuration files
├── 📁 logger/                       # Structured logging system  
├── 📁 exception/                    # Custom exception handling
├── 📁 utils/                        # Utility functions
├── 📁 prompt/                       # LLM prompt templates
├── 📁 model/                        # Data models & schemas
├── 📁 static/                       # Web assets
├── 📁 templates/                    # HTML templates
└── 📁 data/                         # Document storage
```

## ⚙️ Configuration

The application uses YAML-based configuration in `config/config.yaml`:

```yaml
# LLM Configuration
llm:
  openai:
    provider: "openai"
    model_name: "gpt-4o-mini"
    temperature: 0.0
    max_tokens: 2048
  groq:
    provider: "groq"
    model_name: "deepseek-r1-distill-llama-70b"
    temperature: 0
    max_output_tokens: 2048

# Embedding Configuration  
embedding_model:
  provider: "openai"
  model_name: "text-embedding-3-small"

# Retrieval Settings
retriever:
  top_k: 5
  similarity_threshold: 0.7

# Vector Database
faiss_db:
  collection_name: "document_portal"
```

## 🔧 API Reference

### Core Classes

#### DocumentHandler
- `save_pdf(uploaded_file)` - Save uploaded PDF with session management
- `read_pdf(pdf_path)` - Extract text with page numbering
- Session-based organization for multi-user environments

#### DocumentAnalyzer  
- `analyze_metadata(document_text)` - Extract structured metadata
- Supports custom Pydantic models for output schemas
- Built-in error recovery with OutputFixingParser

#### DocumentComparator
- `compare_documents(combined_text)` - Generate page-wise comparisons
- Intelligent diff detection with change highlighting
- Support for encrypted PDF handling

### Exception Handling

```python
from exception.custom_exception import DocumentPortalException, ExceptionSeverity

# Raise exceptions with context
raise DocumentPortalException(
    "Processing failed", 
    original_exception, 
    ExceptionSeverity.HIGH,
    context={"file_size": 1024, "user_id": "123"}
)
```

### Logging

```python
from logger.custom_logger import CustomLogger

# Singleton logger with structured output
log = CustomLogger().get_logger(__name__)
log.info("Processing started", file="document.pdf", user="john")
```

## 🔍 Advanced Features

### Retrieval Strategies

- **Basic Retrieval**: Standard similarity search
- **MMR**: Balances relevance and diversity
- **Contextual Compression**: Filters irrelevant context
- **Hybrid**: Combines multiple strategies

### Evaluation Metrics

- **Retrieval Metrics**: Precision, Recall, F1-Score  
- **Generation Metrics**: BLEU, ROUGE, Semantic Similarity
- **Custom Metrics**: Domain-specific evaluation

### Multi-Model Support

- **OpenAI**: GPT-4, GPT-3.5-turbo, text-embedding-3
- **Groq**: Llama, Mixtral, DeepSeek models
- **Extensible**: Easy integration of new providers

## 🔑 API Keys & Setup

### Required API Keys

- **[Groq API Key](https://console.groq.com/keys)** (Free) - [Documentation](https://console.groq.com/docs/overview)
- **OpenAI API Key** (Paid) - For GPT models and embeddings
- **[Gemini API Key](https://aistudio.google.com/apikey)** (15 Days Free) - [Documentation](https://ai.google.dev/gemini-api/docs/models)

### Supported Models

#### LLM Models
- **Groq** (Free) - Llama, Mixtral, DeepSeek
- **OpenAI** (Paid) - GPT-4, GPT-3.5-turbo
- **Gemini** (Free Trial) - Gemini Pro
- **Claude** (Paid) - Claude 3
- **Hugging Face** (Free) - Open source models
- **Ollama** (Local) - Local deployment

#### Embedding Models
- **OpenAI** - text-embedding-3-small/large
- **Hugging Face** - sentence-transformers
- **Gemini** - text-embedding-gecko

#### Vector Databases
- **FAISS** - In-memory and on-disk storage
- **Chroma** - Local persistent storage
- **Pinecone** - Cloud-based (optional)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/document_portal.git

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/

# Run linting
flake8 src/
black src/
```

### Code Standards

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Add docstrings for public methods
- Write unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📖 [Documentation](https://github.com/AutoQAce/document_loader/wiki)
- 🐛 [Issue Tracker](https://github.com/AutoQAce/document_loader/issues)  
- 💬 [Discussions](https://github.com/AutoQAce/document_loader/discussions)
- 📧 [Email Support](mailto:support@documentportal.com)

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [Streamlit](https://streamlit.io/) for the web interface
- [FAISS](https://faiss.ai/) for vector similarity search
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing

---

<div align="center">

**[⭐ Star this repository](https://github.com/AutoQAce/document_loader)** if you find it helpful!

Made with ❤️ by the Document Portal Team

</div>
