# 🔬 ArXiv Research Expert— RAG System

A Retrieval-Augmented Generation (RAG) application that allows users to explore and ask questions about research papers from **arXiv** using natural language.

The system combines semantic search, metadata-based filtering, automatic filter extraction, and document reranking to retrieve relevant research papers and generate context-aware answers using locally running language models.

## 🖥️ Interface

### Main Interface



### Search and Filtering



## ✨ Features

- **Semantic Search:** Retrieves relevant research papers using vector embeddings.
- **Metadata Filtering:** Filters papers using metadata such as categories and publication information.
- **Automatic Filter Extraction:** An LLM interprets the user's question and extracts relevant metadata filters automatically.
- **Hybrid Retrieval Workflow:** Combines semantic retrieval with metadata-based filtering.
- **Document Reranking:** Reorders retrieved documents to improve relevance before generation.
- **Context-Aware Generation:** Generates answers grounded in the retrieved research papers.
- **Local LLM Inference:** Uses Ollama to run language models locally. (Streamlit UI)

## 🏗️ Architecture

The application follows a RAG workflow:

1. **Dataset Acquisition:** Downloads the arXiv dataset.
2. **Data Processing:** Filters and prepares the research-paper metadata and abstracts.
3. **Embedding Generation:** Generates document embeddings using BAAI's BGE embedding model.
4. **Vector Storage:** Stores embeddings and associated metadata in ChromaDB.
5. **Question Processing:** Receives the user's question and extracts relevant metadata filters using an LLM.
6. **Retrieval:** Searches the vector database using semantic similarity and applicable metadata filters.
7. **Reranking:** Reranks retrieved documents to prioritize the most relevant results.
8. **Answer Generation:** Uses the retrieved context to generate a response.

## 🧰 Technology Stack

| Component                  | Technology                       |
| -------------------------- | -------------------------------- |
| User Interface             | Streamlit                        |
| RAG Pipeline               | LangChain                        |
| Vector Database            | ChromaDB                         |
| Embedding Model            | `BAAI/bge-small-en-v1.5`         |
| Metadata Filter Extraction | `qwen2.5:1.5b`                   |
| Answer Generation          | `llama3.1:latest`                |
| Local Model Runtime        | Ollama                           |
| Dataset                    | Cornell University arXiv Dataset |

### Models

**Embedding Model**

`BAAI/bge-small-en-v1.5`

Used to convert research-paper content and user queries into vector representations for semantic retrieval.

**Metadata Extraction Model**

`qwen2.5:1.5b`

Used to interpret natural-language questions and extract relevant metadata filters.

**Generation Model**

`llama3.1:latest`

Used to generate answers based on the retrieved research-paper context. It runs locally through Ollama.

## 📚 Dataset

The project uses the [Cornell University arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv).

The dataset contains arXiv paper metadata and abstracts in a large JSON Lines file:

`arxiv-metadata-oai-snapshot.json`

The dataset is downloaded through `kagglehub` during the indexing process.

The notebook filters the original dataset and caches the processed records in:

`data/arxiv_filtered.jsonl`

**Note:** The original dataset and processed data are not included in this repository because of their size. They are downloaded and generated locally.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/arxiv-rag.git
cd arxiv-rag
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle Credentials

A free Kaggle account is required to download the dataset.

Configure Kaggle authentication using either:

- `~/.kaggle/kaggle.json`
- `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

See [Kaggle API documentation](https://www.kaggle.com/docs/api) for authentication instructions.

### 5. Install and Configure Ollama

Install [Ollama](https://ollama.com/) and download the required models:

```bash
ollama pull llama3.1
ollama pull qwen2.5:1.5b
```

Make sure Ollama is running locally before launching the application.

### 6. Build the Vector Store

Open and execute:

`arxiv_vector_store.ipynb`

The notebook handles dataset acquisition, preprocessing, embedding generation, and ChromaDB vector-store creation.

The initial indexing process may take some time depending on the dataset size and available hardware.

### 7. Run the Application

After creating the vector store:

```bash
streamlit run app.py
```

Open the local URL provided by Streamlit in your browser.

## 📁 Project Structure

```text
arxiv-rag/
│
├── app.py                      # Streamlit user interface
├── rag_pipeline.py             # Retrieval and generation pipeline
├── arxiv_vector_store.ipynb    # Dataset processing and vector indexing
│
├── assets/                     # Interface screenshots
│   ├── interface-main.png
│   └── interface-filters.png
│
├── data/                       # Locally generated dataset files
│
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── LICENSE
```

## ⚙️ Configuration and Requirements

- Python 3.10+
- Kaggle account and API credentials
- Ollama installed locally
- Sufficient disk space for the dataset and vector database
- Internet connection for downloading the dataset and embedding model

**Important:** The vector database is generated locally and is not included in the GitHub repository. Run the indexing notebook before launching the application.

## 🔒 Notes

- Kaggle credentials and API keys must never be committed to GitHub.
- The raw and processed datasets are excluded from version control.
- ChromaDB files are generated locally.
- Both language models run locally through Ollama.

## 👨‍💻 Author

**Nashat Alfarajat**

[nashatfr](https://github.com/nashatfr)  
