# Multimodal RAG and Vision LLMs

## 1. Project Overview

Build a Multimodal RAG pipeline that processes PDF documents, images, and videos to retrieve relevant information and generate answers using Gemini Vision.

The implementation uses LangChain, Hugging Face Embeddings, and FAISS.

**Pipeline:** 

PDF, Images, and Videos → Processing → Textual Representations → Embeddings → FAISS → Retrieval → Multimodal Context → Gemini Vision → Answer

## 2. Project Structure

```text
multimodal-rag/
├── data/
│   ├── genai_engineer_roadmap.pdf
│   ├── rag_architecture.png
│   └── rag_chunking_demo.mp4
│
├── outputs/
│   └── execution_output.txt
│
├── .env.example
├── .gitignore
├── requirements.txt
└── multimodal_rag.py
```

- `data/` – Contains the input PDF, image, and video.
- `outputs/` – Contains a sample execution output.
- `requirements.txt` – Lists the required Python packages.
- `multimodal_rag.py` – Contains the complete implementation.

The application automatically creates folders for extracted video frames and audio during execution.

## 3. Prerequisites

Before starting, make sure you have:

- Python 3.12
- Visual Studio Code
- Git
- A Google Gemini API key

## 4. Setup and Installation

**Step 1: Clone the repository**

```bash
git clone https://github.com/chandu-123-git/multimodal-rag.git
cd multimodal-rag
```

**Step 2: Create and activate a virtual environment**

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

**Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4: Configure the Gemini API key**

Create a `.env` file in the project root and add:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Replace the placeholder with your actual Gemini API key.

Keep your `.env` file private. Never upload your API key to GitHub.

**Step 5: Verify the input files**

Make sure the following files are available inside the `data/` directory:

- `genai_engineer_roadmap.pdf`
- `rag_architecture.png`
- `rag_chunking_demo.mp4`

## 5. Run the Application

Execute the following command from the project root:

```bash
python multimodal_rag.py
```

The application runs the complete pipeline and displays intermediate results and the final answer in the terminal.

A sample execution output is available here:

[`outputs/execution_output.txt`](outputs/execution_output.txt)

### Sample Execution Results

The recorded execution produced the following results:

| Result | Value |
|---|---:|
| PDF pages | 13 |
| PDF chunks | 37 |
| Image descriptions | 1 |
| Video frames extracted | 10 |
| Transcript segments | 2 |
| Total documents | 50 |
| Embedding dimensions | 384 |
| FAISS vectors stored | 50 |
| Retrieved documents | 5 |

These results correspond to the supplied input files. Retrieval results and generated responses may vary between runs.

## 6. Experiment with Your Own Queries

Open `multimodal_rag.py` and locate the sample query in the main execution block:

```python
query = "What are the different chunking strategies in RAG?"
```

Replace it with your own question. For example:

```python
query = "Explain the architecture shown in the RAG diagram."
```

Save the file and run the application again:

```bash
python multimodal_rag.py
```

Observe the retrieved documents, their modalities, similarity scores, and the final answer.

**Important Notes:**

- Textual representations are used for embedding and retrieval.
- Original images and video frames are supplied to Gemini Vision when their corresponding visual documents are retrieved.
- Video audio is transcribed and indexed as searchable text with timestamps.
- The FAISS index is created in memory during execution and is not saved to disk.
- Gemini API usage is subject to applicable quotas and pricing.
- If you encounter an installation or execution error, check your Python environment, dependencies, API key, and input file paths.

**You have now built a Multimodal RAG pipeline that retrieves information from different modalities and uses textual and visual context to generate answers.**
