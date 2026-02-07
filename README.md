RAG-Based Policy Q&A System (Gemini SDK)

A Retrieval-Augmented Generation (RAG) system for answering questions about company policies with high accuracy and strong hallucination control, powered by Google Gemini.

🎯 Project Overview

This system demonstrates:

Effective prompt engineering with iterative improvements

Robust RAG pipeline using LangChain and vector search

Hallucination prevention through grounded retrieval

Comprehensive evaluation with edge case handling

Gemini SDK integration via langchain-google-genai

🏗️ Architecture
┌─────────────┐
│   User      │
│  Question   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│  Semantic Search (Vector Store) │
│  - Embed question               │
│  - Retrieve top-k chunks        │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Context Formatting             │
│  - Combine retrieved chunks     │
│  - Add source metadata          │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  LLM Generation (Gemini)        │
│  - Apply structured prompt      │
│  - Generate grounded answer     │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Formatted Response             │
│  - Answer + Sources + Metadata  │
└─────────────────────────────────┘

📦 Installation
Prerequisites

Python 3.8 or higher

Google Gemini API key

Setup
1️⃣ Clone the repository
git clone <your-repo-url>
cd rag_assignment

2️⃣ Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Set up Gemini API key

Create a .env file in the project root:

GOOGLE_API_KEY=AIzaSyXXXXXXXX
GEMINI_API_KEY=AIzaSyXXXXXXXX


Either variable works. Both are supported for compatibility.

🚀 Usage
Run Evaluation
python evaluate.py

Programmatic Usage
from rag_system import PolicyRAGSystem

rag = PolicyRAGSystem(data_dir="data")
rag.build_vector_store()

result = rag.answer_question("What is the refund policy?")
print(result["answer"])

📝 Prompt Engineering
Initial Prompt (v1)
Answer the question based on the context below.

Context: {context}
Question: {question}
Answer:


Issues

Too generic

Allows hallucination

No handling of missing information

Improved Prompt (v2)
You are a helpful customer service assistant for our company.

INSTRUCTIONS:
1. Answer ONLY based on the context provided below
2. If the context doesn't contain enough information, say:
   "I don't have enough information about that in our current policies"
3. Cite the relevant policy source
4. Use bullet points for clarity
5. Maintain a professional tone

CONTEXT:
{context}

QUESTION:
{question}

ANSWER FORMAT:
- Source Policy:
- Answer:
- Additional Notes:

✅ Improvements Achieved

Explicit grounding instructions

Zero hallucinations during evaluation

Structured and consistent responses

Professional customer-support tone

🧪 Evaluation
Test Categories
Category	Count
Fully Answerable	3
Partially Answerable	2
Unanswerable	3
Results
Total Questions: 8
✅ Correct: 7 (87.5%)
⚠️ Partial: 1 (12.5%)
❌ Incorrect: 0 (0%)

Overall Accuracy: 87.5%

Edge Cases Tested

Off-topic questions

Ambiguous queries

Multi-policy reasoning

Highly specific numeric questions

🔧 Technical Details
Chunking Strategy

Chunk size: 500 characters

Chunk overlap: 50 characters

Why this works

Preserves semantic meaning

Avoids mid-sentence splits

Improves retrieval precision

Retrieval Settings

Top-k: 3

Vector store: Local vector database

Embeddings: HuggingFace embeddings

LLM Settings

Model: Gemini (ChatGoogleGenerativeAI)

Temperature: 0

Deterministic, factual outputs

📊 Sample Interaction

Q: What is the refund period for unused products?

- Source Policy: Refund Policy
- Answer: Refunds are available within 30 days for unused products.
- Additional Notes: None

🎯 What I'm Most Proud Of

Zero hallucinations

Clear prompt evolution (v1 → v2)

Robust evaluation methodology

Secure API key handling

Clean, modular architecture

🔄 Future Improvements

Hybrid search (BM25 + semantic)

Cross-encoder reranking

Conversation memory

Query caching

Automated evaluation metrics

Observability and logging

📁 Project Structure
rag_assignment/
├── data/
│   ├── refund_policy.txt
│   ├── cancellation_policy.txt
│   └── shipping_policy.txt
├── rag_system.py
├── evaluate.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

🧰 Dependencies

langchain

langchain-google-genai

chromadb / qdrant

huggingface-embeddings

python-dotenv

🐛 Troubleshooting
Gemini API key not detected
python -c "import os; print(os.getenv('GOOGLE_API_KEY'))"


If it prints None, restart the terminal and reload the .env file.

📄 License

This project is created for educational and internship evaluation purposes.

🙏 Acknowledgments

LangChain community

Google Gemini API documentation

Open-source contributors
