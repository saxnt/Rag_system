# RAG-Based Policy Q&A System

A Retrieval-Augmented Generation (RAG) system for answering questions about company policies with high accuracy and hallucination control.

## 🎯 Project Overview

This system demonstrates:
- **Effective prompt engineering** with iterative improvements
- **Robust RAG pipeline** using LangChain, Qdrant, and HuggingFace
- **Hallucination prevention** through grounded retrieval
- **Comprehensive evaluation** with edge case handling
- **Free and open-source stack** (Gemini API + HuggingFace embeddings + Qdrant)

## 🏗️ Architecture

```
┌─────────────┐
│   User      │
│  Question   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│  Semantic Search (Qdrant)       │
│  - Embed question (HuggingFace) │
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
│  LLM Generation (Gemini Flash)  │
│  - Apply structured prompt      │
│  - Generate grounded answer     │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Formatted Response             │
│  - Answer + Sources + Metadata  │
└─────────────────────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- Google Gemini API key (free tier available)

### Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd rag_assignment
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up API key**
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Google Gemini API key
# Or export directly:
export GOOGLE_API_KEY='AIza...'
```

Get your free Gemini API key from: https://aistudio.google.com/

## 🚀 Usage

### Interactive CLI
```bash
python cli.py
```

Example interaction:
```
Your question: What is the refund policy for unused products?

GEMINI'S ANSWER:
- Source Policy: Refund Policy
- Answer: Full refunds are available within 30 days of purchase for unused 
  products in their original packaging.
- Additional Notes: None

Sources: refund_policy.txt
Confidence: high
```

### Run Evaluation
```bash
python evaluate.py
```

### Programmatic Usage
```python
from rag_system import PolicyRAGSystem

# Initialize
rag = PolicyRAGSystem(
    gemini_api_key="your-key", 
    data_dir="data"
)
rag.build_vector_store()

# Ask questions
result = rag.answer_question("What is the refund policy?")
print(result['answer'])
print(f"Sources: {result['sources']}")
```

## 📝 Prompt Engineering

### Initial Prompt (v1)
```
Answer the question based on the context below.

Context: {context}
Question: {question}
Answer:
```

**Issues:**
- Too generic, allows hallucination
- No guidance on missing information
- Unstructured output

### Improved Prompt (v2)
```
You are a helpful customer service assistant for our company. 
Your job is to answer questions about our policies accurately and clearly.

INSTRUCTIONS:
1. Answer ONLY based on the context provided below
2. If the context doesn't contain enough information, say "I don't have enough 
   information about that in our current policies"
3. Be specific and cite which policy the information comes from
4. Use bullet points for clarity when listing multiple items
5. Be professional and helpful in tone

CONTEXT:
{context}

QUESTION: {question}

ANSWER (following the format below):
- Source Policy: [Name of the relevant policy]
- Answer: [Your detailed answer based only on the context]
- Additional Notes: [Any relevant caveats or related information, or "None"]

If you cannot answer from the context, respond with:
"I don't have enough information about that in our current policies. Please 
contact our customer service team at support@company.com for assistance."
```

**Improvements:**
1. ✅ **Explicit grounding instruction** - "Answer ONLY based on the context"
2. ✅ **Hallucination control** - Clear guidance on missing information
3. ✅ **Structured output** - Required format with source citation
4. ✅ **Professional tone** - Customer service context
5. ✅ **Clarity enhancement** - Bullet point guidance

**Results:** Improved accuracy from ~62% to ~87% in evaluation.

## 🧪 Evaluation

### Test Set (8 Questions)

| Category | Count | Description |
|----------|-------|-------------|
| Fully Answerable | 3 | Information clearly in documents |
| Partially Answerable | 2 | Some info available, some missing |
| Unanswerable | 3 | Information not in documents |

### Scoring Rubric
- ✅ **Correct** - Accurate answer, properly grounded
- ⚠️ **Partial** - Has issues but usable
- ❌ **Incorrect** - Wrong or hallucinated

### Results (Prompt v2)

```
Total Questions: 8
✅ Correct: 7 (87.5%)
⚠️ Partial: 1 (12.5%)
❌ Incorrect: 0 (0.0%)

Overall Accuracy: 87.5%
```

**Key Findings:**
- **Zero hallucinations** - System correctly acknowledges missing information
- **Strong grounding** - All answers cite sources
- **Good edge case handling** - Appropriately handles ambiguous questions

### Edge Cases Tested

1. **Off-topic questions** - Correctly refuses with helpful message
2. **Ambiguous questions** - Requests clarification
3. **Multi-policy questions** - Synthesizes from multiple sources
4. **Specific numeric queries** - Provides exact information when available

## 🔧 Technical Details

### Chunking Strategy

**Configuration:**
- Chunk size: 500 characters
- Chunk overlap: 50 characters

**Rationale:**
- **500 chars** captures complete policy sections without fragmenting
- **50 char overlap** ensures context continuity across chunks
- Prevents splitting mid-sentence or mid-concept
- Balances retrieval precision vs. context preservation

**Alternatives Considered:**
- 200 chars: Too small, fragments context
- 1000 chars: Too large, reduces retrieval precision

### Retrieval Settings

- **Top-k**: 3 documents
- **Embedding model**: sentence-transformers/all-MiniLM-L6-v2 (HuggingFace)
- **Vector store**: Qdrant (local storage, no external service needed)

### LLM Settings

- **Model**: Gemini 2.5 Flash
- **Temperature**: 0 (deterministic answers)
- **Provider**: Google Generative AI

## 📊 Sample Interactions

**Example 1: Fully Answerable**
```
Q: What is the refund period for unused products?
A: Source Policy: Refund Policy
   Answer: Full refunds are available within 30 days of purchase for unused 
   products in their original packaging. The product must be in the same 
   condition as when you received it, with all tags attached.
   Additional Notes: None
✅ Sources: refund_policy.txt
```

**Example 2: Unanswerable**
```
Q: What is your privacy policy regarding customer data?
A: I don't have enough information about that in our current policies. 
   Please contact our customer service team at support@company.com for assistance.
✅ Sources: None (correctly identified as missing)
```

**Example 3: Multi-source**
```
Q: If I cancel my order after shipping, can I get a refund?
A: Source Policy: Cancellation Policy, Refund Policy
   Answer: Once an order has shipped, it cannot be cancelled. However, you can 
   return the item following our refund policy, which allows full refunds within 
   30 days for unused products. You would need to return the item in original 
   packaging.
   Additional Notes: Original shipping costs are non-refundable, and return 
   shipping is the customer's responsibility unless the item is defective.
✅ Sources: cancellation_policy.txt, refund_policy.txt
```

## 🎯 What I'm Most Proud Of

1. **Hallucination Control** - Zero hallucinations in evaluation through explicit prompt constraints
2. **Structured Prompt Design** - Clear improvement from v1 to v2 with measurable impact
3. **Comprehensive Evaluation** - Thoughtful test cases covering edge cases
4. **Clean Code Architecture** - Modular, well-documented, and extensible
5. **Free Tech Stack** - No paid services required (free Gemini API + HuggingFace + local Qdrant)

## 🔄 Future Improvements

Given more time, I would implement:

1. **Reranking** - Cross-encoder reranking for better retrieval precision
2. **Hybrid Search** - Combine semantic + keyword search (BM25)
3. **Query Expansion** - Rephrase user questions for better retrieval
4. **Conversation Memory** - Track context across multi-turn conversations
5. **Advanced Evaluation** - Automated metrics (BLEU, ROUGE, semantic similarity)
6. **Logging & Monitoring** - LangSmith or similar for production tracing
7. **Caching** - Cache frequent queries for faster responses
8. **Fine-tuning** - Fine-tune embeddings on policy-specific data
9. **Async Processing** - Asynchronous document processing for larger datasets
10. **Web Interface** - Streamlit or Gradio UI for easier access

### Trade-offs Made

| Decision | Chosen | Alternative | Rationale |
|----------|--------|-------------|-----------|
| Vector Store | Qdrant | Pinecone/Weaviate | Local storage, no external service needed |
| Embeddings | HuggingFace | OpenAI | Free, good quality, runs locally |
| LLM Model | Gemini 2.5 Flash | GPT-4/Claude | Free tier, fast, sufficient quality |
| Chunk Size | 500 chars | 1000 chars | Better retrieval precision |
| Evaluation | Manual scoring | Automated metrics | Better captures nuanced quality |
| Prompt Format | Structured | Conversational | Ensures consistent output |

## 📁 Project Structure

```
rag_assignment/
├── data/                      # Policy documents
│   ├── refund_policy.txt
│   ├── cancellation_policy.txt
│   └── shipping_policy.txt
├── qdrant_storage/            # Local Qdrant vector database (auto-generated)
├── embedding_cache/           # HuggingFace model cache (auto-generated)
├── rag_system.py             # Core RAG implementation
├── evaluate.py               # Evaluation script
├── cli.py                    # Interactive CLI
├── requirements.txt          # Python dependencies
├── .env.example             # Environment template
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## 🧰 Dependencies

Core libraries:
- **langchain**: RAG orchestration framework
- **langchain-qdrant**: Qdrant vector store integration
- **langchain-huggingface**: HuggingFace embeddings integration
- **langchain-google-genai**: Google Gemini LLM integration
- **qdrant-client**: Vector database client
- **sentence-transformers**: Embedding models
- **python-dotenv**: Environment variable management

## 🐛 Troubleshooting

**Error: "GOOGLE_API_KEY not set"**
```bash
export GOOGLE_API_KEY='AIza...'
```
Get your free key from: https://aistudio.google.com/

**Error: "No module named 'langchain'"**
```bash
pip install -r requirements.txt
```

**Qdrant storage errors**
- Delete the `qdrant_storage/` directory and rebuild vector store
- Ensure sufficient disk space

**Slow first run**
- HuggingFace models download on first use (cached afterward)
- First embedding generation takes longer
- Subsequent runs use cached models and are much faster

**Out of memory errors**
- Reduce chunk_size in PolicyRAGSystem initialization
- Reduce number of documents processed at once
- Use a smaller embedding model

## 🔑 API Keys and Costs

### Gemini API
- **Free tier**: 15 requests per minute, 1500 requests per day
- **Cost**: Free for development and light usage
- **Get key**: https://aistudio.google.com/

This system uses the free tier and should work well for development and testing.

## 📄 Example Environment File

Create a `.env` file in the project root:

```env
# Google Gemini API Key
GOOGLE_API_KEY=AIza...your-key-here
```

Or use `.env.example` as a template:
```bash
cp .env.example .env
# Edit .env and add your API key
```

## 🧪 Running Tests

Run the evaluation suite:
```bash
python evaluate.py
```

This will:
1. Initialize the RAG system
2. Build the vector store from policy documents
3. Run 8 test questions across different categories
4. Test edge cases
5. Output detailed results and save to `evaluation_results.json`

Compare prompt versions:
```python
# In evaluate.py, uncomment the line:
evaluator.compare_prompts()
```

## 🎨 Customization

### Adding New Policies

1. Add `.txt` files to the `data/` directory
2. Rebuild the vector store:
```python
rag = PolicyRAGSystem(gemini_api_key=api_key, data_dir="data")
rag.build_vector_store()
```

### Adjusting Chunk Size

```python
rag = PolicyRAGSystem(
    gemini_api_key=api_key,
    chunk_size=1000,  # Larger chunks
    chunk_overlap=100  # More overlap
)
```

### Using Different Models

```python
# In rag_system.py, modify the __init__ method:

# For embeddings:
self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"  # Better quality
)

# For LLM:
self.llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # More capable model
    temperature=0.3  # Slightly more creative
)
```

### Changing Retrieval Settings

```python
# Retrieve more context documents
context, docs = rag.retrieve_context(question, k=5)
```

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [HuggingFace Sentence Transformers](https://www.sbert.net/)
- [Google Gemini API](https://ai.google.dev/)

## 📝 License

This project is created for educational purposes as part of a take-home assignment.

## 🙏 Acknowledgments

- LangChain community for excellent RAG framework
- Qdrant team for powerful vector database
- HuggingFace for open-source embeddings
- Google for free Gemini API access

---

**Note**: This system uses completely free and open-source technologies, making it accessible for development, learning, and small-scale deployment without any costs.
