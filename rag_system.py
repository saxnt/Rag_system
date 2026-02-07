

import os
from typing import List, Dict, Optional
from pathlib import Path
import json
from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# Load environment variables



class PolicyRAGSystem:
    """
    RAG system for answering questions about company policies.
    Uses HuggingFace embeddings and Qdrant vector store.
    """
    
    def __init__(
        self, 
        gemini_api_key: str, 
        data_dir: str = "data", 
        chunk_size: int = 500, 
        chunk_overlap: int = 50,
        collection_name: str = "policy_collection",
        qdrant_path: str = "./qdrant_storage"
    ):
        """
        Initialize the RAG system.
        
        Args:
            gemini_api_key: Google Gemini API key
            data_dir: Directory containing policy documents
            chunk_size: Size of text chunks (default: 500)
            chunk_overlap: Overlap between chunks (default: 50)
            collection_name: Name for Qdrant collection
            qdrant_path: Path for Qdrant storage
        
        Chunking Strategy:
        - 500 chars chosen to capture complete policy sections
        - Balances context preservation with retrieval precision
        - Overlap of 50 ensures continuity between chunks
        """
        self.gemini_api_key = gemini_api_key
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.qdrant_path = qdrant_path
        
        # Initialize HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./embedding_cache"
        )
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=gemini_api_key,
            temperature=0  # Deterministic for consistent answers
        )
        
        # Initialize vector store
        self.vector_store = None
        self.documents = []
        
        # Prompt templates (v1 and v2)
        self.prompt_v1 = self._create_prompt_v1()
        self.prompt_v2 = self._create_prompt_v2()
        
        # Use improved prompt by default
        self.current_prompt = self.prompt_v2
        
    def _create_prompt_v1(self) -> PromptTemplate:
        """
        Initial prompt version - Basic instruction.
        """
        template = """Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_prompt_v2(self) -> PromptTemplate:
        """
        Improved prompt version - Better structure and hallucination control.
        
        Improvements:
        1. Clear instructions to only use provided context
        2. Explicit handling of insufficient information
        3. Structured output format with sections
        4. Citation requirement for grounding
        5. Tone guidance for professional responses
        """
        template = """You are a helpful customer service assistant for our company. Your job is to answer questions about our policies accurately and clearly.

INSTRUCTIONS:
1. Answer ONLY based on the context provided below
2. If the context doesn't contain enough information, say "I don't have enough information about that in our current policies"
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
"I don't have enough information about that in our current policies. Please contact our customer service team at support@company.com for assistance."
"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def load_documents(self) -> List[Document]:
        """
        Load and process policy documents from the data directory.
        
        Returns:
            List of LangChain Document objects
        """
        documents = []
        data_path = Path(self.data_dir)
        
        # Support .txt files
        for file_path in data_path.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "policy_type": file_path.stem.replace('_', ' ').title()
                }
            )
            documents.append(doc)
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for better retrieval.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # Hierarchical splitting
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        return chunks
    
    def build_vector_store(self):
        """
        Build the Qdrant vector store from documents.
        """
        # Load and chunk documents
        self.documents = self.load_documents()
        chunks = self.chunk_documents(self.documents)
        
        # Create Qdrant vector store
        self.vector_store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            path=self.qdrant_path  # Local storage
        )
        
        print("✓ Qdrant vector store built successfully")
    
    def load_existing_vector_store(self):
        """
        Load an existing Qdrant vector store.
        """
        try:
            self.vector_store = QdrantVectorStore.from_existing_collection(
                path=self.qdrant_path,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            print("✓ Loaded existing Qdrant vector store")
        except Exception as e:
            print(f"Error loading existing vector store: {e}")
            print("Building new vector store...")
            self.build_vector_store()
    
    def retrieve_context(self, question: str, k: int = 3) -> tuple[str, List[Document]]:
        """
        Retrieve relevant context for a question.
        
        Args:
            question: User question
            k: Number of top documents to retrieve
            
        Returns:
            Tuple of (formatted context string, list of retrieved documents)
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call build_vector_store() first.")
        
        # Semantic search
        retrieved_docs = self.vector_store.similarity_search(question, k=k)
        
        # Format context
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content
            context_parts.append(f"[Source: {source}]\n{content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        return context, retrieved_docs
    
    def answer_question(self, question: str, use_v1: bool = False) -> Dict:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question: User question
            use_v1: If True, use prompt v1; otherwise use v2
            
        Returns:
            Dictionary with answer, context, and metadata
        """
        # Retrieve context
        context, retrieved_docs = self.retrieve_context(question)
        
        # Handle case when no relevant documents found
        if not retrieved_docs or not context.strip():
            return {
                "question": question,
                "answer": "I don't have enough information about that in our current policies. Please contact our customer service team at support@company.com for assistance.",
                "context": "",
                "sources": [],
                "confidence": "low"
            }
        
        # Select prompt version
        prompt = self.prompt_v1 if use_v1 else self.prompt_v2
        
        prompt_text = prompt.format(context=context, question=question)
        response = self.llm.invoke(prompt_text)
        answer = response.content
        
        sources = [doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]
        unique_sources = list(set(sources))
        
        return {
            "question": question,
            "answer": answer,
            "context": context,
            "sources": unique_sources,
            "confidence": "high" if len(retrieved_docs) >= 2 else "medium"
        }
    
    def batch_answer(self, questions: List[str], use_v1: bool = False) -> List[Dict]:
        """
        Answer multiple questions in batch.
        
        Args:
            questions: List of questions
            use_v1: If True, use prompt v1
            
        Returns:
            List of answer dictionaries
        """
        results = []
        for question in questions:
            result = self.answer_question(question, use_v1=use_v1)
            results.append(result)
        
        return results


def main():
    """
    Main function to demonstrate the RAG system.
    """
    gemini_api_key = os.getenv("GOOGLE_API_KEY", "AIza........")
    
    if gemini_api_key == "your-gemini-api-key-here":
        print("WARNING: Using placeholder API key. Set GOOGLE_API_KEY environment variable.")
        print("Example: export GOOGLE_API_KEY='AIza...'")
        return
    
    # Initialize system
    print("Initializing RAG system with HuggingFace + Qdrant...")
    rag = PolicyRAGSystem(gemini_api_key=gemini_api_key, data_dir="data")
    
    # Build vector store
    print("\nBuilding Qdrant vector store...")
    rag.build_vector_store()
    
    # Example questions
    print("\n" + "="*80)
    print("DEMO: Answering Example Questions")
    print("="*80)
    
    questions = [
        "What is the refund policy for digital products?",
        "How much does express shipping cost?",
        "Can I cancel my order after it ships?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = rag.answer_question(question)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}")
        print("-" * 80)


if __name__ == "__main__":
    main()