import os
import sys
# Ensure your rag_system.py is updated to use the Google Generative AI SDK
from rag_system import PolicyRAGSystem

def print_banner():
    """Print welcome banner."""
    print("\n" + "="*80)
    print(" "*20 + "GEMINI POLICY Q&A ASSISTANT")
    print("="*80)
    print("\nAsk me anything about our Refund, Cancellation, or Shipping policies!")
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'sources' to see the last answer's sources.")
    print("="*80 + "\n")

def main():
    """Main interactive CLI function."""
    # Load Gemini API key
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        print("\nTo set your API key:")
        print("  Linux/Mac: export GOOGLE_API_KEY='your-key-here'")
        print("  Windows: set GOOGLE_API_KEY=your-key-here")
        print("\nGet your API key from: https://aistudio.google.com/")
        sys.exit(1)
    
    print("Initializing Gemini-powered Policy system...")
    
    try:
        # Pass the Google API key to your RAG system
        rag = PolicyRAGSystem(gemini_api_key=api_key, data_dir="data")
        rag.build_vector_store()
    except Exception as e:
        print(f"ERROR: Failed to initialize system: {e}")
        sys.exit(1)
    
    print_banner()
    last_result = None
    
    while True:
        try:
            question = input("Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nSession ended. Goodbye!")
                break
            
            if not question:
                continue
            
            if question.lower() == 'sources':
                if last_result:
                    print(f"\nSources: {', '.join(last_result['sources'])}")
                    print(f"Confidence: {last_result['confidence']}\n")
                else:
                    print("\nNo previous answer found.\n")
                continue
            
            print("\nConsulting Gemini and Policy Docs...\n")
            result = rag.answer_question(question)
            last_result = result
            
            print("="*80)
            print("GEMINI'S ANSWER:")
            print("="*80)
            print(result['answer'])
            print("\n" + "-"*80)
            print(f"Sources: {', '.join(result['sources']) if result['sources'] else 'None'}")
            print(f"Confidence: {result['confidence']}")
            print("="*80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")
            continue

if __name__ == "__main__":
    main()