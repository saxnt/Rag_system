"""
Evaluation script for the RAG system.
Tests the system with various question types and evaluates performance.
Updated to use Gemini SDK.
"""

import os
from rag_system import PolicyRAGSystem
from typing import List, Dict
import json


class RAGEvaluator:
    """
    Evaluates the RAG system on various question types.
    """
    
    def __init__(self, rag_system: PolicyRAGSystem):
        self.rag_system = rag_system
        self.evaluation_set = self._create_evaluation_set()
    
    def _create_evaluation_set(self) -> List[Dict]:
        """
        Create evaluation questions with expected characteristics.
        
        Categories:
        1. Fully answerable - Information is clearly in the documents
        2. Partially answerable - Some info available, some missing
        3. Unanswerable - Information not in documents
        """
        
        return [
            # FULLY ANSWERABLE QUESTIONS
            {
                "question": "What is the refund period for unused products?",
                "category": "fully_answerable",
                "expected_info": "30 days",
                "expected_source": "refund_policy.txt"
            },
            {
                "question": "How much does standard shipping cost for orders over $100?",
                "category": "fully_answerable",
                "expected_info": "FREE",
                "expected_source": "shipping_policy.txt"
            },
            {
                "question": "Can I cancel my order within 2 hours of placement?",
                "category": "fully_answerable",
                "expected_info": "Yes, free of charge",
                "expected_source": "cancellation_policy.txt"
            },
            
            # PARTIALLY ANSWERABLE QUESTIONS
            {
                "question": "What happens if I want to return a product after 45 days?",
                "category": "partially_answerable",
                "expected_info": "Policy states 30 days for refunds, unclear about after that",
                "expected_source": "refund_policy.txt"
            },
            {
                "question": "Do you ship to Australia?",
                "category": "partially_answerable",
                "expected_info": "Policy mentions international but not Australia specifically",
                "expected_source": "shipping_policy.txt"
            },
            
            # UNANSWERABLE QUESTIONS
            {
                "question": "What is your privacy policy regarding customer data?",
                "category": "unanswerable",
                "expected_info": "Not in provided documents",
                "expected_source": "None"
            },
            {
                "question": "Do you offer price matching?",
                "category": "unanswerable",
                "expected_info": "Not mentioned in policies",
                "expected_source": "None"
            },
            {
                "question": "What are your business hours?",
                "category": "unanswerable",
                "expected_info": "Only phone support hours mentioned (9 AM - 6 PM EST)",
                "expected_source": "cancellation_policy.txt"
            }
        ]
    
    def evaluate_answer(self, question_data: Dict, answer_data: Dict) -> Dict:
        """
        Evaluate a single answer.
        
        Scoring:
        ✅ Correct - Answer is accurate and appropriately grounded
        ⚠️ Partial - Answer has some issues but usable
        ❌ Incorrect - Answer is wrong or hallucinated
        """
        
        question = question_data["question"]
        category = question_data["category"]
        answer = answer_data["answer"].lower()
        sources = answer_data["sources"]
        
        # Evaluation criteria
        has_hallucination = False
        is_grounded = len(sources) > 0
        acknowledges_limitation = "don't have" in answer or "not available" in answer or "contact" in answer
        
        # Scoring logic based on category
        if category == "fully_answerable":
            # Should provide specific answer with sources
            if is_grounded and not acknowledges_limitation:
                score = "✅"
                notes = "Correctly answered with proper grounding"
            elif acknowledges_limitation:
                score = "⚠️"
                notes = "Overly cautious - information was available"
            else:
                score = "❌"
                notes = "Failed to provide available information"
        
        elif category == "partially_answerable":
            # Should provide partial info OR acknowledge limitation
            if acknowledges_limitation or (is_grounded and "may" in answer):
                score = "✅"
                notes = "Appropriately handled partial information"
            else:
                score = "⚠️"
                notes = "Could better indicate uncertainty"
        
        elif category == "unanswerable":
            # Should acknowledge lack of information
            if acknowledges_limitation:
                score = "✅"
                notes = "Correctly indicated information not available"
            elif is_grounded:
                score = "⚠️"
                notes = "Attempted answer but should acknowledge limitation"
            else:
                score = "❌"
                notes = "Failed to acknowledge missing information"
        
        return {
            "question": question,
            "category": category,
            "score": score,
            "answer": answer_data["answer"],
            "sources": sources,
            "notes": notes,
            "confidence": answer_data.get("confidence", "unknown")
        }
    
    def run_evaluation(self, use_v1: bool = False) -> Dict:
        """
        Run full evaluation on the test set.
        
        Args:
            use_v1: If True, use prompt v1; otherwise use v2
            
        Returns:
            Evaluation results with scores and analysis
        """
        print(f"\n{'='*80}")
        print(f"RUNNING EVALUATION - Prompt Version: {'v1' if use_v1 else 'v2'}")
        print(f"{'='*80}\n")
        
        results = []
        scores = {"✅": 0, "⚠️": 0, "❌": 0}
        
        for question_data in self.evaluation_set:
            # Get answer from RAG system
            answer_data = self.rag_system.answer_question(
                question_data["question"],
                use_v1=use_v1
            )
            
            # Evaluate the answer
            eval_result = self.evaluate_answer(question_data, answer_data)
            results.append(eval_result)
            
            # Update score counts
            scores[eval_result["score"]] += 1
            
            # Print result
            print(f"Question: {eval_result['question']}")
            print(f"Category: {eval_result['category']}")
            print(f"Score: {eval_result['score']}")
            print(f"Answer: {eval_result['answer'][:200]}...")
            print(f"Notes: {eval_result['notes']}")
            print(f"Sources: {', '.join(eval_result['sources']) if eval_result['sources'] else 'None'}")
            print("-" * 80 + "\n")
        
        # Calculate metrics
        total = len(results)
        accuracy = (scores["✅"] / total) * 100
        
        summary = {
            "total_questions": total,
            "correct": scores["✅"],
            "partial": scores["⚠️"],
            "incorrect": scores["❌"],
            "accuracy_percentage": accuracy,
            "prompt_version": "v1" if use_v1 else "v2",
            "detailed_results": results
        }
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Total Questions: {total}")
        print(f"✅ Correct: {scores['✅']} ({scores['✅']/total*100:.1f}%)")
        print(f"⚠️ Partial: {scores['⚠️']} ({scores['⚠️']/total*100:.1f}%)")
        print(f"❌ Incorrect: {scores['❌']} ({scores['❌']/total*100:.1f}%)")
        print(f"\nOverall Accuracy: {accuracy:.1f}%")
        print("="*80 + "\n")
        
        return summary
    
    def compare_prompts(self):
        """
        Compare performance between prompt v1 and v2.
        """
        print("\n" + "="*80)
        print("PROMPT COMPARISON")
        print("="*80 + "\n")
        
        # Evaluate with both prompts
        results_v1 = self.run_evaluation(use_v1=True)
        results_v2 = self.run_evaluation(use_v1=False)
        
        # Compare
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        print(f"\nPrompt V1 Accuracy: {results_v1['accuracy_percentage']:.1f}%")
        print(f"Prompt V2 Accuracy: {results_v2['accuracy_percentage']:.1f}%")
        print(f"\nImprovement: {results_v2['accuracy_percentage'] - results_v1['accuracy_percentage']:.1f} percentage points")
        print("\nKey Differences:")
        print("- V1: Basic instruction, minimal structure")
        print("- V2: Detailed instructions, structured output, hallucination control")
        print("="*80 + "\n")
        
        return {
            "v1_results": results_v1,
            "v2_results": results_v2
        }


def test_edge_cases(rag_system: PolicyRAGSystem):
    """
    Test edge cases and system robustness.
    """
    print("\n" + "="*80)
    print("EDGE CASE TESTING")
    print("="*80 + "\n")
    
    edge_cases = [
        {
            "name": "Completely off-topic question",
            "question": "What is the capital of France?"
        },
        {
            "name": "Ambiguous question",
            "question": "How long does it take?"
        },
        {
            "name": "Question combining multiple policies",
            "question": "If I cancel my order after shipping, can I get a refund and how long will it take?"
        },
        {
            "name": "Very specific numeric question",
            "question": "Exactly how many business days for standard shipping to Alaska?"
        }
    ]
    
    for case in edge_cases:
        print(f"Edge Case: {case['name']}")
        print(f"Question: {case['question']}")
        
        result = rag_system.answer_question(case['question'])
        
        print(f"Answer: {result['answer']}")
        print(f"Sources: {', '.join(result['sources']) if result['sources'] else 'None'}")
        print(f"Confidence: {result['confidence']}")
        print("-" * 80 + "\n")


def main():
    """
    Main evaluation function.
    """
    # Load API key - Changed to GOOGLE_API_KEY for Gemini
    api_key = os.getenv("GOOGLE_API_KEY", "AIzaSy..........")
    
    if not api_key or api_key == "your-gemini-api-key-here":
        print("ERROR: Set GOOGLE_API_KEY environment variable")
        print("Example: export GOOGLE_API_KEY='AIza...'")
        return
    
    # Initialize RAG system
    print("Initializing RAG system for evaluation...")
    rag = PolicyRAGSystem(gemini_api_key=api_key, data_dir="data")
    rag.build_vector_store()
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag)
    
    # Run main evaluation
    results = evaluator.run_evaluation(use_v1=False)
    
    # Test edge cases
    test_edge_cases(rag)
    
    # Optional: Compare prompts (uncomment to run)
    # evaluator.compare_prompts()
    
    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Evaluation complete! Results saved to evaluation_results.json")


if __name__ == "__main__":
    main()