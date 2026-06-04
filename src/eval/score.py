import os
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from datasets import Dataset

load_dotenv()

def score_faithfulness(answer: str, contexts: list) -> float:
    """Score how well the answer is grounded in the contexts."""
    if "I cannot find this in the provided documents" in answer:
        return 1.0  # correct refusal is perfectly faithful

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    context_text = "\n".join(contexts)

    prompt = f"""Given the following context and answer, score how faithful the answer is to the context.
A faithful answer only contains information from the context.

Context:
{context_text[:3000]}

Answer:
{answer[:1000]}

Score from 0.0 to 1.0 where:
1.0 = answer only uses information from context
0.5 = answer partially uses context but adds outside info
0.0 = answer ignores context completely

Respond with ONLY a number between 0.0 and 1.0"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        score = float(response.choices[0].message.content.strip())
        return min(max(score, 0.0), 1.0)
    except Exception:
        return 0.5

def score_relevancy(question: str, answer: str) -> float:
    """Score how relevant the answer is to the question."""
    if "I cannot find this in the provided documents" in answer:
        return 0.5  # neutral — correct but not informative

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""Score how relevant this answer is to the question.

Question: {question}

Answer: {answer[:1000]}

Score from 0.0 to 1.0 where:
1.0 = answer directly and completely addresses the question
0.5 = answer partially addresses the question
0.0 = answer is completely irrelevant to the question

Respond with ONLY a number between 0.0 and 1.0"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        score = float(response.choices[0].message.content.strip())
        return min(max(score, 0.0), 1.0)
    except Exception:
        return 0.5

def score_context_precision(question: str, contexts: list) -> float:
    """Score how precise the retrieved contexts are for the question."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    context_text = "\n---\n".join(contexts[:3])

    prompt = f"""Score how relevant the retrieved context is for answering the question.

Question: {question}

Retrieved Context:
{context_text[:3000]}

Score from 0.0 to 1.0 where:
1.0 = context is highly relevant and contains the answer
0.5 = context is partially relevant
0.0 = context is completely irrelevant

Respond with ONLY a number between 0.0 and 1.0"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        score = float(response.choices[0].message.content.strip())
        return min(max(score, 0.0), 1.0)
    except Exception:
        return 0.5

def run_scoring():
    print("Loading saved results...")
    results = json.loads(
        Path("data/eval_results.json").read_text(encoding="utf-8")
    )

    questions = results["questions"]
    answers = results["answers"]
    contexts = results["contexts"]
    n = len(questions)

    print(f"Scoring {n} QA pairs...")
    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []

    for i, (q, a, c) in enumerate(zip(questions, answers, contexts)):
        print(f"  [{i+1}/{n}] scoring...")
        faithfulness_scores.append(score_faithfulness(a, c))
        relevancy_scores.append(score_relevancy(q, a))
        precision_scores.append(score_context_precision(q, c))

    avg_faithfulness = sum(faithfulness_scores) / n
    avg_relevancy = sum(relevancy_scores) / n
    avg_precision = sum(precision_scores) / n

    print("\n" + "="*50)
    print("EVAL RESULTS")
    print("="*50)
    print(f"Faithfulness:      {avg_faithfulness:.4f}")
    print(f"Answer Relevancy:  {avg_relevancy:.4f}")
    print(f"Context Precision: {avg_precision:.4f}")
    print("="*50)

    scores = {
        "faithfulness": round(avg_faithfulness, 4),
        "answer_relevancy": round(avg_relevancy, 4),
        "context_precision": round(avg_precision, 4),
        "n_questions": n
    }

    Path("data/eval_scores.json").write_text(
        json.dumps(scores, indent=2),
        encoding="utf-8"
    )
    print("Scores saved to data/eval_scores.json")

if __name__ == "__main__":
    run_scoring()