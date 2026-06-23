import json
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Config ---
FAQ_PATH = "faq.json"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # handles English + Chinese
THRESHOLD = 0.6

# --- Load model and FAQ ---
model = SentenceTransformer(MODEL_NAME)

with open(FAQ_PATH, "r") as f:
    faq = json.load(f)

# --- Build index ---
# For each FAQ entry, collect all question variants
all_questions = []
question_to_entry = []

for entry in faq:
    variants = [entry["question"]] + entry.get("similar_questions", [])
    for q in variants:
        all_questions.append(q)
        question_to_entry.append(entry)

# Pre-compute embeddings for all questions
faq_embeddings = model.encode(all_questions, normalize_embeddings=True)
# normalize_embeddings=True means cosine similarity = dot product (simpler + faster)

# --- Core function ---
def find_best_match(user_query: str) -> dict:
    query_embedding = model.encode([user_query], normalize_embeddings=True)
    scores = np.dot(faq_embeddings, query_embedding.T).flatten()

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_entry = question_to_entry[best_idx]

    return {
        "matched_question": all_questions[best_idx],
        "score": best_score,
        "entry": best_entry,
        "should_escalate": best_score < THRESHOLD or best_entry.get("requires_human", False)
    }

# --- Quick test ---
if __name__ == "__main__":
    test_queries = [
        "my package hasn't arrived",
        "how do I set up my grill",
        "I want a refund",
    ]
    for q in test_queries:
        result = find_best_match(q)
        print(f"\nQuery: {q}")
        print(f"  Matched: {result['matched_question']}")
        print(f"  Score:   {result['score']:.3f}")
        print(f"  Escalate: {result['should_escalate']}")
        print(f"  Answer:  {result['entry']['answer'][:80]}...")
