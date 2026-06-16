import json
import os
import numpy as np
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("CRITICAL ERROR: GOOGLE_API_KEY not found. ")

client = genai.Client(api_key=api_key)


# Define Embedding Function
def get_embedding(text: str) -> np.ndarray:
    """
    Generates a 768-dimensional vector embedding using Gemini's model.
    """
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return np.array(response.embeddings[0].values)

# Define Cosine Similarity by Hand

def cosine_similarity(v1, v2):
    """
        Computes the cosine similarity between two vectors using NumPy.
        Formula: (A · B) / (||A|| * ||B||)
        """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    return float(dot_product / (norm_v1 * norm_v2))


# Load Data & Generate Corpus Embeddings
with open("knowledge_base.json", "r") as file:
    passages = json.load(file)

embedded_corpus = []
for p in passages:
    vector = get_embedding(p["text"])

    embedded_corpus.append({
        "id": p["id"],
        "source": p["source"],
        "text": p["text"],
        "embedding": vector
    })


# Querying & Ranking
test_queries = [
    "my laptop won't switch on",
    "how do I stop being billed every month?",
    "access denied error when saving a file",
    "where do I leave my car in the evening?",
    "what's the wifi password?"
]

for query in test_queries:
    print(f"\nResults for: {query}")
    query_vector = get_embedding(query)

    scored_passages = []
    for item in embedded_corpus:
        score = cosine_similarity(query_vector, item["embedding"])
        scored_passages.append((score, item))

    scored_passages.sort(key=lambda x: x[0], reverse=True)

    print("Top 3 matches:")
    for rank, (score, item) in enumerate(scored_passages[:3], 1):
        print(f"Rank {rank} | Score: {score:.4f} | ID: {item['id']} | Source: {item['source']}")
        print(f"Text: {item['text']}\n")


"""
1. WORD OVERLAP ANALYSIS PER QUERY:

* Query: "my laptop won't switch on"
  - Best Match (Rank 1): "To power up a device that won't turn on..."
  - Word Overlap: Virtually none. The query uses "laptop" and "switch on", 
    while the passage uses "device" and "power up" / "turn on". Despite 
    using completely different vocabulary, it surfaced as the top result.

* Query: "how do I stop being billed every month?"
  - Best Match (Rank 1): "To cancel your subscription, open Account Settings..."
  - Word Overlap: None of the core conceptual words overlap. The query asks 
    about stopping a monthly bill, whereas the document discusses "cancel 
    your subscription" and "End Plan". It correctly identified the financial intent.

* Query: "access denied error when saving a file"
  - Best Match (Rank 1): "The error code 0x80070005 means 'access denied'..."
  - Word Overlap: This specific pair contains exact string phrases ("access denied"). 
    Because of both exact semantic match and exact word match, it returned a 
    higher confidence score (0.7584) than the other queries.

* Query: "where do I leave my car in the evening?"
  - Best Match (Rank 1): "Employees may park in lot B after 6pm on weekdays..."
  - Word Overlap: Zero exact word matches for the core intent. The query 
    mentions "leave my car" and "evening", while the document uses "park", 
    "lot B", and "after 6pm". The model successfully paired these concepts.


2. WHAT THE EMBEDDING CAPTURED:

These results clearly demonstrate that the embedding model captures semantic 
meaning, contextual intent, and conceptual relationships, rather than performing 
simple character-by-character keyword matching. 

When text is processed by an embedding model, it transforms sentences into high-
dimensional vectors where synonyms, related intents, and parallel concepts are 
mapped close to one another in vector space. 

Because our custom numpy 'cosine_similarity' function calculates the geometric angle 
between these vectors, it can identify when two sentences point in the exact same 
conceptual direction. This is why the correct answers smoothly rose to Rank 1 
across all test cases, even when the query and the text shared zero literal vocabulary.


* Uncovered Query: "what's the wifi password?"
  - Observed Top Score: 0.5903 (Rank 1 Match: Parking lot B handbook info)
  - Analysis: The top score is quite low compared to our successful semantic 
    matches (which ranged from ~0.66 to 0.75). Because there is no actual 
    Wi-Fi answer in the knowledge base, the model returns a weak match 
    simply because of the loose connection between "password" and "pass from reception". 
    Rank 2 also drags in a "forgot password" text solely on keyword matching.

* Implementing a Similarity Threshold:
  By establishing a strict mathematical similarity threshold of 0.62, we can intercept 
  the results before they print; since the top score of 0.5903 falls below this line, 
  the system can safely conclude "we don't actually have an answer for this" and gracefully 
  refuse to display completely irrelevant information to the user.
"""