import argparse
import torch
import json
import numpy as np
from pathlib import Path

from siamese_model import get_siamese_model
from predict_siamese import get_embedding

def identify():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Hand image to identify")
    parser.add_argument("--db", default="models/embedding_db.json")
    parser.add_argument("--model", default="models/siamese_hybrid.pth")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Database
    if not Path(args.db).exists():
        print(f"Error: Database not found at {args.db}. Run enroll_siamese.py first.")
        return
    
    with open(args.db, "r") as f:
        database = json.load(f)
    
    # Pre-calculate mean embeddings for each person
    person_signatures = {}
    for person, embeddings in database.items():
        person_signatures[person] = np.mean(embeddings, axis=0)

    # 2. Load Model
    model = get_siamese_model(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # 3. Get input embedding
    input_emb = get_embedding(model, args.image, device).cpu().numpy()[0]

    # 4. Compare
    scores = []
    for person, signature in person_signatures.items():
        # Cosine Similarity
        similarity = np.dot(input_emb, signature) / (np.linalg.norm(input_emb) * np.linalg.norm(signature))
        scores.append((person, similarity))
    
    # Sort by similarity
    scores.sort(key=lambda x: x[1], reverse=True)

    # 5. Result
    print(f"\nIdentification Results for: {args.image}")
    print(f"---------------------------------------------")
    print(f"Top Match: {scores[0][0]} (Confidence: {scores[0][1]:.4f})")
    print(f"\nTop {args.top_k} Candidates:")
    for i in range(min(args.top_k, len(scores))):
        person, sim = scores[i]
        bar = "=" * int(sim * 30)
        print(f"  {i+1}. {person:<15} {sim:.4f} {bar}")

if __name__ == "__main__":
    identify()
