import torch
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from siamese_model import get_siamese_model
from predict_siamese import get_embedding

def build_embedding_db():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/siamese_hybrid.pth"
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return

    # Load Model
    model = get_siamese_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Image Root
    image_root = Path("super_database")
    db_output = "models/embedding_db.json"
    
    # Store embeddings as { "Person N": [list of embeddings] }
    database = {}
    
    images = sorted(list(image_root.glob("*.JPG")) + list(image_root.glob("*.jpg")))
    print(f"Generating signatures for {len(images)} images...")
    
    for img_path in tqdm(images):
        # Extract label from filename "IMG_NNN (M).JPG"
        stem = img_path.stem
        if "IMG_" in stem:
            try:
                parts = stem.split(" ")
                person_id = parts[0].replace("IMG_", "")
                label = f"Person {int(person_id)}"
            except:
                continue
        else:
            continue
            
        # Get embedding
        try:
            emb = get_embedding(model, img_path, device).cpu().numpy().tolist()[0]
            
            if label not in database:
                database[label] = []
            database[label].append(emb)
        except Exception as e:
            print(f"Skipped {img_path.name}: {e}")

    # Save to JSON
    with open(db_output, "w") as f:
        json.dump(database, f)
        
    print(f"\nEmbedding database saved to {db_output}")
    print(f"Total persons enrolled: {len(database)}")

if __name__ == "__main__":
    build_embedding_db()
