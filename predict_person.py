import argparse
import difflib
import re
from pathlib import Path

import joblib
import pandas as pd

from feature_extraction import extract_feature_vector


def resolve_image_path(image_arg: str) -> Path:
    input_path = Path(image_arg)
    if input_path.exists():
        return input_path

    project_root = Path(__file__).resolve().parent
    image_root = project_root / "super_database"
    candidate = image_root / image_arg
    if candidate.exists():
        return candidate

    # Handle different zero-padding and spacing styles in filenames.
    pattern = re.compile(r"IMG_(\d+)\s*\((\d+)\)", re.IGNORECASE)
    match = pattern.search(input_path.stem)
    if match:
        person_id = int(match.group(1))
        image_index = int(match.group(2))

        possible_names = [
            f"IMG_{person_id:03d} ({image_index}).JPG",
            f"IMG_{person_id:03d} ({image_index}).jpg",
            f"IMG_{person_id:04d} ({image_index}).JPG",
            f"IMG_{person_id:04d} ({image_index}).jpg",
            f"IMG_{person_id}({image_index}).JPG",
            f"IMG_{person_id}({image_index}).jpg",
        ]

        for name in possible_names:
            p = image_root / name
            if p.exists():
                return p

        # Fallback: parse each image stem and match by IDs.
        for p in image_root.iterdir():
            if not p.is_file() or p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            m = pattern.search(p.stem)
            if m and int(m.group(1)) == person_id and int(m.group(2)) == image_index:
                return p

    available = [p.name for p in (Path(__file__).resolve().parent / "super_database").iterdir() if p.is_file()]
    hints = difflib.get_close_matches(input_path.name, available, n=5)
    hint_text = "\n".join(f"  - {h}" for h in hints) if hints else "  - (no close matches found)"
    raise FileNotFoundError(
        f"Image not found: {image_arg}\n"
        f"Closest file names:\n{hint_text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict person label from a hand image")
    parser.add_argument("--image", required=True, help="Path to hand image")
    parser.add_argument("--model", default="models/finger_geometry_model.joblib", help="Path to trained model artifact")
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    model = artifact["model"]
    label_encoder = artifact["label_encoder"]
    feature_columns = artifact["feature_columns"]

    resolved_image_path = resolve_image_path(args.image)
    features = extract_feature_vector(str(resolved_image_path))
    X = pd.DataFrame([[features[col] for col in feature_columns]], columns=feature_columns)

    pred_idx = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    print("Image used:", resolved_image_path)
    print("Predicted label:", pred_label)
    print("Features:")
    for col in feature_columns:
        print(f"  {col}: {features[col]:.6f}")


if __name__ == "__main__":
    main()
