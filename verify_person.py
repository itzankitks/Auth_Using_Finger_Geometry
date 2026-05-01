#!/usr/bin/env python3
"""Verify a claimed identity for a new image using the trained model.

Behavior:
- Loads saved model artifact (joblib) which must include at least:
  {'model': <sklearn estimator or Pipeline>, 'label_encoder': <LabelEncoder>, 'feature_columns': [...]}
- Extracts features from the probe image and computes a confidence score for the claimed label.
- If the model exposes `predict_proba` it is used. Otherwise falls back to `decision_function` + softmax.

Usage examples:
  python verify_person.py --image new.jpg --claim "Person_67" --model models/finger_geometry_model.joblib --threshold 0.6
"""
import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from feature_extraction import extract_feature_vector


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def load_artifact(path: Path):
    artifact = joblib.load(path)
    model = artifact.get('model', artifact)
    label_encoder = artifact.get('label_encoder', None)
    feature_columns = artifact.get('feature_columns', None)
    return model, label_encoder, feature_columns


def main():
    p = argparse.ArgumentParser(description='Verify claimed identity for an image')
    p.add_argument('--image', required=True, help='Probe image path')
    p.add_argument('--claim', required=True, help='Claimed person identifier (exact label)')
    p.add_argument('--model', default='models/finger_geometry_model.joblib', help='Saved joblib artifact')
    p.add_argument('--threshold', type=float, default=0.65, help='Accept if confidence >= threshold')
    args = p.parse_args()

    img = Path(args.image)
    if not img.exists():
        print('Image not found:', img)
        sys.exit(2)

    model, label_encoder, feature_columns = load_artifact(Path(args.model))
    if label_encoder is None:
        print('Model artifact is missing label_encoder. Cannot verify reliably.')
        sys.exit(3)

    fv = extract_feature_vector(str(img))
    # ensure ordering
    if feature_columns is None:
        X = pd.DataFrame([fv])
    else:
        X = pd.DataFrame([{k: fv.get(k, None) for k in feature_columns}])

    # compute probabilities / scores
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[0]
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(X)[0]
        # decision_function sometimes returns scalar for binary; ensure array
        if np.isscalar(scores):
            scores = np.asarray([ -scores, scores ])
        probs = softmax(scores)
    else:
        # fallback: use predicted label as 1.0 if matches claim else 0.0
        pred = model.predict(X)[0]
        probs = np.zeros(len(label_encoder.classes_))
        probs[np.where(label_encoder.classes_ == pred)] = 1.0

    classes = list(label_encoder.classes_)
    claim = args.claim
    if claim not in classes:
        print(f'Claim "{claim}" is not enrolled. Known classes: {len(classes)} classes available.')
        sys.exit(4)

    idx = classes.index(claim)
    confidence = float(probs[idx])

    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    pred_conf = float(probs[pred_idx])

    print('Probe image:', img)
    print('Claim:', claim)
    print('Predicted:', pred_label, f'(confidence={pred_conf:.3f})')
    print('Claim confidence:', f'{confidence:.3f}')

    if confidence >= args.threshold and pred_label == claim:
        print('VERIFICATION: ACCEPT')
        sys.exit(0)
    else:
        print('VERIFICATION: REJECT')
        sys.exit(1)


if __name__ == '__main__':
    main()
