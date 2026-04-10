# Finger Geometry Biometric Identification

Finger Geometry is a biometric identification project that uses hand landmark geometry to classify a person from a hand image.

## Project Pipeline

1. Detect 21 hand landmarks using MediaPipe.
2. Compute geometric properties from landmark points.
3. Convert properties to scale-invariant features (aF1, bF1, cF1, dF1, eF1, F2, F3, F4, F5, F6).
4. Build a labeled CSV dataset.
5. Train and evaluate multiple classifiers.
6. Save the best model and run prediction on new images.

## Repository Scripts

- `loopcode.py`: full batch dataset generator that scans every image in `super_database/` and overwrites `all_features.csv`.
- `feature_extraction.py`: reusable feature extraction for single images.
- `train_and_evaluate.py`: trains and compares KNN, SVM, Random Forest.
- `predict_person.py`: predicts person label from a new hand image.

## Setup

Windows (Command Prompt):

1. `python -m venv .venv`
2. `.venv\Scripts\activate`
3. `pip install -r requirements.txt`

Git Bash:

1. `python -m venv .venv`
2. `source .venv/Scripts/activate`
3. `pip install -r requirements.txt`

## Step 1: Generate Feature CSV

Run:

`python loopcode.py`

This writes `all_features.csv` with columns:

`aF1,bF1,cF1,dF1,eF1,F2,F3,F4,F5,F6,Label`

Current `loopcode.py` scans every image file in `super_database/`, so the expected output is 1304 rows across 163 people.

## Step 2: Validate Dataset

Run:

`python -c "import pandas as pd; df=pd.read_csv('all_features.csv'); print('Rows:', len(df)); print('Classes:', df['Label'].nunique()); print('Columns:', df.columns.tolist())"`

## Step 3: Train and Evaluate Models

Run:

`python train_and_evaluate.py --data all_features.csv`

Outputs:

- `models/finger_geometry_model.joblib` (best model + label encoder)
- `models/model_report.json` (CV and holdout metrics)
- `models/classification_report.txt` (per-class precision/recall/F1)
- `models/class_distribution.png` (samples per class)
- `models/confusion_matrix.png` (normalized holdout confusion matrix)

Evaluation protocol:

- 5-fold Stratified Cross Validation
- Holdout split (20% test, stratified)
- Accuracy, Macro-F1, Weighted-F1

## Step 4: Predict for New Image

Run:

`python predict_person.py --image "path/to/new_hand_image.jpg" --model models/finger_geometry_model.joblib`

Output:

- Predicted label (example: Person 123)
- Feature values used for prediction

## Important Notes

- `loopcode.py` now overwrites `all_features.csv` each run, so you do not need to delete it first.
- MediaPipe informational line `Created TensorFlow Lite XNNPACK delegate for CPU` is normal.
- `predict_person.py` now resolves common filename variants (for example `IMG_067 (1).JPG` to `IMG_0067 (1).JPG`).

## Suggested Presentation Flow

1. Problem statement: contactless biometric identification.
2. Landmark detection: show 21 hand keypoints.
3. Feature engineering: explain scale-invariant ratios.
4. Dataset generation: show CSV schema and class distribution.
5. Model comparison: KNN vs SVM vs Random Forest.
6. Live demo: predict label from a new image.
7. Limitations and future improvements.

## 10-Minute Presentation Script (Professor Demo)

1. Introduction (1 minute)
   - "This project identifies a person from a hand image using finger geometry instead of fingerprints."
   - "It is contactless and uses only camera images with landmark-based features."

2. Motivation (1 minute)
   - "Fingerprint scanners need close contact and specialized hardware."
   - "Our approach uses hand shape ratios that are stable under scale changes."

3. Data Collection (1 minute)
   - "We collected 1304 hand images from 163 classes."
   - "Each image is named to encode person ID and sample index."

4. Feature Engineering (2 minutes)
   - "MediaPipe extracts 21 hand landmarks."
   - "From those points, we compute geometric properties and convert them into normalized features aF1..F6."
   - "The feature set is designed to be scale-invariant."

5. Training and Validation (2 minutes)
   - "We train KNN, SVM, and Random Forest using stratified 5-fold cross-validation."
   - "Best model is selected by macro-F1 for balanced multiclass performance."

6. Results (1 minute)
   - "Show model_report.json summary."
   - "Show class_distribution.png and confusion_matrix.png for error analysis."

7. Live Demo (1 minute)
   - Run: `python predict_person.py --image "super_database/IMG_067 (1).JPG" --model models/finger_geometry_model.joblib`
   - Show predicted person and extracted features.

8. Limitations and Future Work (1 minute)
   - "Current setup assumes controlled hand pose and lighting."
   - "Future work: data augmentation, deep metric learning, and deployment as an API/web app."
