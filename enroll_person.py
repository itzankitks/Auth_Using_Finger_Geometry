#!/usr/bin/env python3
"""Enroll a new person by copying images into an enrollment folder and
extracting feature vectors to append to the dataset CSV.

Usage examples:
  python enroll_person.py --name "John_Doe" --src /path/to/images --retrain
  python enroll_person.py --name Alice --src ./new_images
"""
import argparse
import os
import shutil
import sys
from pathlib import Path
import csv

from feature_extraction import extract_feature_vector


def is_image_file(p: Path):
    return p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def gather_images(src: Path):
    if src.is_dir():
        return [p for p in sorted(src.iterdir()) if p.is_file() and is_image_file(p)]
    if src.is_file() and is_image_file(src):
        return [src]
    # treat src as glob or a comma-separated list
    parts = str(src).split(',')
    paths = []
    for part in parts:
        p = Path(part.strip())
        if p.exists() and is_image_file(p):
            paths.append(p)
    return paths


def append_rows_to_csv(csv_path: Path, rows, header):
    write_header = not csv_path.exists()
    with csv_path.open('a', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    p = argparse.ArgumentParser(description='Enroll a new person')
    p.add_argument('--name', required=True, help='Canonical person identifier (folder name)')
    p.add_argument('--src', required=True, help='Source image file or folder of images to enroll')
    p.add_argument('--enroll-dir', default='enrollments', help='Where to copy/store enroll images')
    p.add_argument('--csv', default='all_features.csv', help='CSV dataset to append features to')
    p.add_argument('--retrain', action='store_true', help='Optional: retrain model after enrollment')
    args = p.parse_args()

    name = args.name.strip()
    src = Path(args.src)
    enroll_root = Path(args.enroll_dir)
    enroll_root.mkdir(parents=True, exist_ok=True)
    dest_dir = enroll_root / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    images = gather_images(src)
    if not images:
        print('No images found at', src)
        sys.exit(2)

    rows = []
    for img in images:
        # copy image into enrollment folder (preserve name)
        dst = dest_dir / img.name
        if not dst.exists():
            shutil.copy2(img, dst)
        fv = extract_feature_vector(str(dst))
        # feature_extraction returns dict; append label column as 'label'
        fv['label'] = name
        rows.append(fv)

    csv_path = Path(args.csv)
    # Determine header ordering: feature keys then 'label'
    header = list(rows[0].keys())
    append_rows_to_csv(csv_path, rows, header)

    print(f'Enrolled {len(rows)} images for "{name}" into {dest_dir}.')
    print(f'Appended {len(rows)} rows to {csv_path}.')

    if args.retrain:
        print('Retraining model now (this runs train_and_evaluate.py)...')
        cmd = [sys.executable, 'train_and_evaluate.py', '--data', str(csv_path)]
        import subprocess
        subprocess.run(cmd)


if __name__ == '__main__':
    main()
