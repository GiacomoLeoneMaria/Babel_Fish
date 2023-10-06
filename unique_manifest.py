import argparse
import json
import os
import random
from pathlib import Path
import subprocess
import numpy as np

# Define the fields to include in the manifest
MANIFEST_FIELDS = ["audio_filepath", "duration", "text"]

def get_args():
    parser = argparse.ArgumentParser(description='Download dataset and create a single manifest.')

    parser.add_argument("--data-root", type=Path, default="/data")
    parser.add_argument(
        "--seed-for-ds-split",
        default=100,
        type=int,
        help="Seed for deterministic split of train/dev/test, NVIDIA's default is 100",
    )

    args = parser.parse_args()
    return args

EXTRACTED_FOLDER = "il_fu_mattia_pascal"

def __process_transcript(file_path: str):
    entries = []
    with open(file_path / "metadata.csv", encoding="utf-8") as fin:
        for line in fin:
            wav_file, text = line.strip().split('|') 
            wav_file = file_path / "wavs" / (wav_file + ".wav")
            assert os.path.exists(wav_file), f"{wav_file} not found!"
            duration = subprocess.check_output(f"soxi -D {wav_file}", shell=True)
            entry = {
                'audio_filepath': os.path.abspath(wav_file),
                'duration': float(duration),
                'text': text,
            }
            entries.append({key: entry[key] for key in MANIFEST_FIELDS})  # Include only specified fields

    # Save a single manifest file
    manifest_path = file_path / "manifest.json"
    with open(manifest_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

def main():
    args = get_args()
    dataset_root = args.data_root
    dataset_root.mkdir(parents=True, exist_ok=True)
    __process_transcript(
        dataset_root / EXTRACTED_FOLDER
    )

if __name__ == "__main__":
    main()
