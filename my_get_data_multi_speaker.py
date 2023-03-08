import argparse
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
# https://github.com/NVIDIA/NeMo-text-processing/blob/main/nemo_text_processing/text_normalization/normalize.py
from nemo_text_processing.text_normalization.normalize import Normalizer

# specify multiple data root directories as command-line arguments, separated by spaces.
def get_args():
    parser = argparse.ArgumentParser(description='Importing the Italian dataset and creating the manifest')

    parser.add_argument("--data-root", type=Path, default="/data")
    #parser.add_argument("--data_root", nargs="+", type=Path, default=["/data"])
    parser.add_argument("--val-size", default=0.1, type=float)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument(
        "--seed-for-ds-split",
        default=100,
        type=float,)
    parser.add_argument("--output-dir", type=Path, default=".", help="Output directory for manifest files (default: current directory).")

    args = parser.parse_args()
    return args

def __process_transcript(file_path: str):
    # Create normalizer
    text_normalizer = Normalizer(
        lang="en", input_case="cased", overwrite_cache=True, cache_dir=str(file_path / "cache_dir_multispeakers"),
    )
    text_normalizer_call_kwargs = {"punct_pre_process": True, "punct_post_process": True}
    normalizer_call = lambda x: text_normalizer.normalize(x, **text_normalizer_call_kwargs)
    entries = []
    with open(file_path / "metadata.csv", encoding="utf-8") as fin:
        for line in fin:
            #wav_file, text, speaker_id = line.strip().split('|')
            wav_file, text = line.strip().split('|')
            wav_file = file_path / "wavs" / (wav_file + ".wav")
            assert os.path.exists(wav_file), f"{wav_file} not found!"
            duration = subprocess.check_output(f"soxi -D {wav_file}", shell=True)
            normalized_text = normalizer_call(text)
            entry = {
                'audio_filepath': os.path.abspath(wav_file),
                'duration': float(duration),
                'text': text,
                #'speaker': speaker_id,
                'normalized_text': normalized_text,
            }
            entries.append(entry)
    return entries

def __process_data(dataset_paths, val_size, test_size, seed_for_ds_split):
    all_entries = []
    for dataset_path in dataset_paths:
        entries = __process_transcript(dataset_path)
        all_entries.extend(entries)

    random.Random(seed_for_ds_split).shuffle(all_entries)

    train_size = 1.0 - val_size - test_size
    train_entries, validate_entries, test_entries = np.split(
        all_entries, [int(len(all_entries) * train_size), int(len(all_entries) * (train_size + val_size))]
    )

    assert len(train_entries)>0, "Not enough data for train, val and test"

    def save(p, data):
        with open(p, 'w') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')

    args = get_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    save(output_dir / "trainvaltest_manifest.json", all_entries)
    save(output_dir / "train_manifest.json", train_entries)
    save(output_dir / "val_manifest.json", validate_entries)
    save(output_dir / "test_manifest.json", test_entries)

FOLDERS = ['la_contessa_di_karolystria', 'le_meraviglie_del_duemila']

def main():
    args = get_args()
    dataset_paths = [args.data_root / folder for folder in FOLDERS]
    __process_data(
        dataset_paths, args.val_size, args.test_size, args.seed_for_ds_split,
    )

if __name__ == "__main__":
    main()
