# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Download dataset and create manifests.')

    parser.add_argument("--data-root", type=Path, default="/data")
    parser.add_argument("--val-size", default=0.1, type=float)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument(
        "--seed-for-ds-split",
        default=100,
        type=float,
        help="Seed for deterministic split of train/dev/test, NVIDIA's default is 100",
    )

    args = parser.parse_args()
    return args


EXTRACTED_FOLDER = "malavoglia"


def __process_transcript(file_path: str):

    entries = []
    with open(file_path / "metadata.csv", encoding="utf-8") as fin:
        for line in fin:
            wav_file, text = line.strip().split('|') 
            # wav_file, text, speaker_id = line.strip('|')
            wav_file = file_path / "wavs" / (wav_file + ".wav")
            assert os.path.exists(wav_file), f"{wav_file} not found!"
            duration = subprocess.check_output(f"soxi -D {wav_file}", shell=True)
            entry = {
                'audio_filepath': os.path.abspath(wav_file),
                'duration': float(duration),
                'text': text,
                # 'speaker': speaker_id,
            }
            entries.append(entry)
    return entries


def __process_data(dataset_path, val_size, test_size, seed_for_ds_split):
    entries = __process_transcript(dataset_path)

    random.Random(seed_for_ds_split).shuffle(entries)

    train_size = 1.0 - val_size - test_size
    train_entries, validate_entries, test_entries = np.split(
        entries, [int(len(entries) * train_size), int(len(entries) * (train_size + val_size))]
    )

    assert len(train_entries) > 0, "Not enough data for train, val and test"

    def save(p, data):
        with open(p, 'w') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')

    save(dataset_path / "train_manifest_vits.json", train_entries)
    save(dataset_path / "val_manifest_vits.json", validate_entries)
    save(dataset_path / "test_manifest_vits.json", test_entries)


def main():
    args = get_args()
    dataset_root = args.data_root
    dataset_root.mkdir(parents=True, exist_ok=True)
    __process_data(
        dataset_root / EXTRACTED_FOLDER, args.val_size, args.test_size, args.seed_for_ds_split,
    )


if __name__ == "__main__":
    main()
