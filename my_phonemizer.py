import json
from pathlib import Path
from phonemizer.backend import EspeakBackend
from tqdm import tqdm

import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description='Phonemize text data.')

    parser.add_argument("--manifests", nargs='+', type=Path, required=True, help="List of input manifest file paths to phonemize.")
    parser.add_argument("--language", type=str, default="it", help="Language code")
    parser.add_argument("--preserve-punctuation", action="store_true", help="Whether to preserve punctuation in phonemized text.")
    parser.add_argument("--output-dir", type=Path, default=".", help="Output directory for phonemized manifest files (default: current directory).")

    args = parser.parse_args()
    return args

def phonemization(manifest, language, preserve_punctuation, output_manifest):
    # you can also consider with_stress=True and add stress symbols into charset of tokenizer for experimental purpose.
    backend = EspeakBackend(language=language, preserve_punctuation=preserve_punctuation)
    print(f"Phonemizing: {manifest}")
    entries = []
    with open(manifest, 'r') as fjson:
        for line in tqdm(fjson):
            # grapheme
            grapheme_dct = json.loads(line.strip())
            grapheme_dct.update({"is_phoneme": 0})
            # phoneme
            phoneme_dct = grapheme_dct.copy()
            # you can also add a separator.Separator(phone="_") to distinguish phone or word boundaries for experimental purpose.
            phonemes = backend.phonemize([grapheme_dct["normalized_text"]], strip=True)
            phoneme_dct["normalized_text"] = phonemes[0]
            phoneme_dct["is_phoneme"] = 1

            entries.append(grapheme_dct)
            entries.append(phoneme_dct)

    with open(output_manifest, "w", encoding="utf-8") as fout:
        for entry in entries:
            fout.write(f"{json.dumps(entry)}\n")
    print(f"Phonemizing is complete: {manifest} --> {output_manifest}")

def main():
    args = get_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for manifest in args.manifests:
        output_manifest = output_dir / f"{manifest.stem}_phonemes{manifest.suffix}"
        phonemization(manifest, args.language, args.preserve_punctuation, output_manifest)

if __name__ == "__main__":
    main()
